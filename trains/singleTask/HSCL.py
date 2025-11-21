import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# 你原有的工具函数与依赖
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss

# 可视化与数据处理依赖
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger('MMSA')

# ------------------------ 各种损失函数 ------------------------
class SemanticConsistencyLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(SemanticConsistencyLoss, self).__init__()
        self.margin = margin

    def forward(self, data):
        feature_num, feature_dim = data.size()
        # 计算所有特征向量对的余弦相似度
        similarity_matrix = F.cosine_similarity(data.unsqueeze(1), data.unsqueeze(0), dim=2)
        # 排除自身与自身的比较
        mask = torch.eye(feature_num, device=data.device).bool()
        similarity_matrix.masked_fill_(mask, 0)
        # 引入边缘损失
        margin_loss = F.relu(self.margin - similarity_matrix)
        # 计算平均相似度
        avg_similarity = margin_loss.sum() / (feature_num * (feature_num - 1) * feature_dim)
        total_loss = avg_similarity
        return total_loss

class geometric_consistency_loss_new(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2):
        super(geometric_consistency_loss_new, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, vision, language, audio, vision_restruct, language_restruct, audio_restruct):
        batch_size = vision.size(0)
        structural_modal_loss = ((vision - audio).square().mean() + 
                                 (language - vision).square().mean() + 
                                 (language - audio).square().mean()) / batch_size
        
        logits_language_per_vision = language @ vision.t()
        logits_vision_per_language = vision @ language.t()
        logits_language_per_audio = language @ audio.t()
        logits_audio_per_language = audio @ language.t()
        logits_audio_per_vision = audio @ vision.t()
        logits_vision_per_audio = vision @ audio.t()
        cross_modal_loss = ((logits_language_per_vision - logits_vision_per_language).square().mean() + 
                            (logits_audio_per_language - logits_language_per_audio).square().mean() + 
                            (logits_vision_per_audio - logits_audio_per_vision).square().mean()) / batch_size

        structural_modal_loss_restruct = ((vision_restruct - audio_restruct).square().mean() + 
                                          (language_restruct - vision_restruct).square().mean() + 
                                          (language_restruct - audio_restruct).square().mean()) / batch_size
        structural_modal_loss_origin_restruct = ((vision - vision_restruct).square().mean() + 
                                                 (language - language_restruct).square().mean() + 
                                                 (audio - audio_restruct).square().mean()) / batch_size

        logits_restruct_language_per_vision = language_restruct @ vision_restruct.t()
        logits_restruct_vision_per_language = vision_restruct @ language_restruct.t()
        logits_restruct_language_per_audio = language_restruct @ audio_restruct.t()
        logits_restruct_audio_per_language = audio_restruct @ language_restruct.t()
        logits_restruct_audio_per_vision = audio_restruct @ vision_restruct.t()
        logits_restruct_vision_per_audio = vision_restruct @ audio_restruct.t()
        restruct_cross_modal_loss = ((logits_restruct_language_per_vision - logits_restruct_vision_per_language).square().mean() + 
                                     (logits_restruct_audio_per_language - logits_restruct_language_per_audio).square().mean() + 
                                     (logits_restruct_vision_per_audio - logits_restruct_audio_per_vision).square().mean()) / batch_size

        gc_total_loss_new = (structural_modal_loss + restruct_cross_modal_loss + structural_modal_loss_origin_restruct) * self.alpha + \
                            (structural_modal_loss_restruct + cross_modal_loss) * self.beta
        return gc_total_loss_new

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

# ------------------------ 主要训练/测试类 ------------------------
class HSCL():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()
        self.geometric_consistency_loss_new = geometric_consistency_loss_new()
        self.SemanticConsistencyLoss = SemanticConsistencyLoss()
        self.E = 0

    # ---------- 新增：对所有三种模态（Text, Audio, Visual）预解耦与解耦后状态对比可视化函数 ----------
    def visualize_pre_and_post_decoupling_features_all(self, model, dataloader, device, epoch, save_directory="./visualizations"):
        """
        对文本、音频、视觉三个模态分别进行预解耦（原始特征 o_x）与解耦后状态（模态相关 s_x 与模态无关 c_x）的对比可视化：
          - 对于每个模态，预解耦状态直接使用模型输出的原始特征 o_x (shape: [batch, feat_dim])；
          - 解耦后状态分别对 s_x 和 c_x（原始尺寸为 [seq_len, batch, feat_dim]）进行均值池化得到 [batch, feat_dim]；
          - 对每个模态分别使用 PCA 降维到二维空间，并绘制两幅图：预解耦状态与解耦后状态（后者使用不同颜色区分 s_x 与 c_x）。
        总共生成 6 幅图，并在日志中输出各模态解耦后状态的定量指标。
        """
        modalities = {"l": "Text", "a": "Audio", "v": "Visual"}
        for mod_key, mod_name in modalities.items():
            pre_features_list = []   # 存放预解耦特征：o_{mod}
            post_s_list = []         # 存放解耦后模态相关特征：s_{mod}
            post_c_list = []         # 存放解耦后模态无关特征：c_{mod}
            
            with torch.no_grad():
                for batch in dataloader:
                    text = batch['text'].to(device)
                    audio = batch['audio'].to(device)
                    vision = batch['vision'].to(device)
                    output = model(text, audio, vision, is_distill=False)
                    
                    # 预解耦特征：o_{mod} 已为 [batch, feat_dim]
                    pre_feat = output["o_" + mod_key]
                    pre_features_list.append(pre_feat.cpu().numpy())
                    
                    # 解耦后特征：s_{mod} 与 c_{mod} 原始尺寸 [seq_len, batch, feat_dim] → 均值池化后为 [batch, feat_dim]
                    s_feat = output["s_" + mod_key].permute(1, 0, 2).mean(dim=1)
                    c_feat = output["c_" + mod_key].permute(1, 0, 2).mean(dim=1)
                    post_s_list.append(s_feat.cpu().numpy())
                    post_c_list.append(c_feat.cpu().numpy())
            
            pre_features_all = np.concatenate(pre_features_list, axis=0)
            post_s_all = np.concatenate(post_s_list, axis=0)
            post_c_all = np.concatenate(post_c_list, axis=0)
            
            # ---------- 预解耦状态可视化（PCA） ----------
            pca = PCA(n_components=2)
            pre_pca_2d = pca.fit_transform(pre_features_all)
            plt.figure(figsize=(8, 6))
            plt.scatter(pre_pca_2d[:, 0], pre_pca_2d[:, 1], c='green', alpha=0.6)
            plt.title(f"Pre-Decoupling {mod_name} Features (PCA) Epoch {epoch}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            os.makedirs(save_directory, exist_ok=True)
            pre_path = os.path.join(save_directory, f"pre_decoupling_{mod_key}_epoch_{epoch}.png")
            plt.savefig(pre_path)
            plt.close()
            
            # ---------- 解耦后状态可视化（PCA） ----------
            combined_post = np.concatenate([post_s_all, post_c_all], axis=0)
            labels = np.concatenate([np.zeros(len(post_s_all)), np.ones(len(post_c_all))], axis=0)  # 0: modality-specific, 1: modality-invariant
            pca_post = PCA(n_components=2)
            post_pca_2d = pca_post.fit_transform(combined_post)
            plt.figure(figsize=(8, 6))
            plt.scatter(post_pca_2d[labels == 0, 0], post_pca_2d[labels == 0, 1], c='red', alpha=0.6, label='Modality-Specific')
            plt.scatter(post_pca_2d[labels == 1, 0], post_pca_2d[labels == 1, 1], c='blue', alpha=0.6, label='Modality-Invariant')
            plt.title(f"Post-Decoupling {mod_name} Features (PCA) Epoch {epoch}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.legend()
            post_path = os.path.join(save_directory, f"post_decoupling_{mod_key}_epoch_{epoch}.png")
            plt.savefig(post_path)
            plt.close()
            
            # ---------- 定量指标：计算解耦后状态的平均余弦相似度和欧式距离 ----------
            s_norm = post_s_all / np.linalg.norm(post_s_all, axis=1, keepdims=True)
            c_norm = post_c_all / np.linalg.norm(post_c_all, axis=1, keepdims=True)
            cos_sims = np.sum(s_norm * c_norm, axis=1)
            avg_cos_sim = np.mean(cos_sims)
            euc_dists = np.linalg.norm(post_s_all - post_c_all, axis=1)
            avg_euc_dist = np.mean(euc_dists)
            logger.info(f"[Post-Decoupling {mod_name}] Epoch {epoch}: Avg Cosine Sim: {avg_cos_sim:.4f}, Avg Euclidean Dist: {avg_euc_dist:.4f}")

    # ---------- 原有的解耦后可视化函数（仅用于单模态，可保留备用） ----------
    def visualize_decoupled_features(self, model, dataloader, device, epoch, save_directory="./visualizations"):
        model.eval()
        all_s_l = []
        all_c_l = []
        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                vision = batch['vision'].to(device)
                output = model(text, audio, vision, is_distill=False)
                s_l = output['s_l']
                c_l = output['c_l']
                s_l_agg = s_l.permute(1, 0, 2).mean(dim=1)
                c_l_agg = c_l.permute(1, 0, 2).mean(dim=1)
                all_s_l.append(s_l_agg.cpu().numpy())
                all_c_l.append(c_l_agg.cpu().numpy())
        all_s_l = np.concatenate(all_s_l, axis=0)
        all_c_l = np.concatenate(all_c_l, axis=0)
        s_norm = all_s_l / np.linalg.norm(all_s_l, axis=1, keepdims=True)
        c_norm = all_c_l / np.linalg.norm(all_c_l, axis=1, keepdims=True)
        cos_sims = np.sum(s_norm * c_norm, axis=1)
        avg_cos_sim = np.mean(cos_sims)
        euc_dists = np.linalg.norm(all_s_l - all_c_l, axis=1)
        avg_euc_dist = np.mean(euc_dists)
        logger.info(f"[Decoupling Visualization] Epoch {epoch}:")
        logger.info(f"  >> Avg Cosine Similarity (s_l vs c_l): {avg_cos_sim:.4f} (closer to 0 is better)")
        logger.info(f"  >> Avg Euclidean Distance (s_l vs c_l): {avg_euc_dist:.4f} (larger is better)")

        pca = PCA(n_components=2)
        combined_data = np.concatenate([all_s_l, all_c_l], axis=0)
        pca_2d = pca.fit_transform(combined_data)
        s_l_2d = pca_2d[: len(all_s_l)]
        c_l_2d = pca_2d[len(all_s_l):]

        plt.figure(figsize=(8, 6))
        plt.scatter(s_l_2d[:, 0], s_l_2d[:, 1], color='red', alpha=0.6, label='Modality-Specific')
        plt.scatter(c_l_2d[:, 0], c_l_2d[:, 1], color='blue', alpha=0.6, label='Modality-Invariant')
        plt.title(f"[PCA] Decoupled Features (Epoch {epoch})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        os.makedirs(save_directory, exist_ok=True)
        pca_path = os.path.join(save_directory, f"decoupled_pca_epoch_{epoch}.png")
        plt.savefig(pca_path)
        plt.close()

    # ---------- 训练主循环 ----------
    def do_train(self, model, dataloader, return_epoch_results=False):
        params = list(model[0].parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)
        
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_hscl = model[0]  # hscl model
        net.append(net_hscl)
        model = net

        while True:
            epochs += 1
            self.E += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model[0](text, audio, vision, is_distill=False)

                    # 任务损失
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    loss_task_l_homo = self.criterion(output['logits_l_homo'], labels)
                    loss_task_v_homo = self.criterion(output['logits_v_homo'], labels)
                    loss_task_a_homo = self.criterion(output['logits_a_homo'], labels)
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    loss_task = (loss_task_all + loss_task_l_homo + loss_task_v_homo + loss_task_a_homo + 
                                 loss_task_l_hetero + loss_task_v_hetero + loss_task_a_hetero + loss_task_c)

                    # 正交/解耦损失
                    output['s_l'] = output['s_l'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    output['c_l'] = output['c_l'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    output['s_v'] = output['s_v'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    output['c_v'] = output['c_v'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    output['s_a'] = output['s_a'].transpose(0, 1).contiguous().view(labels.size(0), -1)
                    output['c_a'] = output['c_a'].transpose(0, 1).contiguous().view(labels.size(0), -1)

                    cosine_similarity_s_c_l = self.cosine(output['s_l'], output['c_l'], torch.tensor([-1]).to(self.args.device)).mean(0)
                    cosine_similarity_s_c_v = self.cosine(output['s_v'], output['c_v'], torch.tensor([-1]).to(self.args.device)).mean(0)
                    cosine_similarity_s_c_a = self.cosine(output['s_a'], output['c_a'], torch.tensor([-1]).to(self.args.device)).mean(0)
                    loss_ort = (cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a)

                    # margin loss
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)
                    
                    # 语义一致性损失与几何一致性损失
                    orignsemanticLoss = (self.SemanticConsistencyLoss(output['o_v']) + 
                                          self.SemanticConsistencyLoss(output['o_a']) + 
                                          self.SemanticConsistencyLoss(output['o_l']))
                    reconstructsemanticLoss = (self.SemanticConsistencyLoss(output['r_v']) + 
                                               self.SemanticConsistencyLoss(output['r_a']) + 
                                               self.SemanticConsistencyLoss(output['r_l']))
                    gc_total_loss_new = self.geometric_consistency_loss_new(
                        output['o_v'], output['o_l'], output['o_a'], 
                        output['r_v'], output['r_l'], output['r_a']
                    )

                    loss_args = [0, 0, 0, 0, 0]
                    if self.args.dataset_name == 'mosi' and self.args.need_data_aligned:
                        loss_args = [1, 0.001, 0.001, 0.00001, 0.00001]
                    elif self.args.dataset_name == 'mosi' and not self.args.need_data_aligned:
                        loss_args = [0.1, 0.001, 0.001, 0.001, 0.001]
                    elif self.args.dataset_name == 'mosei' and self.args.need_data_aligned:
                        loss_args = [0.001, 0.0001, 0.0001, 0.001, 0.001]
                    elif self.args.dataset_name == 'mosei' and not self.args.need_data_aligned:
                        loss_args = [1, 0.001, 0.001, 0.01, 0.01]

                    combined_loss = (loss_task +
                                     gc_total_loss_new * loss_args[0] +
                                     loss_sim * loss_args[1] +
                                     loss_ort * loss_args[2] +
                                     orignsemanticLoss * loss_args[3] +
                                     reconstructsemanticLoss * loss_args[4])
                    
                    combined_loss.backward()
                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            # 每 5 个 epoch 调用新增的预解耦与解耦后对比可视化函数（针对 Text, Audio, Visual 三模态，共 6 个图）
            if epochs % 5 == 0:
                self.visualize_pre_and_post_decoupling_features_all(
                    model[0],
                    dataloader['valid'],
                    device=self.args.device,
                    epoch=epochs,
                    save_directory="./visualizations"
                )

            # 验证
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            torch.save(model[0].state_dict(), './pt/' + str(epochs) + '.pth')

            isBetter = (cur_valid <= (best_valid - 1e-6)) if min_or_max == 'min' else (cur_valid >= (best_valid + 1e-6))
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/hscl.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    # ---------- 测试函数 ----------
    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision, is_distill=True)
                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
