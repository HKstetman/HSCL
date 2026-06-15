# Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition

<div align="center">

**HSCL: Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition**

[Paper](https://doi.org/10.1109/TMM.2026.3694526) | [IEEE Xplore](https://doi.org/10.1109/TMM.2026.3694526)

</div>

## Overview

This repository provides the official implementation of **Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition (HSCL)**, published in *IEEE Transactions on Multimedia*.

HSCL addresses two important challenges in multimodal emotion recognition:

1. **Insufficient intra-modal interaction**, which limits the model's ability to capture fine-grained emotional information within each modality.
2. **Modality asynchrony**, which causes language, audio, and visual signals to express emotional information at different temporal rhythms.

The method combines feature decoupling, attention-based multimodal fusion, and hierarchical structure consistency learning. The consistency constraints are applied to both the original features and the reconstructed features.

## Contents

- [Method](#method)
- [Main Results](#main-results)
- [Ablation Studies](#ablation-studies)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Notice](#usage-notice)
- [Citation](#citation)

## Method

### Overall Framework

HSCL first extracts language, audio, and visual representations, then separates each modality into modality-exclusive and modality-irrelevant components. The resulting representations are processed by cross-modal and self-attention modules for emotion prediction.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3af53001-2fc5-435f-9c82-d932a88c4aa1" alt="HSCL framework" width="900" />
</p>

### Hierarchical Structure Consistency

HSCL introduces consistency constraints at three levels:

| Level | Consistency Type | Scope | Main Purpose |
|:--|:--|:--|:--|
| Feature level | Semantic consistency | Features within the same modality | Strengthen intra-modal interaction and feature aggregation |
| Modality level | Representation consistency | Different modalities of the same sample | Align modality representations and improve modality synchrony |
| Sample level | Geometric consistency | Different samples across mismatched modalities | Preserve cross-sample geometry and reduce modality discrepancy |

These constraints are applied to both the original multimodal features and the reconstructed features obtained from the decoupled representations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/0fd91190-6925-4770-a4ea-5bf73a791927" alt="Hierarchical structure consistency" width="900" />
</p>

## Main Results

HSCL is evaluated on **CMU-MOSI**, **CMU-MOSEI**, and **CH-SIMS** under aligned and unaligned settings. Accuracy and F1 values are reported as percentages. Higher values are better for accuracy, F1, and correlation, while lower values are better for MAE.

### CMU-MOSI and CMU-MOSEI

| Dataset | Setting | ACC2 ↑ | ACC5 ↑ | ACC7 ↑ | F1 ↑ | MAE ↓ | Corr ↑ |
|:--|:--|--:|--:|--:|--:|--:|--:|
| CMU-MOSI | Unaligned | 85.2 | 52.4 | 46.9 | 85.2 | 0.716 | **0.806** |
| CMU-MOSI | Aligned | 85.5 | 51.5 | 46.5 | **85.6** | **0.712** | 0.785 |
| CMU-MOSEI | Unaligned | **85.6** | **55.3** | **53.7** | **85.4** | 0.541 | 0.766 |
| CMU-MOSEI | Aligned | 85.4 | 54.5 | 53.1 | **85.4** | **0.538** | **0.769** |

### CH-SIMS

| Dataset | Setting | ACC2 ↑ | ACC3 ↑ | ACC5 ↑ | F1 ↑ | MAE ↓ | Corr ↑ |
|:--|:--|--:|--:|--:|--:|--:|--:|
| CH-SIMS | Unaligned | **80.5** | **65.2** | **43.5** | **80.5** | **0.415** | **0.609** |

The complete comparisons with all baselines are available in the paper.

## Ablation Studies

### Effect of Hierarchical Structure Consistency

The following table compares the basic multimodal emotion recognition network with the complete HSCL model.

| Dataset | Setting | Model | ACC2 ↑ | F1 ↑ | MAE ↓ |
|:--|:--|:--|--:|--:|--:|
| CMU-MOSI | Unaligned | MER Network | 82.2 | 82.1 | 0.775 |
| CMU-MOSI | Unaligned | **HSCL** | **85.2** | **85.2** | **0.716** |
| CMU-MOSEI | Unaligned | MER Network | 84.5 | 84.5 | 0.552 |
| CMU-MOSEI | Unaligned | **HSCL** | **85.6** | **85.4** | **0.541** |
| CH-SIMS | Unaligned | MER Network | 77.7 | 77.6 | 0.435 |
| CH-SIMS | Unaligned | **HSCL** | **80.5** | **80.5** | **0.415** |
| CMU-MOSI | Aligned | MER Network | 84.4 | 84.3 | 0.749 |
| CMU-MOSI | Aligned | **HSCL** | **85.5** | **85.6** | **0.712** |
| CMU-MOSEI | Aligned | MER Network | 85.1 | 85.1 | 0.540 |
| CMU-MOSEI | Aligned | **HSCL** | **85.4** | **85.3** | **0.538** |

### Unimodal Performance on CMU-MOSEI

| Available Modality | Model | ACC2 ↑ | F1 ↑ | MAE ↓ |
|:--|:--|--:|--:|--:|
| Language only | MER Network | 80.2 | 80.6 | 0.570 |
| Language only | **HSCL** | **84.6** | **84.6** | **0.559** |
| Visual only | MER Network | 40.8 | 23.8 | 0.889 |
| Visual only | **HSCL** | **62.9** | **48.5** | **0.833** |
| Audio only | MER Network | 37.2 | 20.1 | 0.856 |
| Audio only | **HSCL** | **62.8** | **48.7** | **0.822** |
| Mean | MER Network | 52.7 | 41.5 | 0.777 |
| Mean | **HSCL** | **70.1** | **60.6** | **0.738** |
| Standard deviation | MER Network | 19.5 | 27.7 | 0.143 |
| Standard deviation | **HSCL** | **10.3** | **17.0** | **0.127** |

### Consistency Weights

| Consistency Term | Symbol | Selected Value |
|:--|:--:|--:|
| Semantic consistency | α | 0.10 |
| Representation consistency | β | 0.02 |
| Geometric consistency | γ | 0.08 |

### Feature Visualization

The PCA visualization shows the distributions of the original and decoupled modality representations.

<img width="1051" height="515" alt="image" src="https://github.com/user-attachments/assets/1d83ca69-e7bf-4792-b532-f4fd21339f0c" />


### Hyperparameter Sensitivity

The sensitivity analysis evaluates the effects of the three consistency weights on the main evaluation metrics.

<img width="1094" height="587" alt="image" src="https://github.com/user-attachments/assets/2c84ec4d-0db2-49f6-833c-71f0ad9cbfb2" />


## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/HKstetman/HSCL.git
cd HSCL
```

### 2. Create the Environment

Python 3.9 is recommended.

```bash
conda create -n hscl python=3.9 -y
conda activate hscl
```

Install a PyTorch build compatible with your CUDA environment, then install the remaining dependencies. The reference experiments used PyTorch `1.10.1+cu111` and Transformers `4.29.2`.

```bash
pip install transformers==4.29.2 numpy pandas scikit-learn easydict
```

### 3. Prepare the Datasets

The code supports the following dataset identifiers:

| Dataset | Identifier Used in the Code |
|:--|:--|
| CMU-MOSI | `mosi` |
| CMU-MOSEI | `mosei` |
| CH-SIMS | `sims` |

Download the corresponding preprocessed multimodal features and update the dataset paths in:

```text
config/config.json
```

In particular, replace the placeholder value of `dataset_root_dir` and verify the aligned or unaligned feature paths for each dataset.

### 4. Train HSCL

Set `dataset_name` in `train.py` to one of `mosi`, `mosei`, or `sims`, then run:

```bash
python train.py
```

By default, checkpoints, experiment results, and logs are written to:

```text
./pt
./result
./log
```

### 5. Evaluate HSCL

Set `dataset_name` in `test.py` and run:

```bash
python test.py
```

> [!IMPORTANT]
> Ensure that the checkpoint path used in test mode in `run.py` points to the model checkpoint you want to evaluate. In the current code, this path may need to be updated manually before testing.

## Project Structure

```text
HSCL/
├── config/          # Dataset and model configurations
├── result/          # Saved experimental results
├── trains/          # Training and evaluation implementations
├── utils/           # Utility functions
├── config.py        # Configuration loader
├── data_loader.py   # Dataset loading and preprocessing
├── run.py           # Main experiment controller
├── train.py         # Training entry point
├── test.py          # Testing entry point
└── README.md
```

## Usage Notice

This repository is provided for academic research. Please cite the associated paper when using the code, method, or experimental results. Please also respect the authorship and applicable publication and redistribution terms.

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{yang2026hierarchical,
  title     = {Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition},
  author    = {Yang, Boan and Shi, Qinghongya and Zong, Xiaofen and Ye, Mang},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/TMM.2026.3694526}
}
```
