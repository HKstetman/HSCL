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

### Table I. Comparison on CMU-MOSI

<p align="center"><strong>Comparison of results on the CMU-MOSI dataset.</strong><br>
<em>* indicates that the method uses BERT-based features. Bold denotes the best result, and underlining denotes the second-best result. ↑ means higher is better, and ↓ means lower is better.</em></p>

<table>
  <thead>
    <tr>
      <th align="center">Setting</th>
      <th align="center">Methods</th>
      <th align="center">ACC<sub>2</sub> ↑ (%)</th>
      <th align="center">ACC<sub>5</sub> ↑ (%)</th>
      <th align="center">ACC<sub>7</sub> ↑ (%)</th>
      <th align="center">F1 ↑ (%)</th>
      <th align="center">MAE ↓</th>
      <th align="center">Corr ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="15">Unaligned</td>
      <td align="center">EF-LSTM</td>
      <td align="center">73.6</td>
      <td align="center">33.1</td>
      <td align="center">31.0</td>
      <td align="center">74.5</td>
      <td align="center">1.420</td>
      <td align="center">0.508</td>
    </tr>
    <tr>
      <td align="center">LF-DNN</td>
      <td align="center">78.2</td>
      <td align="center">36.9</td>
      <td align="center">32.5</td>
      <td align="center">78.3</td>
      <td align="center">0.987</td>
      <td align="center">0.649</td>
    </tr>
    <tr>
      <td align="center">TFN</td>
      <td align="center">76.5</td>
      <td align="center">40.5</td>
      <td align="center">35.3</td>
      <td align="center">76.6</td>
      <td align="center">0.995</td>
      <td align="center">0.617</td>
    </tr>
    <tr>
      <td align="center">LMF</td>
      <td align="center">79.1</td>
      <td align="center">33.2</td>
      <td align="center">31.1</td>
      <td align="center">79.1</td>
      <td align="center">0.963</td>
      <td align="center">0.672</td>
    </tr>
    <tr>
      <td align="center">MFN</td>
      <td align="center">80.0</td>
      <td align="center">38.5</td>
      <td align="center">34.7</td>
      <td align="center">80.1</td>
      <td align="center">0.971</td>
      <td align="center">0.661</td>
    </tr>
    <tr>
      <td align="center">Graph-MFN</td>
      <td align="center">79.4</td>
      <td align="center">38.2</td>
      <td align="center">34.4</td>
      <td align="center">79.2</td>
      <td align="center">0.930</td>
      <td align="center">0.671</td>
    </tr>
    <tr>
      <td align="center">MCTN</td>
      <td align="center">77.1</td>
      <td align="center">33.4</td>
      <td align="center">31.9</td>
      <td align="center">77.3</td>
      <td align="center">1.033</td>
      <td align="center">0.650</td>
    </tr>
    <tr>
      <td align="center">MulT</td>
      <td align="center">80.3</td>
      <td align="center">37.8</td>
      <td align="center">33.2</td>
      <td align="center">80.3</td>
      <td align="center">0.933</td>
      <td align="center">0.685</td>
    </tr>
    <tr>
      <td align="center">MISA*</td>
      <td align="center">83.8</td>
      <td align="center">49.9</td>
      <td align="center">43.6</td>
      <td align="center">83.9</td>
      <td align="center">0.742</td>
      <td align="center">0.797</td>
    </tr>
    <tr>
      <td align="center">Self-MM*</td>
      <td align="center">83.4</td>
      <td align="center"><u>52.2</u></td>
      <td align="center">45.7</td>
      <td align="center">83.6</td>
      <td align="center">0.724</td>
      <td align="center">0.794</td>
    </tr>
    <tr>
      <td align="center">MMIM*</td>
      <td align="center">83.4</td>
      <td align="center">49.8</td>
      <td align="center"><u>45.9</u></td>
      <td align="center">83.4</td>
      <td align="center">0.777</td>
      <td align="center">0.771</td>
    </tr>
    <tr>
      <td align="center">TETFN*</td>
      <td align="center"><u>86.1</u></td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">86.0</td>
      <td align="center">0.717</td>
      <td align="center"><u>0.800</u></td>
    </tr>
    <tr>
      <td align="center">SUGRM*</td>
      <td align="center"><strong>86.3</strong></td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><strong>86.3</strong></td>
      <td align="center"><strong>0.703</strong></td>
      <td align="center">0.800</td>
    </tr>
    <tr>
      <td align="center">MUTA-Net*</td>
      <td align="center">84.9</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">84.9</td>
      <td align="center"><u>0.708</u></td>
      <td align="center">0.798</td>
    </tr>
    <tr>
      <td align="center">HSCL (Ours)*</td>
      <td align="center">85.2</td>
      <td align="center"><strong>52.4</strong></td>
      <td align="center"><strong>46.9</strong></td>
      <td align="center">85.2</td>
      <td align="center">0.716</td>
      <td align="center"><strong>0.806</strong></td>
    </tr>
    <tr>
      <td align="center" rowspan="15">Aligned</td>
      <td align="center">EF-LSTM</td>
      <td align="center">75.3</td>
      <td align="center">35.1</td>
      <td align="center">33.7</td>
      <td align="center">75.2</td>
      <td align="center">1.386</td>
      <td align="center">0.494</td>
    </tr>
    <tr>
      <td align="center">LF-DNN</td>
      <td align="center">78.4</td>
      <td align="center">33.5</td>
      <td align="center">31.5</td>
      <td align="center">78.3</td>
      <td align="center">0.972</td>
      <td align="center">0.650</td>
    </tr>
    <tr>
      <td align="center">TFN</td>
      <td align="center">78.8</td>
      <td align="center">38.3</td>
      <td align="center">31.9</td>
      <td align="center">78.9</td>
      <td align="center">0.953</td>
      <td align="center">0.680</td>
    </tr>
    <tr>
      <td align="center">LMF</td>
      <td align="center">78.7</td>
      <td align="center">41.7</td>
      <td align="center">36.9</td>
      <td align="center">78.7</td>
      <td align="center">0.931</td>
      <td align="center">0.652</td>
    </tr>
    <tr>
      <td align="center">MFN</td>
      <td align="center">78.4</td>
      <td align="center">40.8</td>
      <td align="center">35.6</td>
      <td align="center">78.4</td>
      <td align="center">0.964</td>
      <td align="center">0.657</td>
    </tr>
    <tr>
      <td align="center">Graph-MFN</td>
      <td align="center">78.1</td>
      <td align="center">35.1</td>
      <td align="center">31.5</td>
      <td align="center">78.1</td>
      <td align="center">0.970</td>
      <td align="center">0.661</td>
    </tr>
    <tr>
      <td align="center">MCTN</td>
      <td align="center">79.9</td>
      <td align="center">35.0</td>
      <td align="center">33.1</td>
      <td align="center">80.0</td>
      <td align="center">0.963</td>
      <td align="center">0.681</td>
    </tr>
    <tr>
      <td align="center">MulT</td>
      <td align="center">80.0</td>
      <td align="center">41.7</td>
      <td align="center">35.1</td>
      <td align="center">80.1</td>
      <td align="center">0.936</td>
      <td align="center">0.691</td>
    </tr>
    <tr>
      <td align="center">MISA*</td>
      <td align="center">84.2</td>
      <td align="center">47.8</td>
      <td align="center">41.8</td>
      <td align="center">84.2</td>
      <td align="center">0.754</td>
      <td align="center">0.785</td>
    </tr>
    <tr>
      <td align="center">Self-MM*</td>
      <td align="center">84.9</td>
      <td align="center"><u>51.5</u></td>
      <td align="center">45.3</td>
      <td align="center">84.9</td>
      <td align="center">0.738</td>
      <td align="center">0.738</td>
    </tr>
    <tr>
      <td align="center">MMIM*</td>
      <td align="center">84.6</td>
      <td align="center"><strong>51.9</strong></td>
      <td align="center"><u>45.8</u></td>
      <td align="center">84.5</td>
      <td align="center"><u>0.717</u></td>
      <td align="center">0.783</td>
    </tr>
    <tr>
      <td align="center">SUGRM*</td>
      <td align="center">84.5</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">84.5</td>
      <td align="center">0.723</td>
      <td align="center"><strong>0.798</strong></td>
    </tr>
    <tr>
      <td align="center">MUTA-Net*</td>
      <td align="center">85.0</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">85.0</td>
      <td align="center">0.730</td>
      <td align="center"><u>0.793</u></td>
    </tr>
    <tr>
      <td align="center">CAGC*</td>
      <td align="center"><strong>85.7</strong></td>
      <td align="center">-</td>
      <td align="center">44.8</td>
      <td align="center"><strong>85.6</strong></td>
      <td align="center">0.742</td>
      <td align="center">0.775</td>
    </tr>
    <tr>
      <td align="center">HSCL (Ours)*</td>
      <td align="center"><u>85.5</u></td>
      <td align="center"><u>51.5</u></td>
      <td align="center"><strong>46.5</strong></td>
      <td align="center"><strong>85.6</strong></td>
      <td align="center"><strong>0.712</strong></td>
      <td align="center">0.785</td>
    </tr>
  </tbody>
</table>

### Table II. Comparison on CMU-MOSEI

<p align="center"><strong>Comparison of results on the CMU-MOSEI dataset.</strong><br>
<em>* indicates that the method uses BERT-based features. Bold denotes the best result, and underlining denotes the second-best result. ↑ means higher is better, and ↓ means lower is better.</em></p>

<table>
  <thead>
    <tr>
      <th align="center">Setting</th>
      <th align="center">Methods</th>
      <th align="center">ACC<sub>2</sub> ↑ (%)</th>
      <th align="center">ACC<sub>5</sub> ↑ (%)</th>
      <th align="center">ACC<sub>7</sub> ↑ (%)</th>
      <th align="center">F1 ↑ (%)</th>
      <th align="center">MAE ↓</th>
      <th align="center">Corr ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="15">Unaligned</td>
      <td align="center">EF-LSTM</td>
      <td align="center">76.1</td>
      <td align="center">48.0</td>
      <td align="center">46.3</td>
      <td align="center">75.9</td>
      <td align="center">0.594</td>
      <td align="center">0.725</td>
    </tr>
    <tr>
      <td align="center">LF-DNN</td>
      <td align="center">83.7</td>
      <td align="center">53.8</td>
      <td align="center">52.3</td>
      <td align="center">83.2</td>
      <td align="center">0.561</td>
      <td align="center">0.728</td>
    </tr>
    <tr>
      <td align="center">TFN</td>
      <td align="center">84.2</td>
      <td align="center">51.7</td>
      <td align="center">50.2</td>
      <td align="center">84.0</td>
      <td align="center">0.573</td>
      <td align="center">0.728</td>
    </tr>
    <tr>
      <td align="center">LMF</td>
      <td align="center">83.8</td>
      <td align="center">53.6</td>
      <td align="center">51.9</td>
      <td align="center">83.9</td>
      <td align="center">0.565</td>
      <td align="center">0.735</td>
    </tr>
    <tr>
      <td align="center">MFN</td>
      <td align="center">83.2</td>
      <td align="center">52.5</td>
      <td align="center">51.3</td>
      <td align="center">83.3</td>
      <td align="center">0.567</td>
      <td align="center">0.726</td>
    </tr>
    <tr>
      <td align="center">Graph-MFN</td>
      <td align="center">84.2</td>
      <td align="center">52.9</td>
      <td align="center">51.8</td>
      <td align="center">84.2</td>
      <td align="center">0.568</td>
      <td align="center">0.725</td>
    </tr>
    <tr>
      <td align="center">MFM</td>
      <td align="center">82.3</td>
      <td align="center">53.1</td>
      <td align="center">52.0</td>
      <td align="center">82.5</td>
      <td align="center">0.572</td>
      <td align="center">0.729</td>
    </tr>
    <tr>
      <td align="center">MulT</td>
      <td align="center">84.0</td>
      <td align="center">55.0</td>
      <td align="center">53.2</td>
      <td align="center">84.0</td>
      <td align="center">0.556</td>
      <td align="center">0.740</td>
    </tr>
    <tr>
      <td align="center">MISA*</td>
      <td align="center">84.8</td>
      <td align="center">52.4</td>
      <td align="center">51.0</td>
      <td align="center">84.8</td>
      <td align="center">0.557</td>
      <td align="center">0.756</td>
    </tr>
    <tr>
      <td align="center">Self-MM*</td>
      <td align="center">85.3</td>
      <td align="center"><strong>55.8</strong></td>
      <td align="center">52.9</td>
      <td align="center">84.8</td>
      <td align="center"><strong>0.535</strong></td>
      <td align="center">0.761</td>
    </tr>
    <tr>
      <td align="center">MMIM*</td>
      <td align="center">81.5</td>
      <td align="center">54.1</td>
      <td align="center">52.6</td>
      <td align="center">81.3</td>
      <td align="center">0.578</td>
      <td align="center">0.706</td>
    </tr>
    <tr>
      <td align="center">TETFN*</td>
      <td align="center">85.1</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><u>85.2</u></td>
      <td align="center">0.551</td>
      <td align="center">0.748</td>
    </tr>
    <tr>
      <td align="center">SUGRM*</td>
      <td align="center">84.0</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">84.4</td>
      <td align="center">0.544</td>
      <td align="center">0.748</td>
    </tr>
    <tr>
      <td align="center">MUTA-Net*</td>
      <td align="center"><u>85.2</u></td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><u>85.2</u></td>
      <td align="center"><u>0.537</u></td>
      <td align="center"><u>0.764</u></td>
    </tr>
    <tr>
      <td align="center">HSCL (Ours)*</td>
      <td align="center"><strong>85.6</strong></td>
      <td align="center"><u>55.3</u></td>
      <td align="center"><strong>53.7</strong></td>
      <td align="center"><strong>85.4</strong></td>
      <td align="center">0.541</td>
      <td align="center"><strong>0.766</strong></td>
    </tr>
    <tr>
      <td align="center" rowspan="14">Aligned</td>
      <td align="center">EF-LSTM</td>
      <td align="center">78.2</td>
      <td align="center">51.1</td>
      <td align="center">47.4</td>
      <td align="center">77.9</td>
      <td align="center">0.620</td>
      <td align="center">0.679</td>
    </tr>
    <tr>
      <td align="center">LF-DNN</td>
      <td align="center">83.5</td>
      <td align="center">53.1</td>
      <td align="center">51.7</td>
      <td align="center">83.1</td>
      <td align="center">0.568</td>
      <td align="center">0.734</td>
    </tr>
    <tr>
      <td align="center">TFN</td>
      <td align="center">80.4</td>
      <td align="center">52.4</td>
      <td align="center">50.9</td>
      <td align="center">80.7</td>
      <td align="center">0.574</td>
      <td align="center">0.714</td>
    </tr>
    <tr>
      <td align="center">LMF</td>
      <td align="center">84.7</td>
      <td align="center">53.6</td>
      <td align="center">52.3</td>
      <td align="center">84.5</td>
      <td align="center">0.564</td>
      <td align="center">0.734</td>
    </tr>
    <tr>
      <td align="center">MFN</td>
      <td align="center">84.0</td>
      <td align="center">52.3</td>
      <td align="center">50.8</td>
      <td align="center">84.0</td>
      <td align="center">0.574</td>
      <td align="center">0.722</td>
    </tr>
    <tr>
      <td align="center">Graph-MFN</td>
      <td align="center">84.6</td>
      <td align="center">52.9</td>
      <td align="center">51.6</td>
      <td align="center">84.5</td>
      <td align="center">0.553</td>
      <td align="center">0.740</td>
    </tr>
    <tr>
      <td align="center">MFM</td>
      <td align="center">83.5</td>
      <td align="center">50.0</td>
      <td align="center">49.4</td>
      <td align="center">83.4</td>
      <td align="center">0.590</td>
      <td align="center">0.722</td>
    </tr>
    <tr>
      <td align="center">MulT</td>
      <td align="center">82.7</td>
      <td align="center">53.9</td>
      <td align="center">52.3</td>
      <td align="center">82.8</td>
      <td align="center">0.572</td>
      <td align="center">0.723</td>
    </tr>
    <tr>
      <td align="center">MISA*</td>
      <td align="center"><u>85.3</u></td>
      <td align="center">54.1</td>
      <td align="center">52.3</td>
      <td align="center">85.1</td>
      <td align="center">0.543</td>
      <td align="center"><u>0.764</u></td>
    </tr>
    <tr>
      <td align="center">Self-MM*</td>
      <td align="center">84.5</td>
      <td align="center"><strong>54.9</strong></td>
      <td align="center"><strong>53.2</strong></td>
      <td align="center">84.3</td>
      <td align="center"><u>0.540</u></td>
      <td align="center"><strong>0.769</strong></td>
    </tr>
    <tr>
      <td align="center">MMIM*</td>
      <td align="center">83.6</td>
      <td align="center">51.9</td>
      <td align="center">50.1</td>
      <td align="center">83.5</td>
      <td align="center">0.580</td>
      <td align="center">0.729</td>
    </tr>
    <tr>
      <td align="center">SUGRM*</td>
      <td align="center">85.1</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">85.0</td>
      <td align="center">0.541</td>
      <td align="center">0.758</td>
    </tr>
    <tr>
      <td align="center">MUTA-Net*</td>
      <td align="center">85.0</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">84.9</td>
      <td align="center">0.544</td>
      <td align="center">0.760</td>
    </tr>
    <tr>
      <td align="center">HSCL (Ours)*</td>
      <td align="center"><strong>85.4</strong></td>
      <td align="center"><u>54.5</u></td>
      <td align="center"><u>53.1</u></td>
      <td align="center"><strong>85.4</strong></td>
      <td align="center"><strong>0.538</strong></td>
      <td align="center"><strong>0.769</strong></td>
    </tr>
  </tbody>
</table>

### Table III. Comparison on CH-SIMS

<p align="center"><strong>Comparison of results on the CH-SIMS dataset.</strong><br>
<em>* indicates that the method uses BERT-based features. Bold denotes the best result, and underlining denotes the second-best result. ↑ means higher is better, and ↓ means lower is better.</em></p>

<table>
  <thead>
    <tr>
      <th align="center">Setting</th>
      <th align="center">Methods</th>
      <th align="center">ACC<sub>2</sub> ↑ (%)</th>
      <th align="center">ACC<sub>3</sub> ↑ (%)</th>
      <th align="center">ACC<sub>5</sub> ↑ (%)</th>
      <th align="center">F1 ↑ (%)</th>
      <th align="center">MAE ↓</th>
      <th align="center">Corr ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="12">Unaligned</td>
      <td align="center">EF-LSTM</td>
      <td align="center">69.4</td>
      <td align="center">54.3</td>
      <td align="center">21.2</td>
      <td align="center">56.8</td>
      <td align="center">0.590</td>
      <td align="center">0.055</td>
    </tr>
    <tr>
      <td align="center">LF-DNN</td>
      <td align="center">77.0</td>
      <td align="center">64.3</td>
      <td align="center">39.7</td>
      <td align="center">77.3</td>
      <td align="center">0.446</td>
      <td align="center">0.555</td>
    </tr>
    <tr>
      <td align="center">TFN</td>
      <td align="center">78.4</td>
      <td align="center">65.1</td>
      <td align="center">39.3</td>
      <td align="center">78.6</td>
      <td align="center">0.432</td>
      <td align="center">0.591</td>
    </tr>
    <tr>
      <td align="center">LMF</td>
      <td align="center">77.8</td>
      <td align="center">64.7</td>
      <td align="center">40.5</td>
      <td align="center">77.9</td>
      <td align="center">0.441</td>
      <td align="center">0.576</td>
    </tr>
    <tr>
      <td align="center">MFN</td>
      <td align="center">77.9</td>
      <td align="center">65.7</td>
      <td align="center">39.5</td>
      <td align="center">77.9</td>
      <td align="center">0.435</td>
      <td align="center">0.582</td>
    </tr>
    <tr>
      <td align="center">Graph-MFN</td>
      <td align="center">78.8</td>
      <td align="center">65.7</td>
      <td align="center">39.8</td>
      <td align="center">78.2</td>
      <td align="center">0.445</td>
      <td align="center">0.578</td>
    </tr>
    <tr>
      <td align="center">MFM</td>
      <td align="center">75.1</td>
      <td align="center">54.3</td>
      <td align="center">21.2</td>
      <td align="center">75.6</td>
      <td align="center">0.477</td>
      <td align="center">0.525</td>
    </tr>
    <tr>
      <td align="center">MulT</td>
      <td align="center">78.6</td>
      <td align="center">64.8</td>
      <td align="center">37.9</td>
      <td align="center">79.7</td>
      <td align="center">0.453</td>
      <td align="center">0.564</td>
    </tr>
    <tr>
      <td align="center">MISA*</td>
      <td align="center">69.5</td>
      <td align="center">54.1</td>
      <td align="center">21.8</td>
      <td align="center">57.0</td>
      <td align="center">0.588</td>
      <td align="center">0.542</td>
    </tr>
    <tr>
      <td align="center">Self-MM*</td>
      <td align="center">78.7</td>
      <td align="center"><strong>65.5</strong></td>
      <td align="center"><u>41.5</u></td>
      <td align="center">78.7</td>
      <td align="center"><u>0.422</u></td>
      <td align="center">0.584</td>
    </tr>
    <tr>
      <td align="center">MUTA-Net*</td>
      <td align="center"><u>79.2</u></td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><u>79.3</u></td>
      <td align="center">0.428</td>
      <td align="center"><u>0.594</u></td>
    </tr>
    <tr>
      <td align="center">HSCL (Ours)*</td>
      <td align="center"><strong>80.5</strong></td>
      <td align="center"><u>65.2</u></td>
      <td align="center"><strong>43.5</strong></td>
      <td align="center"><strong>80.5</strong></td>
      <td align="center"><strong>0.415</strong></td>
      <td align="center"><strong>0.609</strong></td>
    </tr>
  </tbody>
</table>


## Ablation Studies

### Table IV. Ablation Study of Each Level of Consistency

<p align="center"><strong>Ablation study of each level of consistency in HSCL under unaligned data settings.</strong></p>

<table>
  <thead>
    <tr>
      <th align="center" rowspan="2">Dataset</th>
      <th align="center" colspan="4">Hierarchical Consistency</th>
      <th align="center" rowspan="2">ACC<sub>2</sub> (%)</th>
      <th align="center" rowspan="2">F1 (%)</th>
      <th align="center" rowspan="2">MAE</th>
    </tr>
    <tr>
      <th align="center">Sem-</th>
      <th align="center">Rep-</th>
      <th align="center">Geo-</th>
      <th align="center">Rec-</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" rowspan="8">MOSI</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">85.2</td>
      <td align="center">85.2</td>
      <td align="center">0.716</td>
    </tr>
    <tr>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">84.2</td>
      <td align="center">84.2</td>
      <td align="center">0.743</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">84.0</td>
      <td align="center">83.9</td>
      <td align="center">0.741</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center">83.7</td>
      <td align="center">83.7</td>
      <td align="center">0.726</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">82.9</td>
      <td align="center">82.9</td>
      <td align="center">0.741</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">83.1</td>
      <td align="center">83.1</td>
      <td align="center">0.738</td>
    </tr>
    <tr>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">82.6</td>
      <td align="center">82.4</td>
      <td align="center">0.750</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">82.2</td>
      <td align="center">82.1</td>
      <td align="center">0.775</td>
    </tr>
    <tr>
      <td align="center" rowspan="8">MOSEI</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">85.6</td>
      <td align="center">85.4</td>
      <td align="center">0.541</td>
    </tr>
    <tr>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">85.4</td>
      <td align="center">85.4</td>
      <td align="center">0.549</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">85.3</td>
      <td align="center">85.1</td>
      <td align="center">0.544</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center">84.8</td>
      <td align="center">84.9</td>
      <td align="center">0.543</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">85.3</td>
      <td align="center">85.3</td>
      <td align="center">0.559</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">84.9</td>
      <td align="center">85.0</td>
      <td align="center">0.537</td>
    </tr>
    <tr>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">84.8</td>
      <td align="center">84.8</td>
      <td align="center">0.550</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">84.5</td>
      <td align="center">84.5</td>
      <td align="center">0.552</td>
    </tr>
    <tr>
      <td align="center" rowspan="8">SIMS</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">80.5</td>
      <td align="center">80.5</td>
      <td align="center">0.415</td>
    </tr>
    <tr>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">79.8</td>
      <td align="center">79.7</td>
      <td align="center">0.414</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">79.2</td>
      <td align="center">79.3</td>
      <td align="center">0.417</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center">79.2</td>
      <td align="center">78.9</td>
      <td align="center">0.439</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center">76.8</td>
      <td align="center">77.4</td>
      <td align="center">0.429</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">77.9</td>
      <td align="center">78.2</td>
      <td align="center">0.431</td>
    </tr>
    <tr>
      <td align="center">✓</td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">78.1</td>
      <td align="center">78.2</td>
      <td align="center">0.420</td>
    </tr>
    <tr>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center"></td>
      <td align="center">77.7</td>
      <td align="center">77.6</td>
      <td align="center">0.435</td>
    </tr>
  </tbody>
</table>

### Table V. Comparison of Feature Decoupling and HSCL

<p align="center"><strong>Comparison of FD and HSCL. FD represents the result of the basic feature decoupling neural network.</strong></p>

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Dataset</th>
      <th align="center">MER Network</th>
      <th align="center">HSCL</th>
    </tr>
    <tr>
      <th align="center">ACC<sub>2</sub> (%) / F1 (%) / MAE</th>
      <th align="center">ACC<sub>2</sub> (%) / F1 (%) / MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="3"><strong>Data Setting: unaligned</strong></td>
    </tr>
    <tr>
      <td align="center">MOSI</td>
      <td align="center">82.2 / 82.1 / 0.775</td>
      <td align="center"><strong>85.2 / 85.2 / 0.716</strong></td>
    </tr>
    <tr>
      <td align="center">MOSEI</td>
      <td align="center">84.5 / 84.5 / 0.552</td>
      <td align="center"><strong>85.6 / 85.4 / 0.541</strong></td>
    </tr>
    <tr>
      <td align="center">SIMS</td>
      <td align="center">77.7 / 77.6 / 0.435</td>
      <td align="center"><strong>80.5 / 80.5 / 0.415</strong></td>
    </tr>
    <tr>
      <td colspan="3"><strong>Data Setting: aligned</strong></td>
    </tr>
    <tr>
      <td align="center">MOSI</td>
      <td align="center">84.4 / 84.3 / 0.749</td>
      <td align="center"><strong>85.5 / 85.6 / 0.712</strong></td>
    </tr>
    <tr>
      <td align="center">MOSEI</td>
      <td align="center">85.1 / 85.1 / 0.540</td>
      <td align="center"><strong>85.4 / 85.3 / 0.538</strong></td>
    </tr>
  </tbody>
</table>

### Table VI. Unimodal Accuracy Comparison on CMU-MOSEI

<p align="center"><strong>Unimodal accuracy comparison on MOSEI.</strong></p>

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Modality</th>
      <th align="center">MER Network</th>
      <th align="center">HSCL</th>
    </tr>
    <tr>
      <th align="center">ACC<sub>2</sub> (%) / F1 (%) / MAE</th>
      <th align="center">ACC<sub>2</sub> (%) / F1 (%) / MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">L only</td>
      <td align="center">80.2 / 80.6 / 0.570</td>
      <td align="center"><strong>84.6 / 84.6 / 0.559</strong></td>
    </tr>
    <tr>
      <td align="center">V only</td>
      <td align="center">40.8 / 23.8 / 0.889</td>
      <td align="center"><strong>62.9 / 48.5 / 0.833</strong></td>
    </tr>
    <tr>
      <td align="center">A only</td>
      <td align="center">37.2 / 20.1 / 0.856</td>
      <td align="center"><strong>62.8 / 48.7 / 0.822</strong></td>
    </tr>
    <tr>
      <td align="center"><strong>Mean</strong></td>
      <td align="center">52.7 / 41.5 / 0.777</td>
      <td align="center"><strong>70.1 / 60.6 / 0.738</strong></td>
    </tr>
    <tr>
      <td align="center"><strong>STD</strong></td>
      <td align="center">19.5 / 27.7 / 0.143</td>
      <td align="center"><strong>10.3 / 17.0 / 0.127</strong></td>
    </tr>
  </tbody>
</table>

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

We recommend that users download these publicly available datasets from official channels, but for quick access, proceeded data files can be downloaded from the following links:

- MOSI and MOSEI: [Google Drive](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi)
- SIMS: [Google Drive](https://drive.google.com/file/d/1W210HygAtU-Cp-TGwALGDHjN9MfsacMB/view?usp=drive_link)


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
