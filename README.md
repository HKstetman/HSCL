# Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition (HSCL)

This repository contains the code for our project **Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition (HSCL)**. This work is puiblished on IEEE Transactions on Multimedia.

We build a neural network framework based on **decoupling** and **attention mechanisms**, and introduce a **hierarchical structure consistency learning** algorithm. The proposed approach alleviates **cross-modal asynchrony** and enhances **intra-modal aggregation**. 

<img width="2097" height="1008" alt="fig2" src="https://github.com/user-attachments/assets/3af53001-2fc5-435f-9c82-d932a88c4aa1" />

Our **hierarchical structure consistency learning** algorithm including three levels of consistency: feature level, modality level, and sample level.

<img width="1974" height="689" alt="fig3" src="https://github.com/user-attachments/assets/0fd91190-6925-4770-a4ea-5bf73a791927" />

Experiments on multiple public benchmark datasets demonstrate that our method achieves competitive or superior performance compared to existing baselines.

<img width="1231" height="874" alt="image" src="https://github.com/user-attachments/assets/9c3b0479-bd24-43a3-8964-21cec3862a6e" />

<img width="1254" height="1290" alt="image" src="https://github.com/user-attachments/assets/b4f534a8-e3cd-466a-b194-e99dc2dfb5c2" />

We have also done various ablation study to prove the effectiveness and robustness of the core pasts.

<img width="624" height="735" alt="image" src="https://github.com/user-attachments/assets/5a456448-49a4-40ee-afde-826fe0ea2475" />

<img width="1295" height="1376" alt="image" src="https://github.com/user-attachments/assets/8dd0c945-6dc8-4e0f-9626-7cb1fbc6e49d" />

<img width="631" height="411" alt="image" src="https://github.com/user-attachments/assets/e6e426fe-db5d-4c90-8e78-cf70c2369cb8" />

<img width="573" height="321" alt="image" src="https://github.com/user-attachments/assets/697a34f6-7b95-4668-9959-fbdb663d18a6" />

---

## Quick Start

### 1. Environment Setup

1. Install Python 3.9 (or version compatible with your PyTorch).
2. Install the required dependencies

### 2. Dataset Preparation

1. Download and prepare the multimodal emotion recognition datasets you would like to use (e.g., CMU-MOSI, CMU-MOSEI, CH-SIMS etc., if applicable to your experiments).
2. Modify the dataset **paths** in the configuration files so that they correctly point to your local dataset directories.

### 3. Training

To train the HSCL model, run:

```bash
python train.py
```

This script will:

* Load the specified dataset and configuration.
* Train the HSCL model.
* Save the trained model checkpoints (e.g., under `./pt` or another directory specified in the config).

### 4. Testing

After training, you can evaluate the trained model by running:

```bash
python test.py
```

This script will:

* Load the trained model.
* Perform evaluation on the test set.
* Report the performance metrics on the chosen dataset(s).

---


## Copyright & Statement

All technical innovations and methods proposed in the associated paper are the intellectual property of the authors.

We firmly oppose and reject any form of **plagiarism** or unauthorized use of the ideas and implementations presented in our work.


