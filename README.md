# Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition (HSCL)

This repository contains the code for our project **Hierarchical Structure Consistency Learning for Multimodal Emotion Recognition (HSCL)**.

We build a neural network framework based on **decoupling** and **attention mechanisms**, and introduce a **hierarchical structure consistency learning** algorithm. The proposed approach alleviates **cross-modal asynchrony** and enhances **intra-modal aggregation**. Experiments on multiple public benchmark datasets demonstrate that our method achieves competitive or superior performance compared to existing baselines.

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


