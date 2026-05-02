# HybridFR_NRModel

This repository contains my implementation of a **Hybrid Full-Reference (FR) and No-Reference (NR) Image Quality Assessment model** built using features extracted from the **DisQUE** framework.

The focus of this project is:

* Extracting quality-aware features
* Training models for IQA
* Evaluating performance on datasets

---

## Project Structure

```bash
HybridFR_NRModel/
└── disque/
    ├── disque/                  # Core DisQUE feature extractor
    ├── train_disque.py         # Training script
    ├── train_distill.py        # Distillation training
    ├── evaluate_final.py       # Final evaluation
    ├── extract_features.py     # Feature extraction (single pair)
    ├── extract_features_from_dataset.py
    ├── download_data.py
    ├── requirements.txt
```

---

## What I Implemented

* Feature extraction using DisQUE representations
* Training pipeline for IQA models
* Distillation-based training approach
* Evaluation pipeline for model performance
* Support for dataset-level feature extraction

---

## Setup

### 1. Clone repository

```bash
git clone https://github.com/rahul09140102/HybridFR_NRModel.git
cd HybridFR_NRModel
```

---

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r disque/requirements.txt
```

---

### 4. Download required data

```bash
python disque/download_data.py
```

---

## Feature Extraction

### Extract features from a single video pair

```bash
python disque/extract_features.py \
  --ref_video <reference_video> \
  --dis_video <distorted_video> \
  --ckpt_path <model_checkpoint>
```

---

### Extract features from a dataset

```bash
python disque/extract_features_from_dataset.py \
  --dataset <dataset_file> \
  --ckpt_path <model_checkpoint>
```

---

## Training

### Train model

```bash
python disque/train_disque.py
```

### Train distillation model

```bash
python disque/train_distill.py
```

---

## Evaluation

```bash
python disque/evaluate_final.py
```

---

## Notes

* Large files (checkpoints, datasets) are not included in the repository
* Use `.gitignore` to exclude:

```bash
disque/DisQUE_Checkpoints/
disque/DisQUE_Images/
*.pth
*.pt
.virtual_documents/
```

---

## Summary

This project focuses on building a practical IQA pipeline using DisQUE-based features, including:

* Feature extraction
* Model training
* Performance evaluation

---

## Contact

Rahul
GitHub: https://github.com/rahul09140102
