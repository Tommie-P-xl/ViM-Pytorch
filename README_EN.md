[简体中文](README.md) | [English](README_EN.md)

# ViM UAV RF Open-Set Recognition

> **Virtual-logit Matching (ViM)** algorithm-based open-set recognition system for UAV RF signals
> Input: 512×512 STFT spectrograms (`.npy` format)
> Backbone: ResNet-50 (configurable)
> Framework: PyTorch 2.x

---

## Table of Contents

- [Project Overview](#project-overview)
- [Algorithm Principle](#algorithm-principle)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration File Guide](#configuration-file-guide)
- [Training](#training)
- [Testing](#testing)
- [Evaluation Metrics](#evaluation-metrics)
- [Module Extension Guide](#module-extension-guide)

---

## Project Overview

This project applies the ViM (Virtual-logit Matching) algorithm from CVPR 2022 to **open-set recognition** tasks for UAV RF signals.

The key difference between **open-set recognition** and traditional closed-set classification is that during testing, **unknown classes** (Unknown Classes) that were never seen during training may appear. The model needs to simultaneously accomplish:

1. **OOD Detection**: Determine whether an input sample is from a known class (In-Distribution) or unknown class (Out-of-Distribution)
2. **Known Class Classification**: For samples identified as known classes, provide specific class labels

The advantage of the ViM algorithm is that it requires **no network structure modification, no additional OOD data, and has extremely fast inference speed**, efficiently detecting unknown samples by analyzing the geometric structure of the pre-trained model's feature space.

---

## Algorithm Principle

The core idea of ViM is to construct an additional logic value for a "virtual OOD class" and fuse complementary information from both feature space and logit space:

```
ViM Score = Energy Score − Residual Score
          = log Σ exp(lᵢ)  −  α · ‖x^{P⊥}‖
```

- **Energy Score**: Measures the classifier's confidence in known classes (higher → more likely to be known class)
- **Residual Score**: Measures how much features deviate from the principal subspace of known classes (higher → more likely to be unknown class)
- **α**: Scaling constant used to unify the numerical dimensions of both parts

The higher the ViM score, the more likely the sample is in-distribution (ID); if below the threshold, it's classified as OOD (unknown class).

---

## Project Structure

```
vim_uav_osr/
├── configs/
│   └── config.yaml              # Main configuration file (all hyperparameters centralized here)
├── datasets/
│   ├── __init__.py
│   └── uav_rf_dataset.py        # Dataset loading, class division, sample splitting
├── models/
│   ├── __init__.py
│   ├── model_factory.py         # Model factory (backbone network construction)
│   └── optim_factory.py         # Optimizer and learning rate scheduler factory
├── trainers/
│   ├── __init__.py
│   └── trainer.py               # Training loop (with TensorBoard, AMP, progress bar)
├── evaluators/
│   ├── __init__.py
│   ├── vim_scorer.py            # ViM algorithm core (feature extraction cache + score calculation)
│   └── metrics.py               # Open-set recognition evaluation metrics
├── utils/
│   ├── __init__.py
│   └── utils.py                 # Utility functions (config reading, random seed, device selection)
├── tools/
│   ├── verify_dataset.py        # Dataset structure verification tool
│   └── generate_dummy_dataset.py # Generate dummy dataset (for code debugging)
├── data/                        # Dataset storage directory (need to prepare yourself)
│   └── uav_rf/
│       ├── drone_DJI/           # Known classes (subfolder names are class names)
│       │   ├── sample_0001.npy
│       │   └── ...
│       ├── drone_Parrot/
│       ├── unknown_drone_A/     # Unknown classes (specified in config file)
│       └── ...
├── outputs/                     # Training outputs (checkpoints, evaluation results)
├── runs/                        # TensorBoard logs
├── train.py                     # Training entry point
├── test.py                      # Testing entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Dataset format requirements:

```
<dataset_root>/
├── <class_name_1>/          # Subfolder name is the class name
│   ├── sample_001.npy       # 512×512 STFT spectrogram
│   ├── sample_002.npy
│   └── ...
├── <class_name_2>/
│   └── ...
└── ...
```

For quick code verification, you can generate a dummy dataset:

```bash
python tools/generate_dummy_dataset.py
```

### 3. Verify Dataset Structure

```bash
python tools/verify_dataset.py --config configs/config.yaml
# Add --check_shape parameter to verify each npy file's shape
python tools/verify_dataset.py --config configs/config.yaml --check_shape
```

### 4. Modify Configuration File

Edit `configs/config.yaml`, focusing on configuring:

```yaml
dataset:
  root: "./data/uav_rf"          # Dataset root directory
  unknown_classes:               # Specify which classes as unknown classes
    - "unknown_drone_A"
    - "unknown_drone_B"
```

### 5. Training

```bash
python train.py --config configs/config.yaml
```

### 6. Testing

```bash
# Automatically use the optimal checkpoint specified in the config file
python test.py --config configs/config.yaml

# Or manually specify checkpoint
python test.py --config configs/config.yaml --checkpoint outputs/best_model.pth

# Force feature re-extraction (ignore cache)
python test.py --config configs/config.yaml --force_recompute
```

---

## Configuration File Guide

All parameters are centralized in `configs/config.yaml`, allowing experimental configuration without modifying any Python code.

| Config Section | Key Parameters | Description |
|--------|---------|------|
| `dataset` | `root` | Dataset root directory |
| `dataset` | `unknown_classes` | List of unknown class names |
| `dataset` | `train_ratio / val_ratio / test_ratio` | Known class sample split ratios |
| `model` | `backbone` | Backbone network (resnet50 / resnet18 / resnet101 / efficientnet_b0) |
| `model` | `pretrained` | Whether to use ImageNet pretrained weights (imagenet / none) |
| `model` | `single_channel_input` | Whether input is single-channel (STFT is single-channel) |
| `train` | `epochs / batch_size / amp` | Training hyperparameters |
| `train.optimizer` | `type / lr / weight_decay` | Optimizer type and parameters |
| `train.scheduler` | `type` | Scheduler type (cosine / step / multistep / plateau / warmup_cosine) |
| `vim` | `dim` | Principal subspace dimension (-1 for automatic) |
| `vim` | `tpr` | OOD detection threshold corresponding to true positive rate (0.95 = retain 95% ID samples) |
| `paths` | `output_dir / result_txt` | Output directory and result file path |

---

## Training

```bash
python train.py --config configs/config.yaml
```

During training:
- **Terminal**: Real-time display of loss, accuracy, and learning rate for each batch using tqdm progress bar
- **TensorBoard**: Records train/val loss, accuracy, and learning rate curves

View TensorBoard:
```bash
tensorboard --logdir=runs
```

---

## Testing

```bash
python test.py --config configs/config.yaml
```

Testing process:
1. Load optimal model weights (automatically obtained from config file path, no manual adjustment needed)
2. Extract features for training, validation, and test sets separately (**first run** performs forward propagation, results cached to `outputs/feature_cache/`; **subsequent runs** load cache directly)
3. Fit ViM parameters (principal subspace, α) on training set features
4. Determine OOD detection threshold on validation set
5. Perform comprehensive evaluation on test set, results printed to terminal and saved to txt file

---

## Evaluation Metrics

### OOD Detection Metrics

| Metric | Meaning |
|------|------|
| **AUROC** | Area under ROC curve, higher is better (1.0 = perfect) |
| **AUPR-In** | Area under PR curve with ID as positive class |
| **AUPR-Out** | Area under PR curve with OOD as positive class |
| **FPR@TPR95** | Ratio of OOD samples misclassified as ID when TPR=95%, lower is better |
| **Detection Accuracy** | ID/OOD binary classification accuracy at given threshold |
| **Detection F1** | Binary classification F1 score |

### Known Class Classification Metrics

| Metric | Meaning |
|------|------|
| **Closed-Set Acc** | Classification accuracy only for ID test samples |
| **Per-class Acc** | Individual accuracy for each known class |

### Open-Set Recognition Comprehensive Metrics

| Metric | Meaning |
|------|------|
| **Open-Set Acc** | Overall accuracy under two-stage decision (OOD correctly rejected + ID correctly classified) |
| **Open Macro-F1** | Macro-averaged F1 including "rejection class" |
| **Openness** | Degree of openness, defined as $1 - \sqrt{K_{known} / K_{total}}$ |

---

## Module Extension Guide

### Adding New Backbone Networks

In the `build_model()` function in `models/model_factory.py`, add a new `elif` branch following the existing pattern, and set `model.backbone` to the new name in `configs/config.yaml`.

### Adding New Optimizers

Add a new `elif` branch in the `build_optimizer()` function in `models/optim_factory.py`, and set `train.optimizer.type` in the configuration file.

### Adding New Learning Rate Schedulers

Add a new `elif` branch in the `build_scheduler()` function in `models/optim_factory.py`, and set `train.scheduler.type` in the configuration file.
