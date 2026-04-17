[简体中文](README.md) | [English](README_EN.md)

# ViM 无人机射频开集识别

> 基于 **Virtual-logit Matching (ViM)** 算法的无人机射频信号开集识别系统
> 输入：512×512 STFT 能量谱（`.npy` 格式）
> 主干网络：ResNet-50（可配置）
> 框架：PyTorch 2.x

---

## 目录

- [项目简介](#项目简介)
- [算法原理](#算法原理)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置文件说明](#配置文件说明)
- [训练](#训练)
- [测试](#测试)
- [评估指标说明](#评估指标说明)
- [模块扩展指南](#模块扩展指南)

---

## 项目简介

本项目将 CVPR 2022 的 ViM（Virtual-logit Matching）算法应用于无人机射频信号的**开集识别**任务。

**开集识别**与传统闭集分类的核心区别在于：测试阶段会出现训练时从未见过的**未知类别**（Unknown Classes），模型需要同时完成：

1. **OOD 检测**：判断输入样本是已知类（In-Distribution）还是未知类（Out-of-Distribution）
2. **已知类分类**：对被判为已知类的样本，给出具体的类别标签

ViM 算法的优势在于**无需修改网络结构、无需额外 OOD 数据、推理速度极快**，仅通过分析预训练模型的特征空间几何结构即可高效检测未知样本。

---

## 算法原理

ViM 的核心思想是构建一个"虚拟 OOD 类别"的额外逻辑值，融合特征空间和逻辑值空间的互补信息：

```
ViM Score = Energy Score − Residual Score
          = log Σ exp(lᵢ)  −  α · ‖x^{P⊥}‖
```

- **Energy Score**：衡量分类器对已知类别的置信度（越高 → 越像已知类）
- **Residual Score**：衡量特征偏离已知类主子空间的程度（越高 → 越像未知类）
- **α**：缩放常数，用于统一两部分的数值量纲

ViM 得分越高，样本越像分布内（ID）样本；低于阈值则判定为 OOD（未知类）。

---

## 项目结构

```
vim_uav_osr/
├── configs/
│   └── config.yaml              # 主配置文件（所有超参数集中于此）
├── datasets/
│   ├── __init__.py
│   └── uav_rf_dataset.py        # 数据集加载、类别划分、样本划分
├── models/
│   ├── __init__.py
│   ├── model_factory.py         # 模型工厂（主干网络构建）
│   └── optim_factory.py         # 优化器和学习率调度器工厂
├── trainers/
│   ├── __init__.py
│   └── trainer.py               # 训练循环（含 TensorBoard、AMP、进度条）
├── evaluators/
│   ├── __init__.py
│   ├── vim_scorer.py            # ViM 算法核心（特征提取缓存 + 得分计算）
│   └── metrics.py               # 开集识别评估指标
├── utils/
│   ├── __init__.py
│   └── utils.py                 # 工具函数（配置读取、随机种子、设备选择）
├── tools/
│   ├── verify_dataset.py        # 数据集结构验证工具
│   └── generate_dummy_dataset.py # 生成虚拟数据集（用于代码调试）
├── data/                        # 数据集存放目录（需自行准备）
│   └── uav_rf/
│       ├── drone_DJI/           # 已知类（子文件夹名即类别名）
│       │   ├── sample_0001.npy
│       │   └── ...
│       ├── drone_Parrot/
│       ├── unknown_drone_A/     # 未知类（配置文件中指定）
│       └── ...
├── outputs/                     # 训练产物（检查点、评估结果）
├── runs/                        # TensorBoard 日志
├── train.py                     # 训练入口
├── test.py                      # 测试入口
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```


### 2. 准备数据集

数据集格式要求：

```
<dataset_root>/
├── <class_name_1>/          # 子文件夹名即为类别名
│   ├── sample_001.npy       # 512×512 的 STFT 能量谱
│   ├── sample_002.npy
│   └── ...
├── <class_name_2>/
│   └── ...
└── ...
```

如需快速验证代码，可生成虚拟数据集：

```bash
python tools/generate_dummy_dataset.py
```

### 3. 验证数据集结构

```bash
python tools/verify_dataset.py --config configs/config.yaml
# 加 --check_shape 参数可验证每个 npy 文件的形状
python tools/verify_dataset.py --config configs/config.yaml --check_shape
```

### 4. 修改配置文件

编辑 `configs/config.yaml`，重点配置：

```yaml
dataset:
  root: "./data/uav_rf"          # 数据集根目录
  unknown_classes:               # 指定哪些类别作为未知类
    - "unknown_drone_A"
    - "unknown_drone_B"
```

### 5. 训练

```bash
python train.py --config configs/config.yaml
```

### 6. 测试

```bash
# 自动使用配置文件中指定的最优检查点
python test.py --config configs/config.yaml

# 或手动指定检查点
python test.py --config configs/config.yaml --checkpoint outputs/best_model.pth

# 强制重新提取特征（忽略缓存）
python test.py --config configs/config.yaml --force_recompute
```

---

## 配置文件说明

所有参数均集中在 `configs/config.yaml` 中，无需修改任何 Python 代码即可完成实验配置。

| 配置段 | 关键参数 | 说明 |
|--------|---------|------|
| `dataset` | `root` | 数据集根目录 |
| `dataset` | `unknown_classes` | 未知类类别名列表 |
| `dataset` | `train_ratio / val_ratio / test_ratio` | 已知类样本划分比例 |
| `model` | `backbone` | 主干网络（resnet50 / resnet18 / resnet101 / efficientnet_b0） |
| `model` | `pretrained` | 是否使用 ImageNet 预训练权重（imagenet / none） |
| `model` | `single_channel_input` | 是否以单通道输入（STFT 为单通道） |
| `train` | `epochs / batch_size / amp` | 训练超参数 |
| `train.optimizer` | `type / lr / weight_decay` | 优化器类型及参数 |
| `train.scheduler` | `type` | 调度器类型（cosine / step / multistep / plateau / warmup_cosine） |
| `vim` | `dim` | 主子空间维度（-1 表示自动） |
| `vim` | `tpr` | OOD 检测阈值对应的真阳性率（0.95 = 保留 95% ID 样本） |
| `paths` | `output_dir / result_txt` | 输出目录和结果文件路径 |

---

## 训练

```bash
python train.py --config configs/config.yaml
```

训练过程中：
- **终端**：使用 tqdm 进度条实时显示每个 batch 的 loss、acc、lr
- **TensorBoard**：记录 train/val 的 loss、acc 和学习率曲线

查看 TensorBoard：
```bash
tensorboard --logdir=runs
```

---

## 测试

```bash
python test.py --config configs/config.yaml
```

测试流程：
1. 加载最优模型权重（自动从配置文件获取路径，无需手动调整）
2. 对训练集、验证集、测试集分别提取特征（**首次运行**会进行前向传播，结果缓存到 `outputs/feature_cache/`；**再次运行**直接加载缓存）
3. 在训练集特征上拟合 ViM 参数（主子空间、α）
4. 在验证集上确定 OOD 检测阈值
5. 在测试集上进行全面评估，结果打印到终端并保存到 txt 文件

---

## 评估指标说明

### OOD 检测指标

| 指标 | 含义 |
|------|------|
| **AUROC** | ROC 曲线下面积，越高越好（1.0 = 完美） |
| **AUPR-In** | 以 ID 为正例的 PR 曲线下面积 |
| **AUPR-Out** | 以 OOD 为正例的 PR 曲线下面积 |
| **FPR@TPR95** | TPR=95% 时，OOD 样本被误判为 ID 的比率，越低越好 |
| **检测准确率** | 给定阈值下，ID/OOD 二分类准确率 |
| **检测 F1** | 二分类 F1 分数 |

### 已知类分类指标

| 指标 | 含义 |
|------|------|
| **Closed-Set Acc** | 仅对 ID 测试样本的分类准确率 |
| **Per-class Acc** | 每个已知类别各自的准确率 |

### 开集识别综合指标

| 指标 | 含义 |
|------|------|
| **Open-Set Acc** | 两阶段判定下的综合准确率（OOD 被正确拒绝 + ID 被正确分类） |
| **Open Macro-F1** | 包含"拒绝类"的宏平均 F1 |
| **Openness** | 开放度，定义为 $1 - \sqrt{K_{known} / K_{total}}$ |

---

## 模块扩展指南

### 添加新的主干网络

在 `models/model_factory.py` 的 `build_model()` 函数中，按照已有模式添加新的 `elif` 分支即可，并在 `configs/config.yaml` 中将 `model.backbone` 设置为新的名称。

### 添加新的优化器

在 `models/optim_factory.py` 的 `build_optimizer()` 函数中添加新的 `elif` 分支，并在配置文件中设置 `train.optimizer.type`。

### 添加新的学习率调度器

在 `models/optim_factory.py` 的 `build_scheduler()` 函数中添加新的 `elif` 分支，并在配置文件中设置 `train.scheduler.type`。

