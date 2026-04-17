"""
数据集模块：负责无人机射频 STFT 数据的加载与类别/样本划分
支持已知类（Known）和未知类（Unknown）的严格隔离
"""

import os
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ==============================================================================
# 核心 Dataset 类
# ==============================================================================

class UAVRFDataset(Dataset):
    """
    无人机射频 STFT 开集识别数据集
    
    单个样本：512×512 的 STFT 能量谱，以 .npy 格式存储
    支持三通道复制模式（兼容 ImageNet 预训练权重）和单通道模式
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        single_channel: bool = True,
    ):
        """
        参数:
            samples: [(文件路径, 类别标签), ...] 的列表
            transform: 数据变换（torchvision.transforms）
            single_channel: 是否保持单通道输入（True=单通道，False=复制为三通道）
        """
        self.samples = samples
        self.transform = transform
        self.single_channel = single_channel

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        # 加载 npy 文件（512×512 的二维矩阵）
        data = np.load(path).astype(np.float32)

        # 转为 PyTorch Tensor，增加通道维度 → (1, H, W)
        data = torch.from_numpy(data).unsqueeze(0)

        if not self.single_channel:
            # 复制为三通道以兼容 ImageNet 预训练权重
            data = data.repeat(3, 1, 1)

        if self.transform is not None:
            data = self.transform(data)

        return data, label


# ==============================================================================
# 数据集构建器
# ==============================================================================

class DatasetBuilder:
    """
    数据集构建器：负责类别划分、样本划分，并生成各分割的 DataLoader
    
    核心规则：
    - 未知类的样本在训练和验证阶段绝对不可见
    - 未知类的全部样本直接进入最终测试集
    - 已知类按照 train_ratio / val_ratio / test_ratio 进行样本级划分
    """

    def __init__(self, cfg: dict, seed: int = 42):
        """
        参数:
            cfg: 完整配置字典（来自 yaml）
            seed: 随机种子，用于可复现的样本划分
        """
        self.cfg = cfg
        self.seed = seed
        self.data_cfg = cfg["dataset"]
        self.train_cfg = cfg["train"]

        # 解析类别信息
        self._parse_classes()

        # 执行样本划分
        self._split_samples()

    def _parse_classes(self):
        """扫描数据集根目录，区分已知类与未知类"""
        root = self.data_cfg["root"]
        unknown_names = set(self.data_cfg.get("unknown_classes", []))

        # 获取所有子文件夹作为类别
        all_classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        # 区分已知类与未知类
        self.known_classes = [c for c in all_classes if c not in unknown_names]
        self.unknown_classes = [c for c in all_classes if c in unknown_names]

        # 验证未知类是否都存在于数据集中
        for unk in unknown_names:
            if unk not in all_classes:
                raise ValueError(
                    f"配置中指定的未知类 '{unk}' 在数据集根目录 '{root}' 中不存在！"
                )

        # 构建已知类的标签映射（未知类在测试时统一标记为 -1）
        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.known_classes)
        }
        self.idx_to_class: Dict[int, str] = {
            v: k for k, v in self.class_to_idx.items()
        }
        self.num_known_classes = len(self.known_classes)

        print(f"\n{'='*60}")
        print(f"  数据集类别信息")
        print(f"{'='*60}")
        print(f"  数据集根目录  : {self.data_cfg['root']}")
        print(f"  总类别数      : {len(all_classes)}")
        print(f"  已知类数量    : {self.num_known_classes}")
        print(f"  未知类数量    : {len(self.unknown_classes)}")
        print(f"  已知类列表    : {self.known_classes}")
        print(f"  未知类列表    : {self.unknown_classes}")
        print(f"{'='*60}\n")

    def _split_samples(self):
        """
        执行样本划分：
        - 已知类：按比例划分为 train / val / test
        - 未知类：100% 进入 test（标签统一为 -1）
        """
        root = self.data_cfg["root"]
        train_ratio = self.data_cfg["train_ratio"]
        val_ratio = self.data_cfg["val_ratio"]
        # test_ratio 不用显式使用，剩余即为 test

        rng = random.Random(self.seed)

        train_samples, val_samples, test_id_samples = [], [], []

        # ---------- 已知类：样本级划分 ----------
        for cls_name in self.known_classes:
            cls_dir = os.path.join(root, cls_name)
            label = self.class_to_idx[cls_name]

            # 收集该类所有 .npy 文件
            files = sorted([
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.endswith(".npy")
            ])

            if len(files) == 0:
                print(f"  警告：已知类 '{cls_name}' 目录下没有任何 .npy 文件，已跳过！")
                continue

            rng.shuffle(files)
            n = len(files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_files = files[:n_train]
            val_files = files[n_train: n_train + n_val]
            test_files = files[n_train + n_val:]

            train_samples.extend([(f, label) for f in train_files])
            val_samples.extend([(f, label) for f in val_files])
            test_id_samples.extend([(f, label) for f in test_files])

        # ---------- 未知类：全部进入 test（标签 = -1）----------
        test_ood_samples = []
        for cls_name in self.unknown_classes:
            cls_dir = os.path.join(root, cls_name)
            files = sorted([
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.endswith(".npy")
            ])
            test_ood_samples.extend([(f, -1) for f in files])

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_id_samples = test_id_samples
        self.test_ood_samples = test_ood_samples
        # 测试集 = 已知类测试样本 + 未知类全部样本
        self.test_samples = test_id_samples + test_ood_samples

        # 计算开放度（Openness）
        n_known = self.num_known_classes
        n_total = n_known + len(self.unknown_classes)
        openness = 1.0 - (n_known / n_total) ** 0.5 if n_total > 0 else 0.0

        print(f"\n{'='*60}")
        print(f"  样本划分信息")
        print(f"{'='*60}")
        print(f"  训练集样本数    : {len(self.train_samples)}")
        print(f"  验证集样本数    : {len(self.val_samples)}")
        print(f"  测试集（ID）    : {len(self.test_id_samples)}")
        print(f"  测试集（OOD）   : {len(self.test_ood_samples)}")
        print(f"  测试集总计      : {len(self.test_samples)}")
        print(f"  开放度(Openness): {openness:.4f}  "
              f"(公式: 1 - sqrt(K_known/K_total))")
        print(f"{'='*60}\n")

    def _build_transform(self, is_train: bool) -> transforms.Compose:
        """
        构建数据变换流水线
        
        参数:
            is_train: 是否为训练集（训练集可做数据增强）
        """
        single_ch = self.cfg["model"].get("single_channel_input", True)
        mean = self.data_cfg["normalize_mean"]
        std = self.data_cfg["normalize_std"]

        ops = []

        if is_train:
            # 训练集随机增强（STFT 图谱适用的增强方式）
            ops += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]

        # 归一化（对 Tensor 直接操作）
        ops.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(ops)

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = None,
    ) -> DataLoader:
        """
        获取指定分割的 DataLoader
        
        参数:
            split: 'train' | 'val' | 'test'
            shuffle: 是否打乱；默认 train=True，其余=False
        """
        split_map = {
            "train": self.train_samples,
            "val":   self.val_samples,
            "test":  self.test_samples,
        }
        if split not in split_map:
            raise ValueError(f"split 必须为 'train' / 'val' / 'test'，当前: {split}")

        samples = split_map[split]
        is_train = (split == "train")
        if shuffle is None:
            shuffle = is_train

        # 注意：test 集包含标签为 -1 的未知类样本，DataLoader 可正常处理
        dataset = UAVRFDataset(
            samples=samples,
            transform=self._build_transform(is_train),
            single_channel=self.cfg["model"].get("single_channel_input", True),
        )

        return DataLoader(
            dataset,
            batch_size=self.train_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=self.data_cfg["num_workers"],
            pin_memory=self.data_cfg["pin_memory"],
            drop_last=is_train,  # 训练时丢弃最后不完整的 batch
        )

    def get_all_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """返回 (train_loader, val_loader, test_loader) 三元组"""
        return (
            self.get_dataloader("train"),
            self.get_dataloader("val"),
            self.get_dataloader("test"),
        )
