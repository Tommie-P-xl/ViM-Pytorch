"""
ViM（Virtual-logit Matching）核心算法模块
实现特征提取缓存、主子空间构建、虚拟逻辑值计算及 OOD 检测
"""

import os
import time
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from tqdm import tqdm


# ==============================================================================
# 特征提取与缓存模块（避免重复前向传播）
# ==============================================================================

class FeatureExtractor:
    """
    负责将 DataLoader 中的样本全部经过网络前向传播，
    提取并缓存特征向量（全连接层之前）和对应的 logit/标签。
    
    核心设计：特征一经提取即缓存到磁盘，后续 ViM 计算直接加载，
    避免重复耗时的网络前向传播。
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cache_dir: str,
        amp: bool = False,
    ):
        self.model = model
        self.device = device
        self.cache_dir = cache_dir
        self.amp = amp
        os.makedirs(cache_dir, exist_ok=True)

    @torch.no_grad()
    def extract_and_cache(
        self,
        loader,
        split_name: str,
        force_recompute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        提取并缓存指定数据分割的特征、logit 和标签。
        若缓存已存在且 force_recompute=False，直接加载缓存。
        
        参数:
            loader: DataLoader
            split_name: 缓存文件的标识（如 'train' / 'val' / 'test'）
            force_recompute: 是否强制重新计算（忽略缓存）
        返回:
            {
                'features': (N, D) float32,
                'logits':   (N, C) float32,
                'labels':   (N,)   int64,
            }
        """
        cache_path = os.path.join(self.cache_dir, f"{split_name}_features.npz")

        # 检查是否存在有效缓存
        if os.path.exists(cache_path) and not force_recompute:
            print(f"  加载特征缓存: {cache_path}")
            data = np.load(cache_path)
            return {
                "features": data["features"],
                "logits":   data["logits"],
                "labels":   data["labels"],
            }

        print(f"  提取 '{split_name}' 集特征（前向传播中...）")
        self.model.eval()

        all_features = []
        all_logits   = []
        all_labels   = []

        t0 = time.time()
        pbar = tqdm(loader, desc=f"  提取特征 [{split_name}]", dynamic_ncols=True)

        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(self.amp and self.device.type == "cuda")):
                logits, features = self.model(inputs, return_feature=True)

            all_features.append(features.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.numpy())

        features_arr = np.concatenate(all_features, axis=0).astype(np.float32)
        logits_arr   = np.concatenate(all_logits,   axis=0).astype(np.float32)
        labels_arr   = np.concatenate(all_labels,   axis=0).astype(np.int64)

        elapsed = time.time() - t0
        print(f"  特征提取完成！样本数: {len(labels_arr)}，耗时: {elapsed:.1f}s")
        print(f"  特征维度: {features_arr.shape[1]}")

        # 保存缓存
        np.savez_compressed(cache_path, features=features_arr, logits=logits_arr, labels=labels_arr)
        print(f"  特征缓存已保存: {cache_path}")

        return {
            "features": features_arr,
            "logits":   logits_arr,
            "labels":   labels_arr,
        }


# ==============================================================================
# ViM 算法核心
# ==============================================================================

class ViMScorer:
    """
    ViM (Virtual-logit Matching) 算法实现
    
    流程：
    1. 从分类头提取 W 和 b，计算原点 o = -(W^T)^+ b（消除偏置）
    2. 利用训练集特征构建主子空间（PCA）
    3. 计算各样本的特征残差范数
    4. 计算缩放常数 alpha
    5. 计算 ViM 得分（Energy - Residual）
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.vim_cfg = cfg["vim"]

        # 算法中间量（fit 后填充）
        self.origin_o: Optional[np.ndarray] = None  # 坐标系原点
        self.NS: Optional[np.ndarray]       = None  # 正交补空间基矩阵 R
        self.alpha: Optional[float]         = None  # 缩放常数

        # 分类头参数
        self.W: Optional[np.ndarray] = None  # (C, D)
        self.b: Optional[np.ndarray] = None  # (C,)

    # --------------------------------------------------------------------------
    # 拟合：在训练集上确定所有算法参数
    # --------------------------------------------------------------------------

    def fit(
        self,
        train_data: Dict[str, np.ndarray],
        W: np.ndarray,
        b: np.ndarray,
    ):
        """
        在训练集特征上拟合 ViM 算法的所有参数
        
        参数:
            train_data: extract_and_cache 返回的训练集数据字典
            W: 分类头权重 (C, D)
            b: 分类头偏置 (C,)
        """
        print(f"\n{'='*60}")
        print(f"  ViM 算法参数拟合（基于训练集特征）")
        print(f"{'='*60}")

        self.W = W
        self.b = b

        features = train_data["features"]  # (N, D)
        logits   = train_data["logits"]    # (N, C)

        feature_dim = features.shape[1]

        # ---- 步骤 1：计算坐标原点 o = -(W^T)^+ b ----
        # W^T 的维度为 (D, C)，广义逆维度为 (C, D)
        print("  步骤 1: 计算坐标原点（消除分类偏置）...")
        print(f"  W 形状: {W.shape}, b 形状: {b.shape}")
        W_T = W.T  # (D, C)
        print(f"  W_T 形状: {W_T.shape}")
        print(f"  pinv(W_T) 形状: {pinv(W_T).shape}")
        # 正确的计算：o = -(W^T)^+ b
        # pinv(W_T) 形状为 (C, D)，b 形状为 (C,)，所以应该是 b @ pinv(W_T)
        self.origin_o = -(b @ pinv(W_T))  # (D,)
        print(f"  原点向量 o 的范数: {norm(self.origin_o):.4f}")

        # ---- 步骤 2：构建主子空间 ----
        # 决定主子空间维度 D
        dim_cfg = self.vim_cfg.get("dim", -1)
        if dim_cfg == -1:
            if feature_dim >= 2048:
                DIM = 1000
            elif feature_dim >= 768:
                DIM = 512
            else:
                DIM = feature_dim // 2
        else:
            DIM = int(dim_cfg)

        print(f"  步骤 2: 构建主子空间 (特征维度={feature_dim}, 主子空间维度 D={DIM})...")

        # 将训练特征平移到新坐标系
        features_centered = features - self.origin_o  # (N, D)

        # 计算 X^T X 并进行特征值分解
        cov = features_centered.T @ features_centered  # (D, D)
        eig_vals, eig_vecs = np.linalg.eigh(cov)       # 实对称矩阵用 eigh

        # eigh 返回升序排列，取最大的 DIM 个对应的补空间（较小的特征值方向）
        # 降序排列：从最大到最小
        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_vecs_sorted = eig_vecs[:, sorted_idx]  # (D, D)

        # 正交补空间基矩阵 R：去掉前 DIM 个主成分，取剩余列
        self.NS = np.ascontiguousarray(
            eig_vecs_sorted[:, DIM:]  # (D, D-DIM)
        )
        print(f"  正交补空间基矩阵 NS 维度: {self.NS.shape}")

        # ---- 步骤 3：计算 alpha ----
        print("  步骤 3: 计算缩放常数 alpha...")

        # 训练集最大 logit 的均值
        max_logits = logits.max(axis=-1)  # (N,)
        mean_max_logit = max_logits.mean()

        # 训练集特征残差范数的均值
        residuals = self._compute_residual_norms(features_centered)  # (N,)
        mean_residual = residuals.mean()

        self.alpha = float(mean_max_logit / (mean_residual + 1e-8))

        print(f"  平均最大 logit: {mean_max_logit:.4f}")
        print(f"  平均残差范数:   {mean_residual:.4f}")
        print(f"  缩放常数 alpha: {self.alpha:.4f}")
        print(f"{'='*60}\n")

    # --------------------------------------------------------------------------
    # 计算残差范数（内部辅助函数）
    # --------------------------------------------------------------------------

    def _compute_residual_norms(self, features_centered: np.ndarray) -> np.ndarray:
        """
        计算特征在正交补空间上投影的范数
        
        参数:
            features_centered: 已减去原点 o 的特征矩阵 (N, D)
        返回:
            残差范数数组 (N,)
        """
        # x^{P⊥} = R(R^T x)，范数为 || R^T x ||
        # 因为 R 的列向量是正交归一的，||R(R^T x)|| = ||R^T x||
        projections = features_centered @ self.NS  # (N, D-DIM)
        return norm(projections, axis=-1)           # (N,)

    # --------------------------------------------------------------------------
    # 对任意样本集计算 ViM 得分
    # --------------------------------------------------------------------------

    def score(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算数据集的 ViM 得分
        得分 = Energy - Residual（分数越高 → 越像 ID 样本）
        
        参数:
            data: extract_and_cache 返回的数据字典
        返回:
            vim_scores: (N,) float32
        """
        assert self.NS is not None, "请先调用 fit() 方法！"

        features = data["features"]  # (N, D)
        logits   = data["logits"]    # (N, C)

        # 将特征平移到新坐标系
        features_centered = features - self.origin_o

        # 计算残差范数并缩放
        residual_norms = self._compute_residual_norms(features_centered)  # (N,)
        virtual_logit  = self.alpha * residual_norms                       # (N,)

        # 计算能量分 (Energy score)
        energy = logsumexp(logits, axis=-1)  # (N,)

        # ViM 得分 = Energy - Residual（单调等价于 ViM 概率）
        vim_scores = energy - virtual_logit

        return vim_scores.astype(np.float32)

    # --------------------------------------------------------------------------
    # 基于 ID 验证集确定 OOD 检测阈值
    # --------------------------------------------------------------------------

    def compute_threshold(
        self,
        val_data: Dict[str, np.ndarray],
        val_scores: np.ndarray,
        tpr: float = 0.95,
    ) -> float:
        """
        在 ID 验证集上，基于 TPR 确定 OOD 检测阈值
        
        参数:
            val_data: 验证集数据字典（标签均 >= 0，仅 ID 样本）
            val_scores: 验证集 ViM 得分 (N,)
            tpr: 期望的真阳性率（即有多少比例的 ID 样本得分 >= 阈值）
        返回:
            threshold: OOD 检测阈值
        """
        # val 集仅含已知类（标签 >= 0），无需过滤
        # 阈值 = 得分的第 (1-tpr) 百分位数
        threshold = float(np.percentile(val_scores, 100.0 * (1.0 - tpr)))
        print(f"  OOD 阈值（TPR={tpr:.0%}）: {threshold:.4f}")
        return threshold
