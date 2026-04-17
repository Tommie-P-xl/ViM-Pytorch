"""
测试/评估主入口脚本
用法：python test.py --config configs/config.yaml --checkpoint outputs/best_model.pth

设计要点：
- 特征提取结果缓存到磁盘，避免重复前向传播
- 训练和测试无缝衔接：test.py 自动从配置文件获取检查点路径，
  若不手动指定 --checkpoint，则使用配置文件中 paths.best_model_name 指定的路径
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_config, set_seed, get_device, load_checkpoint, print_config
from datasets.uav_rf_dataset import DatasetBuilder
from models.model_factory import build_model
from evaluators.vim_scorer import FeatureExtractor, ViMScorer
from evaluators.metrics import OpenSetEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="ViM 无人机射频开集识别 - 测试脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型检查点路径（默认自动使用配置文件中 paths.best_model_name）",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="强制重新提取特征（忽略已有缓存）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 加载配置 ----
    cfg = load_config(args.config)
    print_config(cfg)

    # ---- 设置随机种子 ----
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    # ---- 选择计算设备 ----
    device = get_device()

    # ---- 确定检查点路径（无缝衔接训练和测试）----
    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(
            cfg["paths"]["output_dir"],
            cfg["paths"]["best_model_name"]
        )
    print(f"\n  使用检查点: {ckpt_path}")

    # ---- 构建数据集 ----
    print("\n  正在构建数据集...")
    builder = DatasetBuilder(cfg, seed=seed)
    cfg["_num_known_classes"] = builder.num_known_classes

    # 获取三个 DataLoader（test_loader 包含 ID 测试集 + OOD 全部样本）
    train_loader, val_loader, test_loader = builder.get_all_dataloaders()

    # ---- 构建模型并加载权重 ----
    model = build_model(cfg)
    model = model.to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    # 提取分类头参数（用于 ViM 算法）
    W, b = model.get_fc_params()
    print(f"\n  分类头参数: W.shape={W.shape}, b.shape={b.shape}")

    # ---- 特征提取（利用缓存避免重复前向传播）----
    cache_dir = cfg["paths"]["feature_cache_dir"]
    amp = cfg["train"].get("amp", False)

    extractor = FeatureExtractor(
        model=model,
        device=device,
        cache_dir=cache_dir,
        amp=amp,
    )

    print(f"\n  特征缓存目录: {cache_dir}")

    # 提取训练集特征（用于 ViM 参数拟合）
    train_data = extractor.extract_and_cache(
        train_loader, split_name="train", force_recompute=args.force_recompute
    )

    # 提取验证集特征（用于确定 OOD 检测阈值）
    val_data = extractor.extract_and_cache(
        val_loader, split_name="val", force_recompute=args.force_recompute
    )

    # 提取测试集特征（包含 ID 和 OOD 样本）
    test_data = extractor.extract_and_cache(
        test_loader, split_name="test", force_recompute=args.force_recompute
    )

    # ---- ViM 算法：拟合参数 ----
    print("\n  开始拟合 ViM 算法参数...")
    vim_scorer = ViMScorer(cfg)
    vim_scorer.fit(train_data, W, b)

    # ---- 计算各集合的 ViM 得分 ----
    print("  计算验证集 ViM 得分...")
    val_scores  = vim_scorer.score(val_data)

    print("  计算测试集 ViM 得分...")
    test_scores = vim_scorer.score(test_data)

    # ---- 确定 OOD 检测阈值（基于验证集 ID 样本）----
    tpr = cfg["vim"].get("tpr", 0.95)
    threshold = vim_scorer.compute_threshold(val_data, val_scores, tpr=tpr)

    # ---- 分离测试集中的 ID 和 OOD 样本 ----
    test_labels  = test_data["labels"]   # ID 样本标签 0~K-1，OOD 样本标签 -1
    test_logits  = test_data["logits"]

    id_mask  = (test_labels >= 0)   # 已知类样本掩码
    ood_mask = (test_labels == -1)  # 未知类样本掩码

    id_scores   = test_scores[id_mask]
    ood_scores  = test_scores[ood_mask]
    id_labels   = test_labels[id_mask]
    id_logits   = test_logits[id_mask]
    ood_labels  = test_labels[ood_mask]

    print(f"\n  测试集 ID 样本数  : {id_mask.sum()}")
    print(f"  测试集 OOD 样本数 : {ood_mask.sum()}")

    # ---- 全面评估 ----
    evaluator = OpenSetEvaluator(cfg, builder.idx_to_class)
    results = evaluator.evaluate(
        id_scores=id_scores,
        ood_scores=ood_scores,
        id_labels=id_labels,
        id_logits=id_logits,
        ood_labels=ood_labels,
        threshold=threshold,
        num_known=builder.num_known_classes,
        num_unknown=len(builder.unknown_classes),
    )

    print(f"\n  测试完成！评估结果已保存至: {cfg['paths']['result_txt']}")


if __name__ == "__main__":
    main()
