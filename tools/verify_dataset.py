"""
数据集结构验证脚本
用法：python tools/verify_dataset.py --config configs/config.yaml

功能：
- 扫描数据集目录，打印类别统计
- 验证所有 .npy 文件的形状是否符合预期（512×512）
- 打印开放度等信息
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="数据集结构验证工具")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--check_shape",
        action="store_true",
        help="是否验证每个 npy 文件的数据形状（较慢）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["dataset"]

    root = data_cfg["root"]
    unknown_names = set(data_cfg.get("unknown_classes", []))

    if not os.path.exists(root):
        print(f"❌ 数据集根目录不存在: {root}")
        return

    all_classes = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    known_classes   = [c for c in all_classes if c not in unknown_names]
    unknown_classes = [c for c in all_classes if c in unknown_names]

    print(f"\n{'='*60}")
    print(f"  数据集验证报告")
    print(f"{'='*60}")
    print(f"  根目录: {root}")
    print(f"  总类别: {len(all_classes)}")
    print(f"  已知类: {len(known_classes)}")
    print(f"  未知类: {len(unknown_classes)}")

    total_files = 0
    print(f"\n  各类别文件统计：")
    for cls in all_classes:
        cls_dir = os.path.join(root, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith(".npy")]
        flag = "（未知类）" if cls in unknown_names else "（已知类）"
        print(f"    {cls:<30s} {flag}: {len(files)} 个样本")
        total_files += len(files)

    print(f"\n  总样本数: {total_files}")

    # 验证数据形状
    if args.check_shape:
        print(f"\n  正在验证数据形状...")
        shape_errors = []
        for cls in tqdm(all_classes, desc="  验证中"):
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if not fname.endswith(".npy"):
                    continue
                fpath = os.path.join(cls_dir, fname)
                try:
                    data = np.load(fpath)
                    if data.shape != (512, 512):
                        shape_errors.append((fpath, data.shape))
                except Exception as e:
                    shape_errors.append((fpath, str(e)))

        if shape_errors:
            print(f"\n  ❌ 发现 {len(shape_errors)} 个异常文件：")
            for fpath, info in shape_errors[:10]:
                print(f"    {fpath}: {info}")
        else:
            print(f"\n  ✅ 所有文件形状均为 (512, 512)，验证通过！")

    n_known  = len(known_classes)
    n_total  = len(all_classes)
    openness = 1.0 - (n_known / n_total) ** 0.5 if n_total > 0 else 0.0

    print(f"\n  开放度 Openness: {openness:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
