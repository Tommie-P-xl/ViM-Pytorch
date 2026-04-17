"""
生成虚拟测试数据集脚本
用法：python tools/generate_dummy_dataset.py

功能：在 data/uav_rf/ 目录下生成若干类别的虚拟 STFT npy 文件，
用于快速验证代码流程是否正确（无需真实数据）。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_dummy_dataset(
    root: str = "data/uav_rf",
    known_classes: list = None,
    unknown_classes: list = None,
    samples_per_class: int = 50,
    shape: tuple = (512, 512),
    seed: int = 42,
):
    """生成虚拟数据集"""
    np.random.seed(seed)

    if known_classes is None:
        known_classes = ["drone_DJI", "drone_Parrot", "drone_Autel", "wifi_bg"]
    if unknown_classes is None:
        unknown_classes = ["unknown_drone_A", "unknown_drone_B"]

    all_classes = known_classes + unknown_classes
    os.makedirs(root, exist_ok=True)

    print(f"  生成虚拟数据集到: {root}")
    print(f"  已知类: {known_classes}")
    print(f"  未知类: {unknown_classes}")
    print(f"  每类样本数: {samples_per_class}")
    print(f"  数据形状: {shape}")

    for cls in all_classes:
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(samples_per_class):
            # 生成带类别区分度的随机 STFT 数据
            data = np.random.randn(*shape).astype(np.float32)
            # 加入类别偏移以模拟类别间差异
            data += known_classes.index(cls) if cls in known_classes else len(known_classes)
            fpath = os.path.join(cls_dir, f"sample_{i:04d}.npy")
            np.save(fpath, data)
        print(f"    {cls}: {samples_per_class} 个样本 ✅")

    print(f"\n  虚拟数据集生成完成！")
    print(f"  请确保 configs/config.yaml 中 dataset.root 设置为: {root}")
    print(f"  并将 dataset.unknown_classes 设置为: {unknown_classes}")


if __name__ == "__main__":
    generate_dummy_dataset()
