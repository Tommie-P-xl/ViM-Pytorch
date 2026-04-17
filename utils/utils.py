"""
工具函数模块：配置文件读取、随机种子设置、设备选择等
"""

import os
import random
import numpy as np
import torch
import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    读取 YAML 配置文件并返回配置字典
    
    参数:
        config_path: yaml 文件路径
    返回:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def set_seed(seed: int):
    """
    设置全局随机种子，确保实验可复现
    
    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 操作的确定性（会略微影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    自动选择计算设备
    
    参数:
        prefer_cuda: 是否优先使用 GPU
    返回:
        torch.device 实例
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  使用 GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print(f"  使用 CPU")
    return device


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> dict:
    """
    加载模型检查点
    
    参数:
        model: 目标模型
        checkpoint_path: 检查点文件路径
        device: 加载到的设备
        strict: 是否严格匹配参数名（默认 True）
    返回:
        检查点字典（包含 epoch、val_acc 等元数据）
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)

    print(f"  已加载检查点: {checkpoint_path}")
    print(f"  训练 Epoch: {ckpt.get('epoch', '未知')}")
    print(f"  验证准确率: {ckpt.get('val_acc', '未知'):.4f}")

    return ckpt


def print_config(cfg: dict):
    """
    以可读格式打印配置信息
    
    参数:
        cfg: 配置字典
    """
    print(f"\n{'='*60}")
    print(f"  运行配置")
    print(f"{'='*60}")
    _print_dict_recursive(cfg, indent=2)
    print(f"{'='*60}\n")


def _print_dict_recursive(d: dict, indent: int = 0):
    """递归打印嵌套字典"""
    for k, v in d.items():
        if k.startswith("_"):
            continue  # 跳过内部注入的私有键
        if isinstance(v, dict):
            print(f"{' ' * indent}{k}:")
            _print_dict_recursive(v, indent + 2)
        else:
            print(f"{' ' * indent}{k}: {v}")
