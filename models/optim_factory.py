"""
优化器和学习率调度器工厂模块
支持：SGD | Adam | AdamW
调度器支持：CosineAnnealingLR | StepLR | MultiStepLR | ReduceLROnPlateau | WarmupCosine
"""

import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    MultiStepLR,
    ReduceLROnPlateau,
    LambdaLR,
    _LRScheduler,
)
from typing import Tuple, Union


# ==============================================================================
# 优化器工厂
# ==============================================================================

def build_optimizer(cfg: dict, model: nn.Module) -> Optimizer:
    """
    根据配置文件构建优化器
    
    参数:
        cfg: 完整配置字典
        model: 待优化的模型
    返回:
        PyTorch Optimizer 实例
    """
    opt_cfg = cfg["train"]["optimizer"]
    opt_type = opt_cfg["type"].lower()
    lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    # 将模型参数分为"有权重衰减"和"无权重衰减"两组
    # Bias 和 LayerNorm/BatchNorm 的参数通常不做权重衰减
    decay_params, no_decay_params = _separate_param_groups(model)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if opt_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            nesterov=bool(opt_cfg.get("nesterov", True)),
        )

    elif opt_type == "adam":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=float(opt_cfg.get("eps", 1e-8)),
        )

    elif opt_type == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=float(opt_cfg.get("eps", 1e-8)),
        )

    else:
        raise ValueError(
            f"不支持的优化器类型: '{opt_type}'，可选: sgd | adam | adamw"
        )

    print(f"  优化器        : {opt_type.upper()}  (lr={lr}, weight_decay={weight_decay})")
    return optimizer


def _separate_param_groups(model: nn.Module):
    """将模型参数分为需要权重衰减和不需要权重衰减两组"""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # bias 和归一化层参数不做权重衰减
        if "bias" in name or "bn" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params


# ==============================================================================
# 学习率调度器工厂
# ==============================================================================

def build_scheduler(
    cfg: dict,
    optimizer: Optimizer,
) -> Union[_LRScheduler, ReduceLROnPlateau, None]:
    """
    根据配置文件构建学习率调度器
    
    参数:
        cfg: 完整配置字典
        optimizer: 优化器实例
    返回:
        PyTorch LRScheduler 实例，或 None（scheduler.type = "none" 时）
    """
    sched_cfg = cfg["train"]["scheduler"]
    sched_type = sched_cfg["type"].lower()
    epochs = cfg["train"]["epochs"]

    if sched_type == "none":
        print(f"  调度器        : 无（固定学习率）")
        return None

    elif sched_type == "cosine":
        t_max = int(sched_cfg.get("t_max", epochs))
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        print(f"  调度器        : CosineAnnealingLR  (T_max={t_max}, eta_min={eta_min})")

    elif sched_type == "step":
        step_size = int(sched_cfg.get("step_size", 30))
        gamma = float(sched_cfg.get("gamma", 0.1))
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"  调度器        : StepLR  (step_size={step_size}, gamma={gamma})")

    elif sched_type == "multistep":
        milestones = list(sched_cfg.get("milestones", [40, 70, 90]))
        gamma = float(sched_cfg.get("gamma", 0.1))
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        print(f"  调度器        : MultiStepLR  (milestones={milestones}, gamma={gamma})")

    elif sched_type == "plateau":
        patience = int(sched_cfg.get("patience", 10))
        factor = float(sched_cfg.get("factor", 0.5))
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=patience, factor=factor, verbose=True
        )
        print(f"  调度器        : ReduceLROnPlateau  (patience={patience}, factor={factor})")

    elif sched_type == "warmup_cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        scheduler = _build_warmup_cosine(optimizer, warmup_epochs, epochs, eta_min)
        print(f"  调度器        : WarmupCosine  (warmup={warmup_epochs}, epochs={epochs})")

    else:
        raise ValueError(
            f"不支持的调度器类型: '{sched_type}'，"
            f"可选: cosine | step | multistep | plateau | warmup_cosine | none"
        )

    return scheduler


def _build_warmup_cosine(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min: float = 1e-6,
) -> LambdaLR:
    """
    构建带线性 Warmup 的余弦退火调度器
    前 warmup_epochs 个 epoch 线性升温，之后余弦退火
    """
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # 线性升温：从 0 增长到 base_lr
            return float(epoch + 1) / float(warmup_epochs)
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = eta_min / base_lr
            return max(min_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ==============================================================================
# 调度器步进辅助函数（统一处理 ReduceLROnPlateau 和其他调度器的差异）
# ==============================================================================

def scheduler_step(
    scheduler,
    metric: float = None,
) -> None:
    """
    统一调用调度器的 step 方法
    
    参数:
        scheduler: 调度器实例（可为 None）
        metric: 当调度器为 ReduceLROnPlateau 时，需要传入监控指标
    """
    if scheduler is None:
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        if metric is not None:
            scheduler.step(metric)
    else:
        scheduler.step()
