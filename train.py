"""
训练主入口脚本
用法：python train.py --config configs/config.yaml
"""

import argparse
import os
import sys

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import load_config, set_seed, get_device, print_config
from datasets.uav_rf_dataset import DatasetBuilder
from models.model_factory import build_model
from models.optim_factory import build_optimizer, build_scheduler
from trainers.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="ViM 无人机射频开集识别 - 训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定检查点恢复训练（传入 .pth 文件路径）",
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
    print(f"  随机种子: {seed}")

    # ---- 选择计算设备 ----
    device = get_device()

    # ---- 构建数据集 ----
    print("\n  正在构建数据集...")
    builder = DatasetBuilder(cfg, seed=seed)

    # 将已知类别数注入配置（供模型工厂使用）
    cfg["_num_known_classes"] = builder.num_known_classes

    train_loader, val_loader, _ = builder.get_all_dataloaders()

    # ---- 构建模型 ----
    model = build_model(cfg)
    model = model.to(device)

    # ---- 构建优化器和调度器 ----
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # ---- 恢复训练（可选）----
    start_epoch = 0
    if args.resume is not None:
        import torch
        from utils.utils import load_checkpoint
        ckpt = load_checkpoint(model, args.resume, device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  从 Epoch {start_epoch} 继续训练")

    # ---- 开始训练 ----
    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    best_ckpt_path = trainer.train()

    print(f"\n  训练完成！最优检查点: {best_ckpt_path}")
    print(f"  可以运行以下命令进行测试：")
    print(f"    python test.py --config {args.config} --checkpoint {best_ckpt_path}")


if __name__ == "__main__":
    main()
