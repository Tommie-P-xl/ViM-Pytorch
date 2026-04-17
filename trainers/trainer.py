"""
训练器模块：负责模型的训练循环、验证循环、检查点保存和 TensorBoard 记录
"""

import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.optim_factory import scheduler_step
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ==============================================================================
# 训练器类
# ==============================================================================

class Trainer:
    """
    封装训练和验证的完整流程，支持：
    - 混合精度训练（AMP）
    - TensorBoard 记录
    - tqdm 进度条
    - 自动保存最优模型检查点
    - 梯度裁剪
    """

    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 训练配置
        self.epochs = cfg["train"]["epochs"]
        self.amp = cfg["train"].get("amp", False)
        self.grad_clip = float(cfg["train"].get("grad_clip", 0.0))

        # 路径配置
        output_dir = cfg["paths"]["output_dir"]
        tb_dir = cfg["paths"]["tensorboard_dir"]
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        self.best_model_path = os.path.join(output_dir, cfg["paths"]["best_model_name"])

        # TensorBoard
        absolute_tb_dir = os.path.abspath(tb_dir)
        self.writer = SummaryWriter(log_dir=absolute_tb_dir)

        # 损失函数：交叉熵
        self.criterion = nn.CrossEntropyLoss()

        # AMP 梯度缩放器
        self.scaler = torch.cuda.amp.GradScaler() if self.amp and device.type == "cuda" else None

        # 训练状态追踪
        self.best_val_acc = 0.0
        self.best_epoch = 0

    # --------------------------------------------------------------------------
    # 训练一个 epoch
    # --------------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        训练单个 epoch，返回 (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"[Epoch {epoch+1:03d}/{self.epochs}] 训练",
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                # 混合精度前向
                with torch.cuda.amp.autocast():
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                self.optimizer.step()

            # 统计准确率
            preds = logits.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            correct += batch_correct
            total += batch_size

            # 更新进度条显示
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{batch_correct / batch_size:.3f}",
                "lr":   f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        avg_loss = total_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc

    # --------------------------------------------------------------------------
    # 验证一个 epoch
    # --------------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, epoch: int) -> Tuple[float, float]:
        """
        在验证集上评估，返回 (平均损失, 准确率)
        注意：验证集只包含已知类样本，标签均 >= 0
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"[Epoch {epoch+1:03d}/{self.epochs}] 验证",
            dynamic_ncols=True,
            leave=False,
        )

        for inputs, labels in pbar:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / total
        avg_acc = correct / total
        return avg_loss, avg_acc

    # --------------------------------------------------------------------------
    # 保存检查点
    # --------------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool):
        """保存模型检查点"""
        ckpt = {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cfg": self.cfg,
        }
        if is_best:
            torch.save(ckpt, self.best_model_path)
            print(f"  ✅ 保存最优模型 → {self.best_model_path}  "
                  f"(val_acc={val_acc:.4f})")

    # --------------------------------------------------------------------------
    # 完整训练流程入口
    # --------------------------------------------------------------------------

    def train(self):
        """执行完整的训练流程"""
        print(f"\n{'='*60}")
        print(f"  开始训练  (共 {self.epochs} 个 epoch)")
        print(f"  设备: {self.device}  AMP: {self.amp}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.epochs):
            # 训练
            train_loss, train_acc = self._train_one_epoch(epoch)

            # 验证
            val_loss, val_acc = self._validate(epoch)

            # 学习率调度
            scheduler_step(
                self.scheduler,
                metric=val_acc if isinstance(self.scheduler, ReduceLROnPlateau) else None
            )

            current_lr = self.optimizer.param_groups[0]["lr"]

            # TensorBoard 记录
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val",   val_loss,   epoch)
            self.writer.add_scalar("Acc/train",  train_acc,  epoch)
            self.writer.add_scalar("Acc/val",    val_acc,    epoch)
            self.writer.add_scalar("LR",         current_lr, epoch)

            # 终端输出本 epoch 摘要
            print(
                f"  Epoch [{epoch+1:03d}/{self.epochs}] "
                f"| train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} "
                f"| val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} "
                f"| lr: {current_lr:.2e}"
            )

            # 保存最优检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
            self._save_checkpoint(epoch, val_acc, is_best)

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  训练完成！总耗时: {elapsed/60:.1f} 分钟")
        print(f"  最优验证准确率: {self.best_val_acc:.4f}  (Epoch {self.best_epoch})")
        print(f"  最优模型已保存至: {self.best_model_path}")
        print(f"{'='*60}\n")

        self.writer.close()

        return self.best_model_path
