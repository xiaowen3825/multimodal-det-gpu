"""文本编码器知识蒸馏训练器。

从大规模CLIP文本编码器(Teacher)蒸馏到轻量4层Transformer(Student)。

蒸馏策略:
1. 特征蒸馏损失 (MSE): 对齐Student和Teacher的[CLS]嵌入
2. 对比对齐损失: 保持Student输出的文本嵌入空间结构与Teacher一致

参考文献:
- CLIP-KD (CVPR 2024): An Empirical Study of CLIP Model Distillation
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """纯文本数据集，用于蒸馏训练。

    每行一条文本，从CC3M等数据集提取。

    Args:
        text_file: 文本文件路径（每行一条文本）
        num_samples: 使用样本数（-1为全部）
    """

    def __init__(self, text_file: str, num_samples: int = -1):
        with open(text_file, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f if line.strip()]

        if num_samples > 0:
            self.texts = self.texts[:num_samples]

        logger.info(f"Loaded {len(self.texts)} texts from {text_file}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class DistillTrainer:
    """文本编码器蒸馏训练器。

    Args:
        teacher: CLIP文本编码器 (Teacher, 冻结)
        student: 轻量文本编码器 (Student, 可训练)
        cfg: 蒸馏配置字典
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        cfg: dict,
    ):
        self.teacher = teacher
        self.student = student
        self.cfg = cfg

        # 确保Teacher冻结
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 蒸馏超参数
        distill_cfg = cfg.get("distill", {})
        self.feat_loss_weight = distill_cfg.get("feat_loss_weight", 1.0)
        self.align_loss_weight = distill_cfg.get("align_loss_weight", 0.5)
        self.temperature = distill_cfg.get("temperature", 2.0)

        # 训练参数
        train_cfg = cfg.get("train", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.batch_size = train_cfg.get("batch_size", 64)
        self.lr = train_cfg.get("initial_lr", 0.001)
        self.final_lr = train_cfg.get("final_lr", 0.00001)

        # 保存路径
        ckpt_cfg = cfg.get("checkpoint", {})
        self.save_dir = ckpt_cfg.get("save_dir", "./runs/distill")

    def train(self, text_file: str, num_samples: int = 100000):
        """执行蒸馏训练。

        Args:
            text_file: 训练文本文件路径
            num_samples: 使用样本数
        """
        # 构建数据集
        dataset = TextDataset(text_file, num_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=lambda batch: batch,  # 返回文本列表
        )

        # 优化器
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )

        # 学习率调度器
        total_steps = len(dataloader) * self.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.final_lr,
        )

        logger.info(f"Starting distillation training:")
        logger.info(f"  Epochs: {self.epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Student params: {sum(p.numel() for p in self.student.parameters()) / 1e6:.1f}M")

        best_loss = float("inf")

        for epoch in range(self.epochs):
            self.student.train()
            epoch_loss = 0.0
            epoch_feat_loss = 0.0
            epoch_align_loss = 0.0

            for step, texts in enumerate(dataloader):
                # Teacher前向 (no grad)
                with torch.no_grad():
                    teacher_embeds = self.teacher(texts=texts)  # [B, D]

                # Student前向
                student_embeds = self.student(texts=texts)  # [B, D]

                # 计算损失
                loss, losses = self._compute_loss(student_embeds, teacher_embeds)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                epoch_feat_loss += losses["feat_loss"]
                epoch_align_loss += losses["align_loss"]

                if (step + 1) % 100 == 0:
                    logger.info(
                        f"Epoch [{epoch+1}/{self.epochs}] Step [{step+1}/{len(dataloader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"Feat: {losses['feat_loss']:.4f} "
                        f"Align: {losses['align_loss']:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )

            # Epoch统计
            n_steps = len(dataloader)
            avg_loss = epoch_loss / n_steps
            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Avg Loss: {avg_loss:.4f} "
                f"Feat: {epoch_feat_loss/n_steps:.4f} "
                f"Align: {epoch_align_loss/n_steps:.4f}"
            )

            # 保存最优模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(epoch, avg_loss, is_best=True)

            # 定期保存
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, avg_loss, is_best=False)

        logger.info(f"Distillation training finished. Best loss: {best_loss:.4f}")

    def _compute_loss(
        self,
        student_embeds: torch.Tensor,
        teacher_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """计算蒸馏损失。

        Args:
            student_embeds: Student输出 [B, D]
            teacher_embeds: Teacher输出 [B, D]

        Returns:
            (total_loss, {feat_loss, align_loss})
        """
        # 1. 特征蒸馏损失 (MSE)
        # 归一化后计算MSE，关注方向而非尺度
        s_norm = F.normalize(student_embeds, dim=-1)
        t_norm = F.normalize(teacher_embeds, dim=-1)
        feat_loss = F.mse_loss(s_norm, t_norm)

        # 2. 对比对齐损失
        # 保持batch内样本间的相对关系一致
        # Teacher的相似度矩阵作为soft target
        t_sim = torch.mm(t_norm, t_norm.t()) / self.temperature  # [B, B]
        s_sim = torch.mm(s_norm, s_norm.t()) / self.temperature  # [B, B]

        # KL散度 (行和列都做)
        t_prob = F.softmax(t_sim, dim=-1)
        s_log_prob = F.log_softmax(s_sim, dim=-1)
        align_loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean")

        # 总损失
        total_loss = self.feat_loss_weight * feat_loss + self.align_loss_weight * align_loss

        return total_loss, {
            "feat_loss": feat_loss.item(),
            "align_loss": align_loss.item(),
        }

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """保存蒸馏checkpoint。"""
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "model_state_dict": self.student.state_dict(),
            "loss": loss,
        }

        if is_best:
            path = os.path.join(self.save_dir, "best_text_encoder.pth")
        else:
            path = os.path.join(self.save_dir, f"text_encoder_epoch_{epoch+1}.pth")

        torch.save(state, path)
        logger.info(f"{'Best ' if is_best else ''}Checkpoint saved: {path}")
