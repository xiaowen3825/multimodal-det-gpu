"""训练引擎：训练循环、梯度累积、CPU优化策略。

CPU训练优化策略：
1. torch.set_num_threads 充分利用多核
2. torch.compile 编译加速
3. BFloat16 混合精度 (Intel CPU)
4. 梯度累积模拟大batch
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.checkpoint import save_checkpoint, load_checkpoint, cleanup_checkpoints
from utils.logger import MetricLogger
from .optimizer import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """训练引擎。

    Args:
        model: 检测器模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器 (可选)
        cfg: 完整配置
        evaluator: 评估器 (可选)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        cfg: dict,
        val_loader: Optional[DataLoader] = None,
        evaluator=None,
    ):
        # 自动检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.evaluator = evaluator

        # 训练参数
        train_cfg = cfg["train"]
        self.epochs = train_cfg.get("epochs", 50)
        self.grad_accum = train_cfg.get("gradient_accumulation", 8)
        self.clip_grad_norm = train_cfg.get("clip_grad_norm", 10.0)

        # GPU混合精度
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # CPU优化 (仅CPU时使用)
        if self.device.type == "cpu":
            self._setup_cpu_optimization(cfg.get("cpu", {}))
        else:
            self.use_bf16 = False

        # 优化器 & 调度器
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch=len(train_loader)
        )

        # Checkpoint
        ckpt_cfg = cfg.get("checkpoint", {})
        self.save_dir = ckpt_cfg.get("save_dir", "./runs")
        self.save_interval = ckpt_cfg.get("save_interval", 5)
        self.keep_last = ckpt_cfg.get("keep_last", 3)

        # 日志
        log_cfg = cfg.get("logging", {})
        self.log_interval = log_cfg.get("log_interval", 50)
        self.metric_logger = MetricLogger(
            log_dir=log_cfg.get("log_dir", "./runs/logs"),
            use_tensorboard=log_cfg.get("use_tensorboard", True),
            use_wandb=log_cfg.get("use_wandb", False),
            experiment_name=log_cfg.get("experiment_name", "default"),
        )

        # 状态
        self.start_epoch = 0
        self.global_step = 0
        self.best_ap = 0.0

    def _setup_cpu_optimization(self, cpu_cfg: dict):
        """配置CPU优化。"""
        # 线程数
        num_threads = cpu_cfg.get("num_threads", 0)
        if num_threads <= 0:
            num_threads = os.cpu_count() or 8
        torch.set_num_threads(num_threads)
        logger.info(f"CPU threads: {num_threads}")

        # torch.compile
        if cpu_cfg.get("enable_compile", False):
            try:
                mode = cpu_cfg.get("compile_mode", "reduce-overhead")
                self.model = torch.compile(self.model, mode=mode)
                logger.info(f"torch.compile enabled (mode={mode})")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, continuing without compilation")

        # BF16
        self.use_bf16 = cpu_cfg.get("enable_bf16", False)
        if self.use_bf16:
            if torch.cpu.is_available() and hasattr(torch.cpu.amp, "autocast"):
                logger.info("BFloat16 mixed precision enabled")
            else:
                logger.warning("BF16 not available, falling back to FP32")
                self.use_bf16 = False

    def train(self):
        """执行完整训练循环。"""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"  Gradient accumulation: {self.grad_accum}")
        logger.info(f"  Effective batch size: {self.cfg['train']['batch_size'] * self.grad_accum}")

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # 训练一个epoch
            train_metrics = self._train_epoch(epoch)

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}] completed in {epoch_time:.1f}s | "
                f"Loss: {train_metrics['total_loss']:.4f} | "
                f"CLS: {train_metrics['cls_loss']:.4f} | "
                f"BOX: {train_metrics['box_loss']:.4f}"
            )

            # 记录指标
            self.metric_logger.log_scalars(
                {f"train/{k}": v for k, v in train_metrics.items()},
                step=epoch + 1,
            )

            # 验证
            if self.val_loader is not None and self.evaluator is not None:
                if (epoch + 1) % self.save_interval == 0 or epoch == self.epochs - 1:
                    val_metrics = self.evaluator.evaluate(self.model, self.val_loader)
                    logger.info(f"  Val metrics: {val_metrics}")
                    self.metric_logger.log_scalars(
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        step=epoch + 1,
                    )

                    # 保存最优
                    ap = val_metrics.get("AP", 0)
                    if ap > self.best_ap:
                        self.best_ap = ap
                        save_checkpoint(
                            self.model, self.optimizer, epoch + 1,
                            val_metrics,
                            os.path.join(self.save_dir, "best.pth"),
                        )

            # 定期保存
            if (epoch + 1) % self.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1,
                    train_metrics,
                    os.path.join(self.save_dir, f"epoch_{epoch+1}.pth"),
                )
                cleanup_checkpoints(self.save_dir, self.keep_last)

        logger.info(f"Training finished. Best AP: {self.best_ap:.4f}")
        self.metric_logger.close()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch。"""
        self.model.train()
        running_losses = {"total_loss": 0, "cls_loss": 0, "box_loss": 0, "obj_loss": 0}
        n_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in batch["targets"]]
            class_names = batch["class_names"]

            # 前向传播 (GPU AMP 或 CPU BF16)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    losses = self.model(
                        images=images,
                        text_prompts=class_names,
                        targets=targets,
                    )
            elif self.use_bf16:
                with torch.cpu.amp.autocast(dtype=torch.bfloat16):
                    losses = self.model(
                        images=images,
                        text_prompts=class_names,
                        targets=targets,
                    )
            else:
                losses = self.model(
                    images=images,
                    text_prompts=class_names,
                    targets=targets,
                )

            # 梯度累积
            loss = losses["total_loss"] / self.grad_accum

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 累积够了才更新
            if (step + 1) % self.grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm
                        )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # 累计指标
            for k in running_losses:
                if k in losses:
                    running_losses[k] += losses[k].item()
            n_steps += 1

            # 打印日志
            if (step + 1) % self.log_interval == 0:
                avg = {k: v / n_steps for k, v in running_losses.items()}
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  [{epoch+1}][{step+1}/{len(self.train_loader)}] "
                    f"loss={avg['total_loss']:.4f} "
                    f"cls={avg['cls_loss']:.4f} "
                    f"box={avg['box_loss']:.4f} "
                    f"lr={lr:.6f}"
                )

        return {k: v / max(n_steps, 1) for k, v in running_losses.items()}

    def resume(self, checkpoint_path: str):
        """从checkpoint恢复训练。"""
        info = load_checkpoint(self.model, checkpoint_path, self.optimizer)
        self.start_epoch = info.get("epoch", 0)
        self.best_ap = info.get("metrics", {}).get("AP", 0)
        logger.info(f"Resumed from epoch {self.start_epoch}, best AP: {self.best_ap:.4f}")
