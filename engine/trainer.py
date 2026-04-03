"""训练引擎：训练循环、梯度累积、GPU优化策略。

GPU训练优化策略：
1. CUDA自动混合精度 (AMP) + GradScaler
2. torch.compile 编译加速
3. cuDNN benchmark 自动选择最优卷积算法
4. pin_memory 加速CPU->GPU数据传输
5. 梯度累积模拟更大batch（可选）
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # compile失败时自动降级为eager模式
from torch.utils.data import DataLoader

from utils.checkpoint import save_checkpoint, load_checkpoint, cleanup_checkpoints
from utils.logger import MetricLogger
from .optimizer import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """训练引擎（GPU版本）。

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
        self.cfg = cfg

        # GPU设备配置
        self.device = self._setup_device(cfg.get("gpu", {}))

        # 将模型移到GPU
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        # 训练参数
        train_cfg = cfg["train"]
        self.epochs = train_cfg.get("epochs", 50)
        self.grad_accum = train_cfg.get("gradient_accumulation", 1)
        self.clip_grad_norm = train_cfg.get("clip_grad_norm", 10.0)

        # GPU优化
        self._setup_gpu_optimization(cfg.get("gpu", {}))

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

    def _setup_device(self, gpu_cfg: dict) -> torch.device:
        """配置GPU设备。"""
        device_str = gpu_cfg.get("device", "cuda")

        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            logger.info(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")

            # cuDNN benchmark
            if gpu_cfg.get("cudnn_benchmark", True):
                torch.backends.cudnn.benchmark = True
                logger.info("cuDNN benchmark enabled")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, falling back to CPU")

        return device

    def _setup_gpu_optimization(self, gpu_cfg: dict):
        """配置GPU优化策略。"""
        # 混合精度训练 (AMP)
        self.use_amp = gpu_cfg.get("enable_amp", True) and self.device.type == "cuda"
        if self.use_amp:
            amp_dtype_str = gpu_cfg.get("amp_dtype", "float16")
            if amp_dtype_str == "bfloat16":
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
            self.grad_scaler = torch.amp.GradScaler("cuda")
            logger.info(f"AMP enabled (dtype={amp_dtype_str})")
        else:
            self.grad_scaler = None
            logger.info("AMP disabled, using FP32")

        # torch.compile
        if gpu_cfg.get("enable_compile", False) and self.device.type == "cuda":
            try:
                mode = gpu_cfg.get("compile_mode", "reduce-overhead")
                self.model = torch.compile(self.model, mode=mode)
                logger.info(f"torch.compile enabled (mode={mode})")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}, continuing without compilation")

    def train(self):
        """执行完整训练循环。"""
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Gradient accumulation: {self.grad_accum}")
        logger.info(f"  Effective batch size: {self.cfg['train']['batch_size'] * self.grad_accum}")
        logger.info(f"  AMP: {self.use_amp}")

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
                    val_metrics = self.evaluator.evaluate(
                        self.model, self.val_loader, device=self.device
                    )
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
            # 将数据移到GPU
            images = batch["images"].to(self.device, non_blocking=True)
            targets = [
                {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                for t in batch["targets"]
            ]
            class_names = batch["class_names"]

            # 前向传播 (使用CUDA AMP混合精度)
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
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

            # 反向传播 (使用GradScaler处理FP16梯度)
            if self.use_amp and self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            # 累积够了才更新
            if (step + 1) % self.grad_accum == 0:
                if self.use_amp and self.grad_scaler is not None:
                    # 先unscale梯度，再裁剪
                    self.grad_scaler.unscale_(self.optimizer)
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_grad_norm
                        )
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
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

                # GPU显存使用情况
                mem_info = ""
                if self.device.type == "cuda":
                    mem_used = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                    mem_info = f" mem={mem_used:.1f}G/{mem_reserved:.1f}G"

                logger.info(
                    f"  [{epoch+1}][{step+1}/{len(self.train_loader)}] "
                    f"loss={avg['total_loss']:.4f} "
                    f"cls={avg['cls_loss']:.4f} "
                    f"box={avg['box_loss']:.4f} "
                    f"lr={lr:.6f}{mem_info}"
                )

        return {k: v / max(n_steps, 1) for k, v in running_losses.items()}

    def resume(self, checkpoint_path: str):
        """从checkpoint恢复训练。"""
        info = load_checkpoint(self.model, checkpoint_path, self.optimizer)
        self.start_epoch = info.get("epoch", 0)
        self.best_ap = info.get("metrics", {}).get("AP", 0)
        logger.info(f"Resumed from epoch {self.start_epoch}, best AP: {self.best_ap:.4f}")
