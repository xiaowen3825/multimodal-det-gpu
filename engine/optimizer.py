"""优化器配置：AdamW + 余弦退火 + 参数分组。

支持不同模块使用不同学习率：
- Backbone (冻结或极低lr)
- 融合模块 AGCMA (正常lr)
- 检测头 (正常lr)
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

logger = logging.getLogger(__name__)


def build_optimizer(
    model: nn.Module,
    cfg: dict,
) -> torch.optim.Optimizer:
    """构建优化器，支持参数分组。

    分三组：
    1. 归一化层参数 (无weight decay)
    2. 卷积/线性层权重 (有weight decay)
    3. 卷积/线性层偏置 (无weight decay)

    Args:
        model: 模型
        cfg: 训练配置

    Returns:
        优化器
    """
    train_cfg = cfg["train"]
    lr = train_cfg.get("initial_lr", 0.001)
    wd = train_cfg.get("weight_decay", 0.0005)

    # 参数分组
    norm_params = []    # BN/LN: 无weight decay
    weight_params = []  # Conv/Linear weight: 有weight decay
    bias_params = []    # bias: 无weight decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "bn" in name or "norm" in name or "ln" in name:
            norm_params.append(param)
        elif "bias" in name:
            bias_params.append(param)
        else:
            weight_params.append(param)

    param_groups = [
        {"params": norm_params, "weight_decay": 0.0, "name": "norm"},
        {"params": weight_params, "weight_decay": wd, "name": "weight"},
        {"params": bias_params, "weight_decay": 0.0, "name": "bias"},
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

    n_trainable = sum(p.numel() for g in param_groups for p in g["params"])
    logger.info(
        f"Optimizer: AdamW, lr={lr}, wd={wd}, "
        f"trainable params: {n_trainable/1e6:.2f}M "
        f"(norm={len(norm_params)}, weight={len(weight_params)}, bias={len(bias_params)})"
    )

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """构建学习率调度器。

    支持：
    - 余弦退火 (cosine)
    - 线性预热 (warmup)

    Args:
        optimizer: 优化器
        cfg: 训练配置
        steps_per_epoch: 每个epoch的步数

    Returns:
        学习率调度器
    """
    train_cfg = cfg["train"]
    scheduler_type = train_cfg.get("scheduler", "cosine")
    epochs = train_cfg.get("epochs", 50)
    warmup_epochs = train_cfg.get("warmup_epochs", 3)
    initial_lr = train_cfg.get("initial_lr", 0.001)
    final_lr = train_cfg.get("final_lr", 0.0001)

    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if scheduler_type == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                # 线性预热
                return step / max(warmup_steps, 1)
            else:
                # 余弦退火
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return (final_lr / initial_lr) + (1 - final_lr / initial_lr) * (
                    1 + math.cos(math.pi * progress)
                ) / 2

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=final_lr)

    logger.info(
        f"Scheduler: {scheduler_type}, epochs={epochs}, "
        f"warmup={warmup_epochs}, final_lr={final_lr}"
    )

    return scheduler
