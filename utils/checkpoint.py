"""模型存取：保存/加载checkpoint、预训练权重加载与部分匹配。"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    save_path: str,
    **kwargs,
):
    """保存训练checkpoint。

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 评估指标
        save_path: 保存路径
        **kwargs: 其他需要保存的信息
    """
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    state.update(kwargs)

    torch.save(state, save_path)
    logger.info(f"Checkpoint saved: {save_path} (epoch={epoch})")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """加载checkpoint。

    Args:
        model: 模型
        checkpoint_path: checkpoint路径
        optimizer: 优化器 (可选)
        strict: 是否严格匹配所有key

    Returns:
        checkpoint中的额外信息 (epoch, metrics等)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 加载模型权重
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    load_state_dict_flexible(model, state_dict, strict=strict)

    # 加载优化器状态
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Optimizer state loaded.")

    info = {
        "epoch": ckpt.get("epoch", 0),
        "metrics": ckpt.get("metrics", {}),
    }
    logger.info(f"Checkpoint loaded: {checkpoint_path} (epoch={info['epoch']})")
    return info


def load_state_dict_flexible(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = False,
):
    """灵活加载state_dict，支持部分匹配和key映射。

    自动处理:
    - 前缀不匹配 (如 'module.' 前缀)
    - 部分权重加载 (missing keys / unexpected keys)
    - 形状不匹配的权重跳过

    Args:
        model: 目标模型
        state_dict: 权重字典
        strict: 是否严格匹配
    """
    # 去除可能的 'module.' 前缀 (DataParallel)
    clean_state = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        clean_state[name] = v

    model_state = model.state_dict()

    # 匹配检查
    matched_keys = []
    mismatched_keys = []
    missing_keys = []

    for key in model_state:
        if key in clean_state:
            if model_state[key].shape == clean_state[key].shape:
                matched_keys.append(key)
            else:
                mismatched_keys.append(
                    f"{key}: model={model_state[key].shape} vs ckpt={clean_state[key].shape}"
                )
        else:
            missing_keys.append(key)

    unexpected_keys = [k for k in clean_state if k not in model_state]

    # 只加载匹配的权重
    filtered_state = {k: clean_state[k] for k in matched_keys}
    model.load_state_dict(filtered_state, strict=False)

    # 日志报告
    logger.info(
        f"Loaded {len(matched_keys)}/{len(model_state)} parameters. "
        f"Missing: {len(missing_keys)}, Mismatched: {len(mismatched_keys)}, "
        f"Unexpected: {len(unexpected_keys)}"
    )
    if mismatched_keys:
        logger.warning(f"Shape mismatched keys (skipped): {mismatched_keys[:5]}")


def cleanup_checkpoints(save_dir: str, keep_last: int = 3):
    """清理旧的checkpoint，只保留最近的N个。

    Args:
        save_dir: checkpoint目录
        keep_last: 保留最近的N个
    """
    ckpt_files = sorted(
        Path(save_dir).glob("epoch_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )

    if len(ckpt_files) > keep_last:
        for f in ckpt_files[:-keep_last]:
            # 不删除best.pth
            if "best" not in f.name:
                f.unlink()
                logger.info(f"Removed old checkpoint: {f}")
