"""指标计算：AP/AR计算封装、FLOPs和参数量统计。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计模型参数量。

    Args:
        model: 模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数总数
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, input_shape: tuple = (1, 3, 640, 640), text_input: Any = None) -> float:
    """估算模型FLOPs。

    使用简化的计算方式，逐层统计卷积和线性层的FLOPs。

    Args:
        model: 模型
        input_shape: 输入张量形状
        text_input: 文本输入 (开放词汇模型需要)

    Returns:
        FLOPs (浮点运算次数)
    """
    total_flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].shape[0]
        out_channels, in_channels_per_group, kh, kw = module.weight.shape
        groups = module.groups
        out_h, out_w = output.shape[2], output.shape[3]

        flops = batch_size * out_channels * out_h * out_w * in_channels_per_group * kh * kw * 2
        total_flops += flops

    def linear_hook(module, input, output):
        nonlocal total_flops
        batch_size = input[0].shape[0]
        flops = batch_size * module.in_features * module.out_features * 2
        total_flops += flops

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        if text_input is not None:
            model(dummy_input, text_input)
        else:
            model(dummy_input)

    for h in hooks:
        h.remove()

    return total_flops


def benchmark_speed(
    model: nn.Module,
    input_shape: tuple = (1, 3, 640, 640),
    text_input: Any = None,
    warmup_iters: int = 50,
    test_iters: int = 200,
) -> dict[str, float]:
    """测试模型推理速度。

    Args:
        model: 模型
        input_shape: 输入张量形状
        text_input: 文本输入
        warmup_iters: 预热迭代次数
        test_iters: 测试迭代次数

    Returns:
        dict: {'latency_ms': float, 'fps': float, 'std_ms': float}
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            if text_input is not None:
                model(dummy_input, text_input)
            else:
                model(dummy_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(test_iters):
            start = time.perf_counter()
            if text_input is not None:
                model(dummy_input, text_input)
            else:
                model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "latency_ms": float(np.mean(latencies)),
        "fps": float(1000.0 / np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
    }


def format_metrics_table(metrics: dict[str, float], title: str = "Evaluation Results") -> str:
    """格式化指标为表格字符串。

    Args:
        metrics: 指标字典
        title: 标题

    Returns:
        格式化的表格字符串
    """
    lines = [f"\n{'='*50}", f"  {title}", f"{'='*50}"]

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:.<35s} {value:.4f}")
        else:
            lines.append(f"  {key:.<35s} {value}")

    lines.append(f"{'='*50}\n")
    return "\n".join(lines)
