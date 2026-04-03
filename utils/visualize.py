"""可视化工具：检测结果绘制、特征图可视化、门控权重热力图。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# 使用非交互式后端
matplotlib.use("Agg")

# COCO类别颜色 (随机但固定)
np.random.seed(42)
COLORS = np.random.randint(50, 255, size=(200, 3), dtype=np.uint8)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: np.ndarray,
    score_thr: float = 0.3,
    line_width: int = 2,
) -> np.ndarray:
    """在图像上绘制检测结果。

    Args:
        image: BGR图像 [H, W, 3]
        boxes: 检测框 [N, 4] (x1, y1, x2, y2)
        labels: 类别名列表 [N]
        scores: 置信度 [N]
        score_thr: 置信度阈值
        line_width: 边框线宽

    Returns:
        绘制后的图像
    """
    vis_img = image.copy()

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score < score_thr:
            continue

        x1, y1, x2, y2 = box.astype(int)
        color = tuple(int(c) for c in COLORS[hash(label) % len(COLORS)])

        # 画框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, line_width)

        # 标签文字
        text = f"{label}: {score:.2f}"
        font_scale = 0.6
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 文字背景
        cv2.rectangle(vis_img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            vis_img, text, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )

    return vis_img


def visualize_feature_map(
    feature: torch.Tensor,
    save_path: str,
    title: str = "Feature Map",
    top_k: int = 16,
):
    """可视化特征图的前K个通道。

    Args:
        feature: 特征图 [C, H, W] 或 [1, C, H, W]
        save_path: 保存路径
        title: 图标题
        top_k: 显示前K个通道（按L2范数排序）
    """
    if feature.dim() == 4:
        feature = feature[0]

    feature = feature.detach().cpu().float()
    c, h, w = feature.shape

    # 按通道L2范数排序，取前K个
    norms = feature.flatten(1).norm(dim=1)
    topk_indices = norms.topk(min(top_k, c)).indices

    ncols = 4
    nrows = (min(top_k, c) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))

    for idx, ax in enumerate(axes.flat if nrows > 1 else [axes] if nrows == 1 and ncols == 1 else axes):
        if idx < len(topk_indices):
            ch_idx = topk_indices[idx].item()
            im = ax.imshow(feature[ch_idx].numpy(), cmap="viridis")
            ax.set_title(f"Ch {ch_idx} (L2={norms[ch_idx]:.2f})", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_gate_weights(
    gate_weights: torch.Tensor,
    text_labels: List[str] | None = None,
    save_path: str = "gate_weights.png",
    title: str = "AGCMA Gate Weights",
):
    """可视化AGCMA模块的门控权重热力图。

    Args:
        gate_weights: 门控权重 [C] 或 [num_scales, C]
        text_labels: 文本标签列表
        save_path: 保存路径
        title: 图标题
    """
    gate_weights = gate_weights.detach().cpu().float()

    if gate_weights.dim() == 1:
        gate_weights = gate_weights.unsqueeze(0)

    fig, ax = plt.subplots(figsize=(12, max(3, gate_weights.shape[0])))

    im = ax.imshow(gate_weights.numpy(), aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)

    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Scale" if gate_weights.shape[0] > 1 else "")

    if gate_weights.shape[0] <= 5:
        ax.set_yticks(range(gate_weights.shape[0]))
        scale_names = ["P3 (1/8)", "P4 (1/16)", "P5 (1/32)"]
        ax.set_yticklabels(scale_names[: gate_weights.shape[0]])

    plt.colorbar(im, ax=ax, label="Gate Value (α)")
    ax.set_title(title)

    plt.tight_layout()
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_attention_map(
    attention_map: torch.Tensor,
    image: np.ndarray,
    text_labels: List[str],
    save_path: str = "attention.png",
):
    """可视化文本-视觉注意力图，叠加到原图上。

    Args:
        attention_map: 注意力图 [N, H, W] (N=文本词数)
        image: 原始BGR图像 [H_img, W_img, 3]
        text_labels: 文本标签
        save_path: 保存路径
    """
    attention_map = attention_map.detach().cpu().float()
    n_texts = min(attention_map.shape[0], len(text_labels), 8)

    fig, axes = plt.subplots(1, n_texts + 1, figsize=(4 * (n_texts + 1), 4))

    # 原图
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image
    axes[0].imshow(rgb)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    h_img, w_img = rgb.shape[:2]

    for i in range(n_texts):
        attn = attention_map[i].numpy()
        # 上采样到原图尺寸
        attn_resized = cv2.resize(attn, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)

        axes[i + 1].imshow(rgb)
        axes[i + 1].imshow(attn_resized, alpha=0.6, cmap="jet")
        axes[i + 1].set_title(text_labels[i], fontsize=10)
        axes[i + 1].axis("off")

    plt.tight_layout()
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    log_data: dict[str, list],
    save_path: str = "training_curves.png",
):
    """绘制训练曲线。

    Args:
        log_data: {'loss': [...], 'lr': [...], 'AP': [...], ...}
        save_path: 保存路径
    """
    n_metrics = len(log_data)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, log_data.items()):
        ax.plot(values, linewidth=1.5)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
