"""RepVL-PAN: YOLO-World的原始融合网络（作为对比基线）。

复现YOLO-World的RepVL-PAN核心思想：
- Text-guided CSPLayer: 使用文本嵌入通过线性投影指导视觉特征
- 推理时可重参数化: 将文本嵌入融入卷积权重

简化版本，保留核心融合机制用于公平对比。

参考文献:
- YOLO-World (CVPR 2024): Re-parameterizable Vision-Language PAN
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.yolov8_backbone import Conv, C2f


class TextGuidedCSPLayer(nn.Module):
    """文本引导的CSP层。

    YOLO-World RepVL-PAN的核心组件：
    通过文本嵌入的线性投影来调制视觉特征。

    融合方式: V_out = V * sigmoid(Linear(T_pooled))

    Args:
        vis_channels: 视觉通道数
        text_dim: 文本维度
    """

    def __init__(self, vis_channels: int, text_dim: int):
        super().__init__()

        # 文本到视觉的投影
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, vis_channels),
            nn.Sigmoid(),
        )

        # 图像到文本的投影 (I2T)
        self.vis_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(vis_channels, text_dim),
        )

    def forward(
        self,
        visual_feat: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """文本引导的视觉特征调制。

        Args:
            visual_feat: [B, C, H, W]
            text_feat: [B, N, D]
            text_mask: [B, N]

        Returns:
            modulated: [B, C, H, W]
        """
        # 文本池化: [B, N, D] -> [B, D]
        if text_mask is not None:
            mask = text_mask.unsqueeze(-1).float()
            text_pooled = (text_feat * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            text_pooled = text_feat.mean(dim=1)

        # T2I: 文本调制视觉
        text_weights = self.text_proj(text_pooled)  # [B, C]
        text_weights = text_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        modulated = visual_feat * text_weights

        return modulated


class RepVL_PAN(nn.Module):
    """RepVL-PAN: YOLO-World的视觉-语言PAN (简化复现)。

    作为AGCMA-PAN的对比基线。

    Args:
        in_channels: backbone输出通道列表
        text_dim: 文本嵌入维度
        out_channels: 输出通道列表
    """

    def __init__(
        self,
        in_channels: List[int] = [128, 256, 512],
        text_dim: int = 512,
        out_channels: List[int] = [128, 256, 512],
        **kwargs,
    ):
        super().__init__()

        # Text-guided层
        self.tg_c5 = TextGuidedCSPLayer(in_channels[2], text_dim)
        self.tg_c4 = TextGuidedCSPLayer(in_channels[1], text_dim)
        self.tg_c3 = TextGuidedCSPLayer(in_channels[0], text_dim)

        # Top-Down
        self.reduce_c5 = Conv(in_channels[2], in_channels[1], 1, 1)
        self.td_c2f_c4 = C2f(in_channels[1] * 2, in_channels[1], n=1, shortcut=False)
        self.reduce_c4 = Conv(in_channels[1], in_channels[0], 1, 1)
        self.td_c2f_c3 = C2f(in_channels[0] * 2, in_channels[0], n=1, shortcut=False)

        # Bottom-Up
        self.bu_down_c3 = Conv(in_channels[0], in_channels[0], 3, 2)
        self.bu_c2f_c4 = C2f(in_channels[0] + in_channels[1], in_channels[1], n=1, shortcut=False)
        self.bu_down_c4 = Conv(in_channels[1], in_channels[1], 3, 2)
        self.bu_c2f_c5 = C2f(in_channels[1] + in_channels[2], in_channels[2], n=1, shortcut=False)

        # 输出投影
        self.out_proj = nn.ModuleList([
            Conv(ch_in, ch_out, 1, 1) if ch_in != ch_out else nn.Identity()
            for ch_in, ch_out in zip(in_channels, out_channels)
        ])

    def forward(
        self,
        features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_feat: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """RepVL-PAN前向传播。

        Args:
            features: (C3, C4, C5)
            text_feat: [B, N, D]
            text_mask: [B, N]

        Returns:
            [P3, P4, P5]
        """
        c3, c4, c5 = features

        # Top-Down
        p5_td = self.tg_c5(c5, text_feat, text_mask)
        p5_reduced = self.reduce_c5(p5_td)
        p5_up = F.interpolate(p5_reduced, size=c4.shape[2:], mode="nearest")

        p4_td = self.td_c2f_c4(torch.cat([c4, p5_up], dim=1))
        p4_td = self.tg_c4(p4_td, text_feat, text_mask)
        p4_reduced = self.reduce_c4(p4_td)
        p4_up = F.interpolate(p4_reduced, size=c3.shape[2:], mode="nearest")

        p3 = self.td_c2f_c3(torch.cat([c3, p4_up], dim=1))
        p3 = self.tg_c3(p3, text_feat, text_mask)

        # Bottom-Up
        p3_down = self.bu_down_c3(p3)
        p4 = self.bu_c2f_c4(torch.cat([p3_down, p4_td], dim=1))

        p4_down = self.bu_down_c4(p4)
        p5 = self.bu_c2f_c5(torch.cat([p4_down, p5_td], dim=1))

        # 输出
        outputs = [p3, p4, p5]
        outputs = [proj(feat) for proj, feat in zip(self.out_proj, outputs)]

        return outputs
