"""AGCMA-PAN: 将AGCMA模块集成到特征金字塔网络。

替代YOLO-World的RepVL-PAN，在FPN的每个尺度上插入AGCMA模块，
实现多尺度视觉-文本融合。

架构:
    C5 → AGCMA → 上采样 → Concat(C4) → Conv → AGCMA → 上采样 → Concat(C3) → Conv → AGCMA → P3
                                                   ↓                                    ↓
                                              P4 (下采样拼接P3后)                   P3 (最终输出)
                                                   ↓
                                              P5 (下采样拼接P4后)

参考文献:
- YOLO-World (CVPR 2024): RepVL-PAN
- PANet (CVPR 2018): Path Aggregation Network
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.yolov8_backbone import Conv, C2f
from .agcma_module import AGCMAModule


class AGCMA_PAN(nn.Module):
    """AGCMA-PAN: 基于AGCMA的特征金字塔网络。

    在标准PAN的每个尺度上集成AGCMA模块，实现多尺度跨模态融合。

    Args:
        in_channels: backbone输出通道数列表 [C3, C4, C5]
        text_dim: 文本嵌入维度
        out_channels: 输出通道数列表
        agcma: AGCMA模块的配置字典
    """

    def __init__(
        self,
        in_channels: List[int] = [128, 256, 512],
        text_dim: int = 512,
        out_channels: List[int] = [128, 256, 512],
        agcma: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        agcma_cfg = agcma or {}

        self.in_channels = in_channels
        self.out_channels = out_channels

        # ===================== Top-Down Path (FPN) =====================

        # C5 -> P5_td (通过AGCMA融合文本)
        self.agcma_c5 = AGCMAModule(
            vis_channels=in_channels[2],
            text_dim=text_dim,
            **agcma_cfg,
        )
        self.reduce_c5 = Conv(in_channels[2], in_channels[1], 1, 1)

        # C4 + upsampled(P5_td) -> P4_td
        self.agcma_c4 = AGCMAModule(
            vis_channels=in_channels[1],
            text_dim=text_dim,
            **agcma_cfg,
        )
        self.td_c2f_c4 = C2f(
            in_channels[1] * 2,  # concat后通道加倍
            in_channels[1],
            n=1,
            shortcut=False,
        )
        self.reduce_c4 = Conv(in_channels[1], in_channels[0], 1, 1)

        # C3 + upsampled(P4_td) -> P3
        self.agcma_c3 = AGCMAModule(
            vis_channels=in_channels[0],
            text_dim=text_dim,
            **agcma_cfg,
        )
        self.td_c2f_c3 = C2f(
            in_channels[0] * 2,
            in_channels[0],
            n=1,
            shortcut=False,
        )

        # ===================== Bottom-Up Path (PAN) =====================

        # P3 -> downsample -> concat(P4_td) -> P4
        self.bu_down_c3 = Conv(in_channels[0], in_channels[0], 3, 2)
        self.bu_c2f_c4 = C2f(
            in_channels[0] + in_channels[1],
            in_channels[1],
            n=1,
            shortcut=False,
        )

        # P4 -> downsample -> concat(P5_td) -> P5
        self.bu_down_c4 = Conv(in_channels[1], in_channels[1], 3, 2)
        self.bu_c2f_c5 = C2f(
            in_channels[1] + in_channels[2],
            in_channels[2],
            n=1,
            shortcut=False,
        )

        # ===================== 输出投影 =====================
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
        """AGCMA-PAN前向传播。

        Args:
            features: backbone输出 (C3, C4, C5)
            text_feat: 文本嵌入 [B, N, D]
            text_mask: 文本掩码 [B, N]

        Returns:
            [P3, P4, P5]: 多尺度融合特征
        """
        c3, c4, c5 = features

        # ========== Top-Down Path ==========

        # C5 + text -> P5_td
        p5_td = self.agcma_c5(c5, text_feat, text_mask)
        p5_reduced = self.reduce_c5(p5_td)  # 降维到C4通道数

        # 上采样并与C4拼接
        p5_up = F.interpolate(p5_reduced, size=c4.shape[2:], mode="nearest")
        p4_td = self.td_c2f_c4(torch.cat([c4, p5_up], dim=1))
        p4_td = self.agcma_c4(p4_td, text_feat, text_mask)
        p4_reduced = self.reduce_c4(p4_td)

        # 上采样并与C3拼接
        p4_up = F.interpolate(p4_reduced, size=c3.shape[2:], mode="nearest")
        p3 = self.td_c2f_c3(torch.cat([c3, p4_up], dim=1))
        p3 = self.agcma_c3(p3, text_feat, text_mask)

        # ========== Bottom-Up Path ==========

        # P3下采样 + P4_td -> P4
        p3_down = self.bu_down_c3(p3)
        p4 = self.bu_c2f_c4(torch.cat([p3_down, p4_td], dim=1))

        # P4下采样 + P5_td -> P5
        p4_down = self.bu_down_c4(p4)
        p5 = self.bu_c2f_c5(torch.cat([p4_down, p5_td], dim=1))

        # 输出投影
        outputs = [p3, p4, p5]
        outputs = [proj(feat) for proj, feat in zip(self.out_proj, outputs)]

        return outputs

    def get_gate_weights(self) -> dict[str, torch.Tensor]:
        """获取所有AGCMA模块的门控权重 (用于可视化)。

        Returns:
            {'scale_8': Tensor, 'scale_16': Tensor, 'scale_32': Tensor}
        """
        gate_weights = {}
        for name, module in [
            ("scale_8", self.agcma_c3),
            ("scale_16", self.agcma_c4),
            ("scale_32", self.agcma_c5),
        ]:
            weights = module.last_gate_weights
            if weights is not None:
                gate_weights[name] = weights
        return gate_weights
