"""自适应门控跨模态注意力模块 (Adaptive Gated Cross-Modal Attention, AGCMA)。

核心创新模块，提出三个子组件：
1. 深度可分离卷积局部增强：增强视觉特征的局部空间感知
2. 轻量通道亲和力注意力：通过通道维度计算视觉-文本亲和力，替代昂贵的空间cross-attention
3. 自适应门控融合：学习逐通道门控权重，控制文本信息融入程度

相比基线方法的优势：
- vs RepVL-PAN (YOLO-World): 融合粒度更细（通道自适应 vs 统一线性投影）
- vs Cross-Attention (Grounding DINO): 计算量更低（O(C×N) vs O(HW×N×C)）
- 门控机制提供可解释性（可视化哪些通道依赖文本信息）

参考文献:
- YOLO-World (CVPR 2024): RepVL-PAN中的Text-guided CSPLayer
- Grounding DINO (ECCV 2024): Feature Enhancer中的cross-attention
- SE-Net (CVPR 2018): 通道注意力机制
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积。

    = Depthwise Conv (逐通道卷积) + Pointwise Conv (1x1卷积)
    参数量: C×K×K + C×C (vs 标准卷积 C×C×K×K)

    Args:
        channels: 输入/输出通道数
        kernel_size: 卷积核大小
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size,
            padding=padding, groups=channels, bias=False,
        )
        self.pw_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw_conv(self.dw_conv(x))))


class ChannelAffinityAttention(nn.Module):
    """轻量通道亲和力注意力。

    通过通道维度计算视觉-文本亲和力矩阵，避免昂贵的空间维度注意力。

    计算流程:
    1. 视觉特征全局平均池化: [B, C, H, W] -> [B, C]
    2. 文本特征线性投影: [B, N, D] -> [B, N, C]
    3. 亲和力矩阵: [B, C] × [B, C, N] -> [B, C, N]
    4. Softmax归一化后加权聚合文本特征

    复杂度: O(C×N + C×HW)，远低于标准cross-attention的 O(HW×N×C)

    Args:
        vis_channels: 视觉特征通道数 C
        text_dim: 文本特征维度 D
        reduction: 降维比率
    """

    def __init__(self, vis_channels: int, text_dim: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(vis_channels // reduction, 32)

        # 文本投影: D -> C
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, vis_channels),
        )

        # 通道特征压缩 (用于计算亲和力)
        self.channel_compress = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Flatten(1),            # [B, C]
        )

        # 温度参数 (可学习)
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(vis_channels))

    def forward(
        self,
        visual_feat: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算通道亲和力注意力，聚合文本特征。

        Args:
            visual_feat: 视觉特征 [B, C, H, W]
            text_feat: 文本特征 [B, N, D]
            text_mask: 文本padding掩码 [B, N]，1为有效位置

        Returns:
            text_aggregated: 聚合后的文本特征 [B, C, H, W]
        """
        b, c, h, w = visual_feat.shape
        n = text_feat.shape[1]

        # 文本投影: [B, N, D] -> [B, N, C]
        text_proj = self.text_proj(text_feat)  # [B, N, C]

        # 视觉通道特征: [B, C]
        vis_channel = self.channel_compress(visual_feat)  # [B, C]

        # 计算亲和力矩阵: [B, C, N]
        # vis: [B, C], text: [B, N, C]
        # affinity[b, c, n] = vis[b, c] * text[b, n, c]
        affinity = vis_channel.unsqueeze(2) * text_proj.transpose(1, 2)  # [B, C, N]
        affinity = affinity / self.temperature

        # 应用文本掩码
        if text_mask is not None:
            # text_mask: [B, N] -> [B, 1, N]
            mask = text_mask.unsqueeze(1).float()
            affinity = affinity.masked_fill(mask == 0, float("-inf"))

        # Softmax归一化 (在文本维度)
        attn_weights = F.softmax(affinity, dim=-1)  # [B, C, N]

        # 加权聚合文本特征: [B, C, N] × [B, N, C] -> [B, C, C]
        # 但我们需要的是 [B, C, H, W]
        # 方案: 将聚合结果扩展到空间维度
        text_aggregated = torch.bmm(attn_weights, text_proj)  # [B, C, C]

        # 通过1x1卷积压缩回C维，然后扩展到H×W
        # 简化方案: 直接取对角线或使用通道池化
        # 这里使用更优雅的方式: 聚合结果作为通道权重调制视觉特征
        channel_weights = text_aggregated.diagonal(dim1=-2, dim2=-1)  # [B, C]
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 文本调制的视觉特征
        text_modulated = visual_feat * torch.sigmoid(channel_weights)

        return text_modulated


class AdaptiveGate(nn.Module):
    """自适应门控机制。

    学习逐通道的门控权重 α ∈ [0, 1]，控制融合比例：
    output = V + α ⊙ T_agg + (1-α) ⊙ V_local

    门控值接近1: 该通道更依赖文本信息
    门控值接近0: 该通道保留视觉信息

    Args:
        channels: 通道数
        gate_bias: 门控偏置初始值 (sigmoid(-1)≈0.27，初始少量依赖文本)
    """

    def __init__(self, channels: int, gate_bias: float = -1.0):
        super().__init__()

        # 门控网络: 将视觉和文本特征拼接后预测门控权重
        self.gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels * 2, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
        )

        # 初始化偏置
        nn.init.constant_(self.gate_fc[-1].bias, gate_bias)

    def forward(
        self,
        visual_local: torch.Tensor,
        text_aggregated: torch.Tensor,
    ) -> torch.Tensor:
        """计算自适应门控权重并融合。

        Args:
            visual_local: 局部增强的视觉特征 [B, C, H, W]
            text_aggregated: 文本聚合特征 [B, C, H, W]

        Returns:
            fused: 融合后的特征 [B, C, H, W]
        """
        # 拼接后预测门控
        combined = torch.cat([visual_local, text_aggregated], dim=1)  # [B, 2C, H, W]
        alpha = self.gate_fc(combined)  # [B, C]
        alpha = torch.sigmoid(alpha).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 门控融合
        fused = alpha * text_aggregated + (1 - alpha) * visual_local

        # 存储门控权重用于可视化
        self._last_gate_weights = alpha.detach().squeeze(-1).squeeze(-1)

        return fused


class AGCMAModule(nn.Module):
    """自适应门控跨模态注意力模块 (AGCMA)。

    完整的融合模块，包含三个子组件：
    1. DepthwiseSeparableConv: 视觉特征局部增强
    2. ChannelAffinityAttention: 轻量通道亲和力注意力
    3. AdaptiveGate: 自适应门控融合

    融合公式: V_out = V + Gate(DWConv(V), ChanAttn(V, T))

    Args:
        vis_channels: 视觉特征通道数
        text_dim: 文本特征维度
        dw_kernel: 深度可分离卷积核大小
        reduction: 通道注意力降维比
        gate_bias: 门控初始偏置
        use_residual: 是否使用残差连接
        dropout: Dropout比率
    """

    def __init__(
        self,
        vis_channels: int,
        text_dim: int,
        dw_kernel: int = 3,
        reduction: int = 4,
        gate_bias: float = -1.0,
        use_residual: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_residual = use_residual

        # 子组件1: 深度可分离卷积局部增强
        self.local_enhance = DepthwiseSeparableConv(vis_channels, dw_kernel)

        # 子组件2: 通道亲和力注意力
        self.channel_attention = ChannelAffinityAttention(vis_channels, text_dim, reduction)

        # 子组件3: 自适应门控融合
        self.adaptive_gate = AdaptiveGate(vis_channels, gate_bias)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(vis_channels, vis_channels, 1, bias=False),
            nn.BatchNorm2d(vis_channels),
        )

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        visual_feat: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """AGCMA前向传播。

        Args:
            visual_feat: 视觉特征 [B, C, H, W]
            text_feat: 文本特征 [B, N, D]
            text_mask: 文本padding掩码 [B, N]

        Returns:
            fused_feat: 融合后的视觉特征 [B, C, H, W]
        """
        identity = visual_feat

        # Step 1: 深度可分离卷积局部增强
        v_local = self.local_enhance(visual_feat)

        # Step 2: 通道亲和力注意力 (文本聚合到视觉空间)
        t_aggregated = self.channel_attention(visual_feat, text_feat, text_mask)

        # Step 3: 自适应门控融合
        fused = self.adaptive_gate(v_local, t_aggregated)

        # 输出投影 + dropout
        fused = self.dropout(self.output_proj(fused))

        # 残差连接
        if self.use_residual:
            fused = fused + identity

        return fused

    @property
    def last_gate_weights(self) -> Optional[torch.Tensor]:
        """获取最近一次前向传播的门控权重 (用于可视化)。"""
        return getattr(self.adaptive_gate, "_last_gate_weights", None)


# ====================== 消融实验变体 ======================


class AGCMAModule_NoGate(AGCMAModule):
    """消融变体：去掉自适应门控，直接相加。"""

    def forward(self, visual_feat, text_feat, text_mask=None):
        identity = visual_feat
        v_local = self.local_enhance(visual_feat)
        t_aggregated = self.channel_attention(visual_feat, text_feat, text_mask)
        # 直接相加，不经过门控
        fused = v_local + t_aggregated
        fused = self.dropout(self.output_proj(fused))
        if self.use_residual:
            fused = fused + identity
        return fused


class AGCMAModule_NoDW(AGCMAModule):
    """消融变体：去掉深度可分离卷积，直接使用原始视觉特征。"""

    def forward(self, visual_feat, text_feat, text_mask=None):
        identity = visual_feat
        # 不做局部增强
        v_local = visual_feat
        t_aggregated = self.channel_attention(visual_feat, text_feat, text_mask)
        fused = self.adaptive_gate(v_local, t_aggregated)
        fused = self.dropout(self.output_proj(fused))
        if self.use_residual:
            fused = fused + identity
        return fused


class AGCMAModule_SpatialAttn(nn.Module):
    """对比变体：使用标准空间cross-attention替代通道亲和力注意力。

    用于对比实验，展示通道亲和力注意力的效率优势。
    """

    def __init__(self, vis_channels, text_dim, num_heads=8, dropout=0.1, **kwargs):
        super().__init__()

        self.vis_proj = nn.Conv2d(vis_channels, vis_channels, 1)
        self.text_proj = nn.Linear(text_dim, vis_channels)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vis_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(vis_channels, vis_channels, 1, bias=False),
            nn.BatchNorm2d(vis_channels),
        )

    def forward(self, visual_feat, text_feat, text_mask=None):
        b, c, h, w = visual_feat.shape
        identity = visual_feat

        # 视觉特征展平: [B, C, H, W] -> [B, H*W, C]
        v_flat = self.vis_proj(visual_feat).flatten(2).transpose(1, 2)

        # 文本投影: [B, N, D] -> [B, N, C]
        t_proj = self.text_proj(text_feat)

        # Cross-attention: Q=visual, K=V=text
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = text_mask == 0

        attn_out, _ = self.cross_attn(
            v_flat, t_proj, t_proj,
            key_padding_mask=key_padding_mask,
        )

        # 恢复形状: [B, H*W, C] -> [B, C, H, W]
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)

        fused = self.output_proj(attn_out) + identity
        return fused
