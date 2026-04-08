"""解耦检测头 (Decoupled Detection Head)。

YOLOv8风格的解耦检测头，分离分类和回归分支：
- 分类分支: 通过视觉-文本相似度计算开放词汇分类
- 回归分支: 基于DFL (Distribution Focal Loss) 的边框回归

参考文献:
- YOLOv8: 解耦头设计
- YOLO-World (CVPR 2024): 文本相似度分类
- DFL (Generalized Focal Loss V2): 分布式边框回归
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.yolov8_backbone import Conv, DWConv


class DFL(nn.Module):
    """Distribution Focal Loss 层。

    将离散分布转换为连续边框坐标。

    Args:
        c1: 分布的离散bin数量 (reg_max + 1)
    """

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将分布logits转换为边框坐标。

        Args:
            x: [B, 4*reg_max, H, W] 或 [B, A, 4*reg_max]

        Returns:
            [B, 4, H, W] 或 [B, A, 4]
        """
        b, _, a = x.shape  # batch, channels, anchors
        x = x.view(b, 4, self.c1, a).transpose(2, 3).softmax(3)
        return self.conv(x.reshape(b, 4, a, self.c1).transpose(2, 3)).reshape(b, 4, a)


class DecoupledHead(nn.Module):
    """解耦检测头。

    分离分类和回归分支，分类使用文本相似度计算。

    Args:
        in_channels: 各尺度输入通道数列表，如 [128, 256, 512]
        num_classes: 类别数，-1表示开放词汇模式
        reg_max: DFL回归的最大值，默认16
        strides: 各尺度步长，如 [8, 16, 32]
    """

    def __init__(
        self,
        in_channels: List[int] = [128, 256, 512],
        num_classes: int = -1,
        reg_max: int = 16,
        strides: List[int] = [8, 16, 32],
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.nl = len(in_channels)  # 检测层数
        self.no_reg = 4 * reg_max   # 回归输出维度

        # 分类分支 (输出视觉embedding，用于和文本计算相似度)
        self.cls_convs = nn.ModuleList()
        self.cls_proj = nn.ModuleList()  # 投影到文本空间

        # 回归分支
        self.reg_convs = nn.ModuleList()
        self.reg_pred = nn.ModuleList()

        # 目标性分支 (objectness)
        self.obj_pred = nn.ModuleList()

        for i, ch in enumerate(in_channels):
            # 分类分支: 2层conv + 投影层
            self.cls_convs.append(
                nn.Sequential(
                    Conv(ch, ch, 3, 1),
                    Conv(ch, ch, 3, 1),
                )
            )
            # 投影到text embedding空间 (将在forward时使用)
            self.cls_proj.append(nn.Conv2d(ch, ch, 1, bias=False))

            # 回归分支: 2层conv + DFL预测
            self.reg_convs.append(
                nn.Sequential(
                    Conv(ch, ch, 3, 1),
                    Conv(ch, ch, 3, 1),
                )
            )
            self.reg_pred.append(nn.Conv2d(ch, self.no_reg, 1))

            # objectness预测
            self.obj_pred.append(nn.Conv2d(ch, 1, 1))

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        # 用于生成anchor points
        self._anchor_points = {}
        self._stride_tensors = {}

        self._init_weights()

    def forward(
        self,
        features: List[torch.Tensor],
        text_embeds: torch.Tensor | None = None,
    ) -> dict:
        """前向传播。

        Args:
            features: 多尺度特征列表 [(B, C_i, H_i, W_i), ...]
            text_embeds: 文本嵌入 [B, N, D] 或 [N, D] (N为类别数)

        Returns:
            dict with keys:
                - 'cls_scores': 分类分数 [B, sum(H_i*W_i), N]
                - 'bbox_preds': 边框预测 [B, sum(H_i*W_i), 4]
                - 'objectness': 目标性分数 [B, sum(H_i*W_i), 1]
                - 'anchor_points': anchor坐标 [sum(H_i*W_i), 2]
                - 'stride_tensor': 步长 [sum(H_i*W_i), 1]
        """
        all_cls_embeds = []
        all_bbox_preds = []
        all_obj_scores = []
        anchor_points_list = []
        stride_list = []

        for i, feat in enumerate(features):
            b, _, h, w = feat.shape

            # 分类分支 -> 视觉嵌入
            cls_feat = self.cls_convs[i](feat)
            cls_embed = self.cls_proj[i](cls_feat)  # [B, C, H, W]

            # 回归分支
            reg_feat = self.reg_convs[i](feat)
            bbox_pred = self.reg_pred[i](reg_feat)  # [B, 4*reg_max, H, W]

            # 目标性分支
            obj_score = self.obj_pred[i](feat)  # [B, 1, H, W]

            # Reshape: [B, C, H, W] -> [B, H*W, C]
            cls_embed = cls_embed.flatten(2).transpose(1, 2)   # [B, H*W, C]
            bbox_pred = bbox_pred.flatten(2).transpose(1, 2)   # [B, H*W, 4*reg_max]
            obj_score = obj_score.flatten(2).transpose(1, 2)   # [B, H*W, 1]

            all_cls_embeds.append(cls_embed)
            all_bbox_preds.append(bbox_pred)
            all_obj_scores.append(obj_score)

            # 生成anchor points
            ap, st = self._make_anchors(h, w, self.strides[i], feat.device)
            anchor_points_list.append(ap)
            stride_list.append(st)

        # 拼接所有尺度
        cls_embeds = torch.cat(all_cls_embeds, dim=1)   # [B, total_anchors, C]
        bbox_preds = torch.cat(all_bbox_preds, dim=1)   # [B, total_anchors, 4*reg_max]
        obj_scores = torch.cat(all_obj_scores, dim=1)   # [B, total_anchors, 1]
        anchor_points = torch.cat(anchor_points_list, dim=0)  # [total_anchors, 2]
        stride_tensor = torch.cat(stride_list, dim=0)          # [total_anchors, 1]

        # 计算文本相似度分类分数
        cls_scores = None
        if text_embeds is not None:
            if text_embeds.dim() == 2:
                text_embeds = text_embeds.unsqueeze(0).expand(b, -1, -1)
            # 归一化后计算余弦相似度
            cls_norm = F.normalize(cls_embeds, dim=-1)
            txt_norm = F.normalize(text_embeds, dim=-1)
            cls_scores = torch.bmm(cls_norm, txt_norm.transpose(1, 2))  # [B, total_anchors, N]

        # DFL解码: 将分布logits转换为距离值 (l,t,r,b)
        # 输出范围 [0, reg_max), 单位是stride (即网格单元)
        if isinstance(self.dfl, DFL):
            bbox_dist = self.dfl(bbox_preds.transpose(1, 2)).transpose(1, 2)  # [B, A, 4]
        else:
            bbox_dist = bbox_preds

        # dist2bbox: 将距离转换为绝对像素坐标 (x1,y1,x2,y2)
        anchor_pixel = anchor_points * stride_tensor  # [A, 2] 像素坐标
        lt = bbox_dist[..., :2]  # [B, A, 2] left, top
        rb = bbox_dist[..., 2:]  # [B, A, 2] right, bottom
        x1y1 = anchor_pixel.unsqueeze(0) - lt * stride_tensor.unsqueeze(0)
        x2y2 = anchor_pixel.unsqueeze(0) + rb * stride_tensor.unsqueeze(0)
        bbox_xyxy = torch.cat([x1y1, x2y2], dim=-1)  # [B, A, 4]

        return {
            "cls_scores": cls_scores,
            "bbox_preds": bbox_xyxy,
            "bbox_preds_dist": bbox_dist,
            "bbox_preds_raw": bbox_preds,
            "objectness": obj_scores,
            "anchor_points": anchor_points,
            "stride_tensor": stride_tensor,
        }

    def _make_anchors(self, h: int, w: int, stride: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成anchor点网格。

        Args:
            h, w: 特征图尺寸
            stride: 该尺度步长
            device: 设备

        Returns:
            anchor_points: [H*W, 2]  中心坐标 (x, y)
            stride_tensor: [H*W, 1]  步长
        """
        key = (h, w, stride)
        if key not in self._anchor_points or self._anchor_points[key].device != device:
            sy = torch.arange(h, device=device, dtype=torch.float32) + 0.5
            sx = torch.arange(w, device=device, dtype=torch.float32) + 0.5
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            stride_tensor = torch.full((h * w, 1), stride, device=device, dtype=torch.float32)
            self._anchor_points[key] = anchor_points
            self._stride_tensors[key] = stride_tensor

        return self._anchor_points[key], self._stride_tensors[key]

    def _init_weights(self):
        """初始化权重，偏置初始化遵循prior probability。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # objectness偏置初始化: sigmoid(bias) ≈ 0.01
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for obj in self.obj_pred:
            nn.init.constant_(obj.bias, bias_value)
