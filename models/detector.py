"""多模态开放词汇目标检测器。

组装 backbone + text_encoder + neck + head 为完整检测器，
管理前向传播、损失计算和参数冻结策略。

参考文献:
- YOLO-World (CVPR 2024): 整体检测框架
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiModalDetector(nn.Module):
    """多模态开放词汇目标检测器。

    组装各组件并管理前向传播流程：
    1. 视觉backbone提取多尺度特征
    2. 文本编码器编码类别文本
    3. Neck(AGCMA-PAN或RepVL-PAN)进行跨模态融合
    4. 检测头输出分类+回归结果

    Args:
        backbone: 视觉backbone (YOLOv8-S)
        text_encoder: 文本编码器 (CLIP或轻量蒸馏版)
        neck: 融合网络 (AGCMA-PAN或RepVL-PAN)
        head: 检测头
        freeze_backbone_stages: 冻结backbone前N个stage
        freeze_text_encoder: 是否冻结文本编码器
    """

    def __init__(
        self,
        backbone: nn.Module,
        text_encoder: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        freeze_backbone_stages: int = 0,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.text_encoder = text_encoder
        self.neck = neck
        self.head = head

        # 冻结策略
        if freeze_backbone_stages > 0:
            self._freeze_backbone(freeze_backbone_stages)
        if freeze_text_encoder:
            self._freeze_text_encoder()

        # 缓存文本嵌入 (推理时同一批类别不需要重复编码)
        self._cached_text_embeds = None
        self._cached_text_mask = None
        self._cached_text_key = None

        self._log_model_info()

    def forward(
        self,
        images: torch.Tensor,
        text_prompts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """前向传播。

        Args:
            images: 输入图像 [B, 3, H, W]
            text_prompts: 文本提示列表 ["person", "car", ...]
            text_embeds: 预计算的文本嵌入 [B, N, D] (与text_prompts二选一)
            text_mask: 文本掩码 [B, N]
            targets: 训练目标列表（仅训练时使用）

        Returns:
            训练模式: {'total_loss': Tensor, 'cls_loss': Tensor, 'box_loss': Tensor, ...}
            推理模式: {'cls_scores': Tensor, 'bbox_preds': Tensor, 'objectness': Tensor, ...}
        """
        # 1. 视觉特征提取
        features = self.backbone(images)  # (C3, C4, C5)
        batch_size = images.shape[0]

        # 2. 文本编码
        if text_embeds is None and text_prompts is not None:
            text_embeds, text_mask = self._encode_texts(text_prompts)
        elif text_embeds is None:
            raise ValueError("Must provide either text_prompts or text_embeds")

        # 扩展text_embeds到batch维度: [1, N, D] -> [B, N, D]
        if text_embeds.dim() == 3 and text_embeds.shape[0] == 1 and batch_size > 1:
            text_embeds = text_embeds.expand(batch_size, -1, -1)
            if text_mask is not None:
                text_mask = text_mask.expand(batch_size, -1)

        # 3. 跨模态融合
        fused_features = self.neck(features, text_embeds, text_mask)  # [P3, P4, P5]

        # 4. 检测头
        cls_text_embeds = text_embeds  # [B, N, D]
        head_outputs = self.head(fused_features, cls_text_embeds)

        # 5. 训练模式：计算损失
        if self.training and targets is not None:
            losses = self._compute_loss(head_outputs, targets)
            return losses

        return head_outputs

    def _encode_texts(
        self,
        text_prompts: List[str],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """编码文本提示。

        使用缓存避免重复编码相同文本。
        返回形状:
            text_embeds: [1, N, D] (N=类别数, D=embed_dim)
            text_mask: [1, N] 或 None

        Args:
            text_prompts: 文本字符串列表 (类别名)

        Returns:
            (text_embeds, text_mask)
        """
        cache_key = tuple(text_prompts)
        if self._cached_text_key == cache_key:
            return self._cached_text_embeds, self._cached_text_mask

        with torch.no_grad() if not self.text_encoder.training else torch.enable_grad():
            if hasattr(self.text_encoder, 'encode_class_names'):
                cls_embeds = self.text_encoder.encode_class_names(text_prompts)
                text_embeds = cls_embeds.unsqueeze(0)  # [N, D] -> [1, N, D]
                text_mask = None
            else:
                text_embeds, text_mask = self.text_encoder.get_all_token_embeddings(
                    texts=text_prompts
                )
                text_embeds = text_embeds.unsqueeze(0) if text_embeds.dim() == 2 else text_embeds
                if text_mask is not None:
                    text_mask = text_mask.unsqueeze(0) if text_mask.dim() == 1 else text_mask

        self._cached_text_embeds = text_embeds
        self._cached_text_mask = text_mask
        self._cached_text_key = cache_key

        return text_embeds, text_mask

    def _compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """计算检测损失。

        Args:
            predictions: 检测头输出
            targets: 标注列表，每个元素包含 {'boxes': [M, 4], 'labels': [M]}

        Returns:
            损失字典
        """
        cls_scores = predictions["cls_scores"]      # [B, A, N]
        bbox_preds = predictions["bbox_preds"]      # [B, A, 4] 已解码的绝对坐标
        objectness = predictions["objectness"]      # [B, A, 1]
        anchor_points = predictions["anchor_points"]  # [A, 2]
        stride_tensor = predictions["stride_tensor"]  # [A, 1]

        device = cls_scores.device
        batch_size = cls_scores.shape[0]
        num_anchors = cls_scores.shape[1]

        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)

        num_pos = 0

        for b in range(batch_size):
            if targets[b]["boxes"].shape[0] == 0:
                total_obj_loss += F.binary_cross_entropy_with_logits(
                    objectness[b].squeeze(-1),
                    torch.zeros(num_anchors, device=device),
                )
                neg_cls_targets = torch.zeros_like(cls_scores[b])
                total_cls_loss += F.binary_cross_entropy_with_logits(
                    cls_scores[b], neg_cls_targets, reduction="mean"
                )
                continue

            gt_boxes = targets[b]["boxes"].to(device)     # [M, 4]
            gt_labels = targets[b]["labels"].to(device)   # [M]

            matched_indices, matched_gt = self._simple_match(
                anchor_points, stride_tensor, gt_boxes
            )

            if matched_indices.shape[0] > 0:
                num_pos += matched_indices.shape[0]

                pos_cls_scores = cls_scores[b, matched_indices]  # [P, N]
                cls_targets = torch.zeros_like(pos_cls_scores)
                cls_targets[torch.arange(len(matched_gt)), gt_labels[matched_gt]] = 1.0
                total_cls_loss += F.binary_cross_entropy_with_logits(
                    pos_cls_scores, cls_targets, reduction="sum"
                )

                # 负样本分类损失: 对未匹配的anchor也施加负类约束
                neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)
                neg_mask[matched_indices] = False
                neg_cls_scores = cls_scores[b, neg_mask]  # [N_neg, N]
                neg_cls_targets = torch.zeros_like(neg_cls_scores)
                if neg_cls_scores.shape[0] > 0:
                    total_cls_loss += F.binary_cross_entropy_with_logits(
                        neg_cls_scores, neg_cls_targets, reduction="mean"
                    ) * num_anchors * 0.25

                # 边框损失: bbox_preds已经是绝对坐标，直接计算CIoU
                pred_boxes = bbox_preds[b, matched_indices]  # [P, 4]
                pos_gt_boxes = gt_boxes[matched_gt]          # [P, 4]
                total_box_loss += self._ciou_loss(pred_boxes, pos_gt_boxes).sum()

            obj_targets = torch.zeros(num_anchors, device=device)
            if matched_indices.shape[0] > 0:
                obj_targets[matched_indices] = 1.0
            total_obj_loss += F.binary_cross_entropy_with_logits(
                objectness[b].squeeze(-1), obj_targets, reduction="mean"
            )

        num_pos = max(num_pos, 1)
        cls_loss = total_cls_loss / num_pos
        box_loss = total_box_loss / num_pos
        obj_loss = total_obj_loss / batch_size
        total_loss = cls_loss + 7.5 * box_loss + obj_loss

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "box_loss": box_loss,
            "obj_loss": obj_loss,
        }

    @staticmethod
    def _simple_match(
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_boxes: torch.Tensor,
        topk: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """简化版正样本匹配（基于距离）。

        Args:
            anchor_points: [A, 2]
            stride_tensor: [A, 1]
            gt_boxes: [M, 4]
            topk: 每个GT选取的最近anchor数

        Returns:
            (matched_anchor_indices, matched_gt_indices)
        """
        # GT中心点
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:4]) / 2  # [M, 2]

        # 计算anchor到GT中心的距离
        # anchor_points已经是在stride坐标系下的，需要转换到像素坐标
        anchor_pixel = anchor_points * stride_tensor  # [A, 2]

        # [A, 1, 2] - [1, M, 2] -> [A, M]
        distances = torch.cdist(anchor_pixel, gt_centers)  # [A, M]

        # 每个GT选取topk个最近的anchor
        k = min(topk, distances.shape[0])
        topk_indices = distances.topk(k, dim=0, largest=False).indices  # [k, M]

        # 展开
        m = gt_boxes.shape[0]
        matched_anchors = topk_indices.t().reshape(-1)  # [M*k]
        matched_gts = torch.arange(m, device=gt_boxes.device).unsqueeze(1).expand(-1, k).reshape(-1)  # [M*k]

        # 去重 (一个anchor只分配给一个GT)
        unique_anchors, inverse = matched_anchors.unique(return_inverse=True)
        # 保留距离最小的分配
        best_gt = torch.zeros(unique_anchors.shape[0], dtype=torch.long, device=gt_boxes.device)
        for i, anchor_idx in enumerate(unique_anchors):
            mask = matched_anchors == anchor_idx
            gt_candidates = matched_gts[mask]
            dists = distances[anchor_idx, gt_candidates]
            best_gt[i] = gt_candidates[dists.argmin()]

        return unique_anchors, best_gt

    @staticmethod
    def _dist2bbox(
        distance: torch.Tensor,
        anchor_points: torch.Tensor,
        stride: torch.Tensor,
    ) -> torch.Tensor:
        """将距离预测转换为边框坐标。

        Args:
            distance: [N, 4] (l, t, r, b)
            anchor_points: [N, 2] (cx, cy)
            stride: [N, 1]

        Returns:
            boxes: [N, 4] (x1, y1, x2, y2)
        """
        lt = distance[:, :2]
        rb = distance[:, 2:4]
        anchor_pixel = anchor_points * stride
        x1y1 = anchor_pixel - lt * stride
        x2y2 = anchor_pixel + rb * stride
        return torch.cat([x1y1, x2y2], dim=-1)

    @staticmethod
    def _ciou_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        """计算CIoU损失。

        Args:
            pred_boxes: [N, 4]
            gt_boxes: [N, 4]

        Returns:
            loss: [N]
        """
        # 交集
        inter_x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # 各自面积
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

        # 并集
        union = pred_area + gt_area - inter_area + 1e-7

        # IoU
        iou = inter_area / union

        # 最小外接框
        enclose_x1 = torch.min(pred_boxes[:, 0], gt_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], gt_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], gt_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], gt_boxes[:, 3])

        # 对角线距离
        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

        # 中心距离
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        center_dist = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

        # 宽高比一致性
        pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-7)
        pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-7)
        gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1e-7)
        gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1e-7)

        import math
        v = (4 / (math.pi ** 2)) * (torch.atan(gt_w / gt_h) - torch.atan(pred_w / pred_h)) ** 2
        alpha = v / (1 - iou + v + 1e-7)

        # CIoU = IoU - (center_dist/diag + alpha*v)
        ciou = iou - (center_dist / enclose_diag + alpha * v)

        return 1 - ciou

    def _freeze_backbone(self, num_stages: int):
        """冻结backbone的前N个stage。"""
        stages = self.backbone.get_stage_params()
        for i, stage in enumerate(stages[:num_stages]):
            for param in stage.parameters():
                param.requires_grad = False
        logger.info(f"Frozen backbone stages: 0-{num_stages-1}")

    def _freeze_text_encoder(self):
        """冻结文本编码器。"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen text encoder")

    def _log_model_info(self):
        """打印模型信息。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model built successfully:")
        logger.info(f"  Total parameters: {total/1e6:.2f}M")
        logger.info(f"  Trainable parameters: {trainable/1e6:.2f}M")
        logger.info(f"  Frozen parameters: {(total-trainable)/1e6:.2f}M")

    def clear_text_cache(self):
        """清除文本缓存（切换类别时调用）。"""
        self._cached_text_embeds = None
        self._cached_text_mask = None
        self._cached_text_key = None
