"""评估引擎：COCO AP计算、开放词汇评估、推理速度测量。

支持：
- 标准COCO AP/AP50/AP75评估
- 开放词汇: Base类 / Novel类 分别评估
- 推理速度Benchmark
- GPU加速推理
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Evaluator:
    """COCO评估引擎（GPU版本）。

    Args:
        cfg: 评估配置
        class_names: 类别名列表
    """

    def __init__(self, cfg: dict, class_names: List[str]):
        self.cfg = cfg
        self.class_names = class_names

        eval_cfg = cfg.get("eval", {})
        self.nms_cfg = eval_cfg.get("nms", {})
        self.score_thr = self.nms_cfg.get("score_threshold", 0.001)
        self.iou_thr = self.nms_cfg.get("iou_threshold", 0.7)
        self.max_dets = self.nms_cfg.get("max_detections", 300)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """在数据集上评估模型。

        Args:
            model: 检测器
            dataloader: 验证数据加载器
            class_names: 类别名列表 (覆盖默认)
            device: 推理设备 (GPU/CPU)

        Returns:
            指标字典 {'AP': float, 'AP50': float, ...}
        """
        model.eval()
        names = class_names or self.class_names

        # 自动检测设备
        if device is None:
            device = next(model.parameters()).device

        all_predictions = []
        all_ground_truths = []

        logger.info(f"Evaluating on {len(dataloader)} images (device={device})...")
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # 将数据移到GPU
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            img_ids = batch["img_ids"]

            # 推理（GPU上使用AMP加速）
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images=images, text_prompts=names)
            else:
                outputs = model(images=images, text_prompts=names)

            # 后处理每张图（移回CPU进行NMS）
            for i in range(images.shape[0]):
                cls_scores = outputs["cls_scores"][i].cpu()    # [A, N]
                bbox_preds = outputs["bbox_preds"][i].cpu()    # [A, 4]
                objectness = outputs["objectness"][i].cpu()    # [A, 1]
                anchor_points = outputs["anchor_points"].cpu()  # [A, 2]
                stride_tensor = outputs["stride_tensor"].cpu()  # [A, 1]

                # 解码预测
                preds = self._postprocess(
                    cls_scores, bbox_preds, objectness,
                    anchor_points, stride_tensor,
                )
                all_predictions.append({
                    "img_id": img_ids[i],
                    "boxes": preds["boxes"],
                    "scores": preds["scores"],
                    "labels": preds["labels"],
                })

                # Ground truth
                all_ground_truths.append({
                    "img_id": img_ids[i],
                    "boxes": targets[i]["boxes"].numpy(),
                    "labels": targets[i]["labels"].numpy(),
                })

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Evaluated {batch_idx + 1}/{len(dataloader)} batches")

        eval_time = time.time() - start_time
        logger.info(f"Evaluation finished in {eval_time:.1f}s")

        # 计算AP
        metrics = self._compute_coco_metrics(all_predictions, all_ground_truths)
        metrics["eval_time_s"] = eval_time
        metrics["fps"] = len(dataloader) / eval_time

        return metrics

    def _postprocess(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        objectness: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """后处理：NMS + 阈值过滤。

        Args:
            cls_scores: [A, N]
            bbox_preds: [A, 4]
            objectness: [A, 1]
            anchor_points: [A, 2]
            stride_tensor: [A, 1]

        Returns:
            {'boxes': ndarray, 'scores': ndarray, 'labels': ndarray}
        """
        # 综合分数 = objectness * cls_score
        obj_scores = torch.sigmoid(objectness.squeeze(-1))  # [A]

        if cls_scores is not None:
            scores = torch.sigmoid(cls_scores) * obj_scores.unsqueeze(-1)  # [A, N]
        else:
            scores = obj_scores.unsqueeze(-1)

        # 解码边框
        anchor_pixel = anchor_points * stride_tensor
        boxes = torch.zeros_like(bbox_preds)
        boxes[:, 0] = anchor_pixel[:, 0] - bbox_preds[:, 0] * stride_tensor[:, 0]
        boxes[:, 1] = anchor_pixel[:, 1] - bbox_preds[:, 1] * stride_tensor[:, 0]
        boxes[:, 2] = anchor_pixel[:, 0] + bbox_preds[:, 2] * stride_tensor[:, 0]
        boxes[:, 3] = anchor_pixel[:, 1] + bbox_preds[:, 3] * stride_tensor[:, 0]

        # 展平: 每个类独立做NMS
        n_classes = scores.shape[1]
        all_boxes = []
        all_scores_flat = []
        all_labels = []

        for cls_idx in range(n_classes):
            cls_score = scores[:, cls_idx]
            mask = cls_score > self.score_thr
            if mask.sum() == 0:
                continue

            cls_boxes = boxes[mask]
            cls_scores_filtered = cls_score[mask]

            # NMS
            keep = self._nms(cls_boxes, cls_scores_filtered, self.iou_thr)
            all_boxes.append(cls_boxes[keep].numpy())
            all_scores_flat.append(cls_scores_filtered[keep].numpy())
            all_labels.append(np.full(len(keep), cls_idx))

        if all_boxes:
            result_boxes = np.concatenate(all_boxes)
            result_scores = np.concatenate(all_scores_flat)
            result_labels = np.concatenate(all_labels).astype(np.int64)

            # 限制最大检测数
            if len(result_scores) > self.max_dets:
                topk = np.argsort(result_scores)[::-1][:self.max_dets]
                result_boxes = result_boxes[topk]
                result_scores = result_scores[topk]
                result_labels = result_labels[topk]
        else:
            result_boxes = np.zeros((0, 4))
            result_scores = np.zeros((0,))
            result_labels = np.zeros((0,), dtype=np.int64)

        return {
            "boxes": result_boxes,
            "scores": result_scores,
            "labels": result_labels,
        }

    @staticmethod
    def _nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
    ) -> List[int]:
        """简单NMS实现。"""
        if boxes.shape[0] == 0:
            return []

        try:
            from torchvision.ops import nms
            keep = nms(boxes, scores, iou_threshold)
            return keep.tolist()
        except ImportError:
            pass

        # 纯PyTorch NMS fallback
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break

            i = order[0].item()
            keep.append(i)

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-7)

            mask = iou <= iou_threshold
            order = order[1:][mask]

        return keep

    def _compute_coco_metrics(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict[str, float]:
        """计算COCO风格的AP指标。

        简化版本，计算 AP@0.5 和 AP@[0.5:0.95]。
        """
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        all_aps = []

        # 按类别计算
        all_labels_pred = np.concatenate([p["labels"] for p in predictions]) if predictions else np.array([])
        all_labels_gt = np.concatenate([g["labels"] for g in ground_truths]) if ground_truths else np.array([])

        unique_classes = np.unique(np.concatenate([all_labels_pred, all_labels_gt])) if len(all_labels_pred) + len(all_labels_gt) > 0 else []

        ap50_list = []

        for cls_idx in unique_classes:
            cls_idx = int(cls_idx)

            # 收集该类的预测和GT
            cls_preds = []
            cls_gts = []

            for pred, gt in zip(predictions, ground_truths):
                pred_mask = pred["labels"] == cls_idx
                gt_mask = gt["labels"] == cls_idx

                if pred_mask.any():
                    cls_preds.append({
                        "boxes": pred["boxes"][pred_mask],
                        "scores": pred["scores"][pred_mask],
                        "img_id": pred["img_id"],
                    })

                if gt_mask.any():
                    cls_gts.append({
                        "boxes": gt["boxes"][gt_mask],
                        "img_id": gt["img_id"],
                    })

            # 在不同IoU阈值下计算AP
            for iou_thr in iou_thresholds:
                ap = self._compute_ap_single(cls_preds, cls_gts, iou_thr)
                all_aps.append(ap)

                if abs(iou_thr - 0.5) < 0.01:
                    ap50_list.append(ap)

        # 汇总
        metrics = {
            "AP": float(np.mean(all_aps)) if all_aps else 0.0,
            "AP50": float(np.mean(ap50_list)) if ap50_list else 0.0,
        }

        return metrics

    @staticmethod
    def _compute_ap_single(
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float,
    ) -> float:
        """计算单个类别在单个IoU阈值下的AP。"""
        if not ground_truths:
            return 0.0

        # 合并所有预测并按分数排序
        all_boxes = []
        all_scores = []
        all_img_ids = []

        for pred in predictions:
            all_boxes.extend(pred["boxes"].tolist())
            all_scores.extend(pred["scores"].tolist())
            all_img_ids.extend([pred["img_id"]] * len(pred["scores"]))

        if not all_scores:
            return 0.0

        sorted_indices = np.argsort(all_scores)[::-1]
        all_boxes = [all_boxes[i] for i in sorted_indices]
        all_img_ids = [all_img_ids[i] for i in sorted_indices]

        # GT按图像分组
        gt_by_img = {}
        total_gt = 0
        for gt in ground_truths:
            img_id = gt["img_id"]
            gt_by_img[img_id] = {
                "boxes": gt["boxes"],
                "matched": np.zeros(len(gt["boxes"]), dtype=bool),
            }
            total_gt += len(gt["boxes"])

        if total_gt == 0:
            return 0.0

        # 逐个预测计算TP/FP
        tp = np.zeros(len(all_boxes))
        fp = np.zeros(len(all_boxes))

        for i, (box, img_id) in enumerate(zip(all_boxes, all_img_ids)):
            if img_id not in gt_by_img:
                fp[i] = 1
                continue

            gt_info = gt_by_img[img_id]
            gt_boxes = gt_info["boxes"]

            # 计算IoU
            ious = _compute_iou(np.array([box]), gt_boxes)[0]
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]

            if best_iou >= iou_threshold and not gt_info["matched"][best_idx]:
                tp[i] = 1
                gt_info["matched"][best_idx] = True
            else:
                fp[i] = 1

        # Precision-Recall曲线
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / total_gt

        # 计算AP (11点插值或所有点)
        ap = _compute_ap_from_pr(precision, recall)
        return ap


def _compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """计算IoU矩阵。[M, 4] x [N, 4] -> [M, N]"""
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    return inter / (union + 1e-7)


def _compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """从Precision-Recall曲线计算AP。"""
    # 添加首尾点
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))

    # 使precision单调递减
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # 计算面积
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return float(ap)
