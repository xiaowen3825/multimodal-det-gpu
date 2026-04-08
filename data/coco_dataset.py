"""COCO数据集加载器。

支持：
- 开放词汇划分 (base/novel类别)
- 子集采样 (CPU训练友好)
- 文本标签生成
- 与数据增强管线集成

参考文献:
- OV-COCO: 48 base + 17 novel 标准划分
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# COCO 80类 (id -> name 映射)
COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
    59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

# OV-COCO标准划分: 48 base + 17 novel
OV_COCO_NOVEL_IDS = {8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35}
OV_COCO_BASE_IDS = set(COCO_CLASSES.keys()) - OV_COCO_NOVEL_IDS


class COCODataset(Dataset):
    """COCO目标检测数据集。

    Args:
        root: 数据集根目录
        ann_file: 标注文件路径（相对于root）
        img_dir: 图像目录（相对于root）
        img_size: 输入图像大小
        transforms: 数据增强
        mode: 'base' (仅base类), 'novel' (仅novel类), 'all' (全部)
        subset_ratio: 使用训练数据的比例 (0.0-1.0)
        prompt_template: 文本prompt模板
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        img_dir: str,
        img_size: int = 640,
        transforms=None,
        mode: str = "all",
        subset_ratio: float = 1.0,
        prompt_template: str = "a photo of a {}",
    ):
        self.root = root
        self.img_dir = os.path.join(root, img_dir)
        self.img_size = img_size
        self.transforms = transforms
        self.mode = mode
        self.prompt_template = prompt_template

        # 加载标注
        ann_path = os.path.join(root, ann_file)
        logger.info(f"Loading annotations from: {ann_path}")
        with open(ann_path, "r") as f:
            coco_data = json.load(f)

        # 构建索引
        self.images = {img["id"]: img for img in coco_data["images"]}

        # 确定使用的类别
        if mode == "base":
            self.valid_cat_ids = OV_COCO_BASE_IDS
        elif mode == "novel":
            self.valid_cat_ids = OV_COCO_NOVEL_IDS
        else:
            self.valid_cat_ids = set(COCO_CLASSES.keys())

        # 类别ID到连续索引的映射
        sorted_cat_ids = sorted(self.valid_cat_ids)
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
        self.idx_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_idx.items()}
        self.num_classes = len(self.cat_id_to_idx)

        # 类别名列表
        self.class_names = [COCO_CLASSES[self.idx_to_cat_id[i]] for i in range(self.num_classes)]

        # 按图像组织标注
        self.img_anns: Dict[int, List[Dict]] = {}
        for ann in coco_data["annotations"]:
            if ann["category_id"] not in self.valid_cat_ids:
                continue
            if ann.get("iscrowd", 0):
                continue
            img_id = ann["image_id"]
            if img_id not in self.img_anns:
                self.img_anns[img_id] = []
            self.img_anns[img_id].append(ann)

        # 只保留有标注的图像
        self.img_ids = sorted([
            img_id for img_id in self.img_anns.keys()
            if img_id in self.images
        ])

        # 子集采样
        if 0 < subset_ratio < 1.0:
            n_samples = max(1, int(len(self.img_ids) * subset_ratio))
            random.seed(42)
            self.img_ids = sorted(random.sample(self.img_ids, n_samples))

        logger.info(
            f"COCO dataset loaded: {len(self.img_ids)} images, "
            f"{self.num_classes} classes ({mode} mode), "
            f"subset={subset_ratio}"
        )

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]

        # 加载图像
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # 解析标注
        boxes = []
        labels = []
        for ann in self.img_anns.get(img_id, []):
            # COCO格式: [x, y, width, height] -> [x1, y1, x2, y2]
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + bw, y + bh
            # 裁剪到图像范围
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(self.cat_id_to_idx[ann["category_id"]])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)

        # 数据增强
        if self.transforms is not None:
            image, boxes, labels = self.transforms(image, boxes, labels)

        # Resize
        image, boxes = self._resize(image, boxes, self.img_size)

        # 转为Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.from_numpy(boxes).float()
        labels = torch.from_numpy(labels).long()

        # 生成该图像涉及的类别文本
        unique_labels = labels.unique().tolist()
        text_prompts = [self.prompt_template.format(self.class_names[l]) for l in unique_labels]

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "img_id": img_id,
            "text_prompts": text_prompts,
            "class_names": self.class_names,
        }

    @staticmethod
    def _resize(
        image: np.ndarray,
        boxes: np.ndarray,
        target_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """等比缩放并padding到目标尺寸。"""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Padding到正方形
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # 缩放边框
        if boxes.shape[0] > 0:
            boxes = boxes * scale

        return image, boxes

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """自定义batch收集函数。"""
        images = torch.stack([item["image"] for item in batch])
        targets = [
            {"boxes": item["boxes"], "labels": item["labels"]}
            for item in batch
        ]
        class_names = batch[0]["class_names"]

        return {
            "images": images,
            "targets": targets,
            "class_names": class_names,
            "img_ids": [item["img_id"] for item in batch],
        }


def build_dataloader(cfg: dict, split: str = "train") -> DataLoader:
    """根据配置构建数据加载器。

    Args:
        cfg: 配置字典
        split: 'train' 或 'val'

    Returns:
        DataLoader
    """
    data_cfg = cfg["data"]

    if split == "train":
        ann_file = data_cfg["train_ann"]
        img_dir = data_cfg["train_img_dir"]
        subset_ratio = data_cfg.get("subset_ratio", 1.0)
        mode = "base"  # 训练只用base类
    else:
        ann_file = data_cfg["val_ann"]
        img_dir = data_cfg["val_img_dir"]
        subset_ratio = 1.0
        mode = "all"   # 验证用所有类

    dataset = COCODataset(
        root=data_cfg["root"],
        ann_file=ann_file,
        img_dir=img_dir,
        img_size=data_cfg.get("img_size", 640),
        mode=mode,
        subset_ratio=subset_ratio,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"] if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=data_cfg.get("num_workers", 4),
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        pin_memory=False,  # CPU模式
        drop_last=(split == "train"),
        collate_fn=COCODataset.collate_fn,
    )

    return loader
