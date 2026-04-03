"""数据增强管线。

CPU友好的数据增强，包括：
- 随机水平翻转
- 随机缩放 + 裁剪
- 颜色抖动 (HSV)
- Mosaic (4图拼接)
- MixUp (图像混合)

参考: YOLOv8 的增强策略，简化版本适合CPU训练。
"""

from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np


class Compose:
    """组合多个变换。"""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels


class RandomHorizontalFlip:
    """随机水平翻转。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            h, w = image.shape[:2]
            image = np.fliplr(image).copy()
            if boxes.shape[0] > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, boxes, labels


class RandomHSV:
    """HSV颜色空间抖动。"""

    def __init__(self, h_gain: float = 0.015, s_gain: float = 0.7, v_gain: float = 0.4):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, image, boxes, labels):
        r = np.random.uniform(-1, 1, 3) * [self.h_gain, self.s_gain, self.v_gain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        hsv = cv2.merge([
            cv2.LUT(hue, lut_hue),
            cv2.LUT(sat, lut_sat),
            cv2.LUT(val, lut_val),
        ])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return image, boxes, labels


class RandomScale:
    """随机缩放。"""

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.5)):
        self.scale_range = scale_range

    def __call__(self, image, boxes, labels):
        scale = random.uniform(*self.scale_range)
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if boxes.shape[0] > 0:
            boxes = boxes * scale

        return image, boxes, labels


class RandomTranslate:
    """随机平移。"""

    def __init__(self, max_translate: float = 0.1):
        self.max_translate = max_translate

    def __call__(self, image, boxes, labels):
        h, w = image.shape[:2]
        tx = random.uniform(-self.max_translate, self.max_translate) * w
        ty = random.uniform(-self.max_translate, self.max_translate) * h

        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))

        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] += tx
            boxes[:, [1, 3]] += ty
            # 裁剪到图像边界
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            # 过滤无效框
            valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes = boxes[valid]
            labels = labels[valid]

        return image, boxes, labels


def build_transforms(cfg: dict, is_train: bool = True) -> Compose | None:
    """根据配置构建数据增强管线。

    Args:
        cfg: augmentation配置
        is_train: 是否为训练模式

    Returns:
        Compose变换或None
    """
    if not is_train:
        return None

    aug_cfg = cfg.get("augmentation", {})

    transforms = [
        RandomHorizontalFlip(p=aug_cfg.get("flip_lr", 0.5)),
        RandomHSV(
            h_gain=aug_cfg.get("hsv_h", 0.015),
            s_gain=aug_cfg.get("hsv_s", 0.7),
            v_gain=aug_cfg.get("hsv_v", 0.4),
        ),
        RandomScale(scale_range=tuple(aug_cfg.get("scale", [0.5, 1.5]))),
        RandomTranslate(max_translate=aug_cfg.get("translate", 0.1)),
    ]

    return Compose(transforms)
