"""交互式检测演示脚本。

用法:
    python scripts/demo.py --checkpoint runs/best.pth --image demo.jpg --text "person, car, dog"
    python scripts/demo.py --checkpoint runs/best.pth --image_dir ./test_images --text "cat, dog"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Detection Demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重")
    parser.add_argument("--config", type=str, default="configs/agcma_yoloworld_s.yaml", help="模型配置")
    parser.add_argument("--image", type=str, default="", help="单张图片路径")
    parser.add_argument("--image_dir", type=str, default="", help="图片目录")
    parser.add_argument("--text", type=str, required=True, help="检测类别，逗号分隔: 'person, car, dog'")
    parser.add_argument("--score_thr", type=float, default=0.3, help="置信度阈值")
    parser.add_argument("--output_dir", type=str, default="./runs/demo", help="输出目录")
    parser.add_argument("--img_size", type=int, default=640, help="输入图像大小")
    return parser.parse_args()


def preprocess_image(image_path: str, img_size: int = 640):
    """预处理图像。"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_image = image.copy()
    h, w = image.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    image = cv2.resize(image, (new_w, new_h))
    # Padding
    pad_h = img_size - new_h
    pad_w = img_size - new_w
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR -> RGB, HWC -> CHW, normalize
    tensor = torch.from_numpy(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ).permute(2, 0, 1).float() / 255.0

    return tensor.unsqueeze(0), orig_image, scale


def main():
    args = parse_args()

    from scripts.train import load_config
    from models import build_model
    from utils.checkpoint import load_checkpoint
    from utils.visualize import draw_detections

    # 加载模型
    cfg = load_config(os.path.join(PROJECT_ROOT, args.config))
    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    # 解析类别
    class_names = [name.strip() for name in args.text.split(",")]
    print(f"Detecting classes: {class_names}")

    # 收集图片
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([
            str(p) for p in Path(args.image_dir).iterdir()
            if p.suffix.lower() in exts
        ])
    else:
        print("Please provide --image or --image_dir")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        # 预处理
        input_tensor, orig_image, scale = preprocess_image(img_path, args.img_size)

        # 推理
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(images=input_tensor, text_prompts=class_names)
        latency = (time.perf_counter() - start) * 1000

        # 后处理
        cls_scores = outputs["cls_scores"][0]   # [A, N]
        bbox_preds = outputs["bbox_preds"][0]   # [A, 4] 已解码的绝对坐标
        objectness = outputs["objectness"][0]   # [A, 1]

        # 简单后处理
        obj = torch.sigmoid(objectness.squeeze(-1))
        if cls_scores is not None:
            scores = torch.sigmoid(cls_scores) * obj.unsqueeze(-1)
            max_scores, max_indices = scores.max(dim=-1)
        else:
            max_scores = obj
            max_indices = torch.zeros_like(obj, dtype=torch.long)

        # 过滤
        mask = max_scores > args.score_thr
        if mask.sum() > 0:
            det_scores = max_scores[mask].numpy()
            det_labels = [class_names[idx] for idx in max_indices[mask].numpy()]

            # bbox_preds已经是绝对像素坐标(x1,y1,x2,y2)
            det_boxes = bbox_preds[mask].clamp(min=0, max=args.img_size)

            # 还原到原图坐标
            det_boxes = det_boxes.numpy() / scale

            # 可视化
            vis = draw_detections(orig_image, det_boxes, det_labels, det_scores, args.score_thr)
            n_det = len(det_scores)
        else:
            vis = orig_image
            n_det = 0

        # 保存
        name = Path(img_path).stem
        save_path = os.path.join(args.output_dir, f"{name}_det.jpg")
        cv2.imwrite(save_path, vis)

        print(f"  {Path(img_path).name}: {n_det} detections, {latency:.1f}ms -> {save_path}")

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
