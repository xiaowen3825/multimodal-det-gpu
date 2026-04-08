"""评估入口脚本。

用法:
    python scripts/eval.py --config configs/eval_open_vocab.yaml --checkpoint runs/best.pth
    python scripts/eval.py --config configs/eval_open_vocab.yaml --checkpoint runs/best.pth --benchmark
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger
from utils.metrics import count_parameters, count_flops, benchmark_speed, format_metrics_table


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Detection Evaluation")
    parser.add_argument("--config", type=str, required=True, help="评估配置文件")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument("--benchmark", action="store_true", help="是否进行推理速度测试")
    parser.add_argument("--vis", action="store_true", help="是否保存可视化结果")
    parser.add_argument("--vis_dir", type=str, default="./runs/vis", help="可视化保存目录")
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    from scripts.train import load_config
    cfg = load_config(args.config)

    logger = setup_logger("eval")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # 构建模型
    from models import build_model
    from utils.checkpoint import load_checkpoint

    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    # 模型统计
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"Parameters: {total_params/1e6:.2f}M (trainable: {trainable_params/1e6:.2f}M)")

    # 数据加载
    from data.coco_dataset import build_dataloader
    val_loader = build_dataloader(cfg, split="val")

    # 评估
    from engine.evaluator import Evaluator
    evaluator = Evaluator(cfg, val_loader.dataset.class_names)
    metrics = evaluator.evaluate(model, val_loader)

    # 打印结果
    print(format_metrics_table(metrics, "COCO Evaluation Results"))

    # 推理速度测试
    if args.benchmark:
        logger.info("Running inference benchmark...")
        text_prompts = val_loader.dataset.class_names
        speed_metrics = benchmark_speed(
            model,
            input_shape=(1, 3, 640, 640),
            text_input=text_prompts,
        )
        metrics.update(speed_metrics)
        metrics["params_M"] = total_params / 1e6

        try:
            flops = count_flops(model, text_input=text_prompts)
            metrics["GFLOPs"] = flops / 1e9
        except Exception as e:
            logger.warning(f"FLOPs computation failed: {e}")

        print(format_metrics_table(speed_metrics, "Inference Benchmark"))

    # 可视化
    if args.vis:
        from utils.visualize import draw_detections
        import cv2
        import numpy as np

        os.makedirs(args.vis_dir, exist_ok=True)
        logger.info(f"Saving visualizations to {args.vis_dir}")

        vis_count = cfg.get("visualization", {}).get("num_images", 50)
        for i, batch in enumerate(val_loader):
            if i >= vis_count:
                break

            with torch.no_grad():
                outputs = model(batch["images"], text_prompts=val_loader.dataset.class_names)

            # 简化可视化
            img = (batch["images"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            save_path = os.path.join(args.vis_dir, f"vis_{i:04d}.jpg")
            cv2.imwrite(save_path, img)

        logger.info(f"Saved {min(vis_count, len(val_loader))} visualizations")

    # 保存评估结果
    import json
    result_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
