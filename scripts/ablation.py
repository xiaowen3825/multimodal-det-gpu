"""消融实验脚本。

自动运行各消融配置并汇总结果。

消融实验设计:
1. Full AGCMA (完整模型)
2. w/o Gate: 去掉自适应门控，直接相加
3. w/o DWConv: 去掉深度可分离卷积局部增强
4. Spatial Attn: 用标准空间cross-attention替代通道亲和力注意力
5. RepVL-PAN: 原始YOLO-World基线

用法:
    python scripts/ablation.py --config configs/agcma_yoloworld_s.yaml --checkpoint_dir runs/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


ABLATION_VARIANTS = {
    "full_agcma": {
        "description": "完整AGCMA模型",
        "neck_type": "AGCMA_PAN",
        "agcma_module": "AGCMAModule",
    },
    "no_gate": {
        "description": "去掉自适应门控 (直接相加)",
        "neck_type": "AGCMA_PAN",
        "agcma_module": "AGCMAModule_NoGate",
    },
    "no_dwconv": {
        "description": "去掉深度可分离卷积局部增强",
        "neck_type": "AGCMA_PAN",
        "agcma_module": "AGCMAModule_NoDW",
    },
    "spatial_attn": {
        "description": "标准空间Cross-Attention (对比)",
        "neck_type": "AGCMA_PAN",
        "agcma_module": "AGCMAModule_SpatialAttn",
    },
    "repvl_pan": {
        "description": "RepVL-PAN基线 (YOLO-World)",
        "neck_type": "RepVL_PAN",
        "agcma_module": None,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--config", type=str, required=True, help="基础配置文件")
    parser.add_argument("--checkpoint_dir", type=str, default="./runs/ablation", help="checkpoint目录")
    parser.add_argument("--variants", type=str, nargs="+", default=list(ABLATION_VARIANTS.keys()),
                       help="要运行的消融变体")
    parser.add_argument("--eval_only", action="store_true", help="只评估不训练")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练epochs")
    return parser.parse_args()


def build_variant_model(cfg: dict, variant_name: str):
    """根据消融变体配置构建模型。"""
    variant = ABLATION_VARIANTS[variant_name]

    from models.backbone import YOLOv8Backbone
    from models.text_encoder import LightTextEncoder
    from models.head import DecoupledHead
    from models.detector import MultiModalDetector

    model_cfg = cfg["model"]

    # Backbone
    backbone = YOLOv8Backbone(**model_cfg["backbone"])

    # Text encoder
    text_encoder = LightTextEncoder(**model_cfg["text_encoder"])

    # Neck (根据变体选择)
    if variant["neck_type"] == "RepVL_PAN":
        from models.neck import RepVL_PAN
        neck = RepVL_PAN(
            in_channels=model_cfg["neck"]["in_channels"],
            text_dim=model_cfg["neck"]["text_dim"],
            out_channels=model_cfg["neck"]["out_channels"],
        )
    else:
        from models.neck.agcma_pan import AGCMA_PAN
        from models.neck import agcma_module as agcma_mod

        # 替换AGCMA模块类
        module_cls = getattr(agcma_mod, variant["agcma_module"])

        # 需要monkey-patch AGCMA_PAN中使用的AGCMAModule
        original_cls = agcma_mod.AGCMAModule
        agcma_mod.AGCMAModule = module_cls

        neck_cfg = copy.deepcopy(model_cfg["neck"])
        neck_type = neck_cfg.pop("type", "AGCMA_PAN")
        neck = AGCMA_PAN(**neck_cfg)

        # 恢复
        agcma_mod.AGCMAModule = original_cls

    # Head
    head = DecoupledHead(**model_cfg["head"])

    # Detector
    freeze_cfg = model_cfg.get("freeze", {})
    detector = MultiModalDetector(
        backbone=backbone,
        text_encoder=text_encoder,
        neck=neck,
        head=head,
        freeze_backbone_stages=freeze_cfg.get("backbone_stages", 0),
        freeze_text_encoder=freeze_cfg.get("text_encoder", True),
    )

    return detector


def main():
    args = parse_args()

    from scripts.train import load_config
    from utils.logger import setup_logger
    from utils.metrics import count_parameters, format_metrics_table

    cfg = load_config(os.path.join(PROJECT_ROOT, args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    logger = setup_logger("ablation")
    logger.info(f"Running ablation study with variants: {args.variants}")

    results = {}

    for variant_name in args.variants:
        variant = ABLATION_VARIANTS[variant_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Variant: {variant_name} - {variant['description']}")
        logger.info(f"{'='*60}")

        # 构建模型
        model = build_variant_model(cfg, variant_name)

        # 统计
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        logger.info(f"  Params: {total_params/1e6:.2f}M (trainable: {trainable_params/1e6:.2f}M)")

        results[variant_name] = {
            "description": variant["description"],
            "params_M": total_params / 1e6,
            "trainable_params_M": trainable_params / 1e6,
        }

        if not args.eval_only:
            logger.info(f"  Training would start here (use --eval_only for param analysis only)")
            # 实际训练可以在这里接入
            # trainer = Trainer(model, ..., cfg=cfg)
            # trainer.train()

    # 汇总结果表
    logger.info(f"\n{'='*60}")
    logger.info("Ablation Study Summary")
    logger.info(f"{'='*60}")

    header = f"{'Variant':<20s} {'Description':<35s} {'Params(M)':>10s} {'Trainable(M)':>12s}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, res in results.items():
        logger.info(
            f"{name:<20s} {res['description']:<35s} "
            f"{res['params_M']:>10.2f} {res['trainable_params_M']:>12.2f}"
        )

    # 保存结果
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    result_path = os.path.join(args.checkpoint_dir, "ablation_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
