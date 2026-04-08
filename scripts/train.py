"""训练入口脚本。

用法:
    python scripts/train.py --config configs/agcma_yoloworld_s.yaml
    python scripts/train.py --config configs/agcma_yoloworld_s.yaml --resume runs/epoch_10.pth
    python scripts/train.py --config configs/distill_text_encoder.yaml  # 蒸馏模式
"""

from __future__ import annotations

import argparse
import os
import sys
import random

import numpy as np
import torch
import yaml

# 项目根目录加入path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Detection Training")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的checkpoint路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件，支持继承。"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 处理继承
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(config_path), cfg.pop("_base_"))
        base_cfg = load_config(base_path)
        # 递归合并
        merged = deep_merge(base_cfg, cfg)
        return merged

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并字典。"""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def set_seed(seed: int):
    """设置所有随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()

    # 加载配置
    cfg = load_config(args.config)
    seed = args.seed or cfg.get("seed", 42)
    set_seed(seed)

    # 设置日志
    log_dir = cfg.get("logging", {}).get("log_dir", "./runs/logs")
    logger = setup_logger("multimodal-det", log_dir=log_dir)
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {seed}")

    # 检查是否为蒸馏模式
    if "distill" in cfg:
        train_distill(cfg, logger)
    else:
        train_detection(cfg, args.resume, logger)


def train_detection(cfg: dict, resume: str, logger):
    """检测模型训练。"""
    from models import build_model
    from data.coco_dataset import build_dataloader
    from engine.trainer import Trainer
    from engine.evaluator import Evaluator

    logger.info("=== Detection Training Mode ===")

    # 构建模型
    model = build_model(cfg)
    logger.info(f"Model built: {type(model).__name__}")

    # 构建数据
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")
    logger.info(f"Train: {len(train_loader.dataset)} images")
    logger.info(f"Val: {len(val_loader.dataset)} images")

    # 评估器
    evaluator = Evaluator(cfg, train_loader.dataset.class_names)

    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        evaluator=evaluator,
    )

    # 恢复训练
    if resume:
        trainer.resume(resume)

    # 开始训练
    trainer.train()


def train_distill(cfg: dict, logger):
    """文本编码器蒸馏训练。"""
    from models.text_encoder import CLIPTextEncoder, LightTextEncoder
    from models.text_encoder.distill_trainer import DistillTrainer

    logger.info("=== Text Encoder Distillation Mode ===")

    # 构建Teacher
    teacher_cfg = cfg["teacher"]
    teacher = CLIPTextEncoder(
        model_name=teacher_cfg.get("model_name", "openai/clip-vit-base-patch32"),
        embed_dim=teacher_cfg.get("embed_dim", 512),
        freeze=True,
    )

    # 构建Student
    student_cfg = cfg["student"]
    student = LightTextEncoder(
        vocab_size=student_cfg.get("vocab_size", 49408),
        embed_dim=student_cfg.get("embed_dim", 512),
        num_layers=student_cfg.get("num_layers", 4),
        num_heads=student_cfg.get("num_heads", 8),
        hidden_dim=student_cfg.get("hidden_dim", 2048),
        max_length=student_cfg.get("max_length", 77),
    )

    # 蒸馏训练
    distill_cfg = cfg.get("distill", {})
    trainer = DistillTrainer(teacher=teacher, student=student, cfg=cfg)
    trainer.train(
        text_file=distill_cfg.get("text_data", "./datasets/cc3m_texts.txt"),
        num_samples=distill_cfg.get("num_samples", 100000),
    )


if __name__ == "__main__":
    main()
