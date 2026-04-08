"""日志管理：统一日志格式、TensorBoard/WandB集成。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any


def setup_logger(
    name: str = "multimodal-det",
    log_dir: str | None = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """创建并配置logger。

    Args:
        name: Logger名称
        log_dir: 日志文件保存目录，None则只输出到控制台
        log_level: 日志级别

    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # 避免重复添加handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件输出
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "train.log"),
            mode="a",
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


class MetricLogger:
    """实验指标记录器，支持TensorBoard和WandB。

    Args:
        log_dir: 日志目录
        use_tensorboard: 是否使用TensorBoard
        use_wandb: 是否使用WandB
        project_name: WandB项目名
        experiment_name: 实验名
    """

    def __init__(
        self,
        log_dir: str = "./runs/logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        project_name: str = "multimodal-det",
        experiment_name: str = "default",
    ):
        self.log_dir = log_dir
        self.tb_writer = None
        self.wandb_run = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard", experiment_name)
                Path(tb_dir).mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
            except ImportError:
                logging.warning("TensorBoard not installed, skipping.")

        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=experiment_name,
                    dir=log_dir,
                    mode="offline",
                )
            except ImportError:
                logging.warning("WandB not installed, skipping.")

    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量指标。"""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, tag_value_dict: dict[str, float], step: int):
        """批量记录标量指标。"""
        for tag, value in tag_value_dict.items():
            self.log_scalar(tag, value, step)

    def log_image(self, tag: str, image, step: int):
        """记录图像。"""
        if self.tb_writer is not None:
            if hasattr(image, "numpy"):
                self.tb_writer.add_image(tag, image, step, dataformats="HWC")
            else:
                self.tb_writer.add_image(tag, image, step)

    def close(self):
        """关闭所有writer。"""
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
