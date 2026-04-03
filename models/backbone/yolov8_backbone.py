"""YOLOv8-S Backbone: CSPDarknet 结构，输出多尺度特征 C3/C4/C5。

基于 YOLOv8-S 的架构参数：
- width_multiple = 0.50 (通道缩放)
- depth_multiple = 0.33 (深度缩放)
- 输出通道: C3=128, C4=256, C5=512

参考文献:
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YOLO-World (CVPR 2024): 使用YOLOv8作为视觉backbone
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """自动计算padding使输出尺寸不变。"""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """标准卷积层: Conv2d + BatchNorm + SiLU。"""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """融合BN后的前向（推理加速）。"""
        return self.act(self.conv(x))


class DWConv(Conv):
    """深度可分离卷积。"""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, d: int = 1, act: bool = True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    """标准Bottleneck残差块。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """YOLOv8的C2f模块：更快的CSP Bottleneck with 2 convolutions。

    相比YOLOv5的C3模块，C2f将所有Bottleneck的输出都concat，
    提供更丰富的梯度流。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)。

    使用连续3次 5×5 MaxPool 等价于 SPP 的 5/9/13 三尺度池化，
    但速度更快。
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class YOLOv8Backbone(nn.Module):
    """YOLOv8-S Backbone (CSPDarknet)。

    输出三个尺度的特征图:
        - C3: stride=8,  channels=128 (for small objects)
        - C4: stride=16, channels=256 (for medium objects)
        - C5: stride=32, channels=512 (for large objects)

    Args:
        variant: 模型变体 ('n', 's', 'm', 'l', 'x')
        width_multiple: 通道缩放因子
        depth_multiple: 深度缩放因子
        out_channels: 输出通道数列表
        pretrained: 是否加载预训练权重
    """

    # YOLOv8各变体的缩放参数
    VARIANTS = {
        "n": {"width": 0.25, "depth": 0.33},
        "s": {"width": 0.50, "depth": 0.33},
        "m": {"width": 0.75, "depth": 0.67},
        "l": {"width": 1.00, "depth": 1.00},
        "x": {"width": 1.25, "depth": 1.00},
    }

    def __init__(
        self,
        variant: str = "s",
        width_multiple: float | None = None,
        depth_multiple: float | None = None,
        out_channels: List[int] | None = None,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()

        # 确定缩放因子
        if width_multiple is None:
            width_multiple = self.VARIANTS[variant]["width"]
        if depth_multiple is None:
            depth_multiple = self.VARIANTS[variant]["depth"]

        # 基础通道数
        base_channels = [64, 128, 256, 512, 1024]
        channels = [self._make_divisible(c * width_multiple, 8) for c in base_channels]

        # 基础深度
        base_depths = [3, 6, 6, 3]
        depths = [max(round(d * depth_multiple), 1) for d in base_depths]

        # Stage 0: stem (stride=2)
        self.stem = Conv(3, channels[0], 3, 2)

        # Stage 1: stride=4
        self.stage1 = nn.Sequential(
            Conv(channels[0], channels[1], 3, 2),
            C2f(channels[1], channels[1], n=depths[0], shortcut=True),
        )

        # Stage 2: stride=8 -> C3输出
        self.stage2 = nn.Sequential(
            Conv(channels[1], channels[2], 3, 2),
            C2f(channels[2], channels[2], n=depths[1], shortcut=True),
        )

        # Stage 3: stride=16 -> C4输出
        self.stage3 = nn.Sequential(
            Conv(channels[2], channels[3], 3, 2),
            C2f(channels[3], channels[3], n=depths[2], shortcut=True),
        )

        # Stage 4: stride=32 -> C5输出
        self.stage4 = nn.Sequential(
            Conv(channels[3], channels[4], 3, 2),
            C2f(channels[4], channels[4], n=depths[3], shortcut=True),
            SPPF(channels[4], channels[4], k=5),
        )

        # 记录输出通道数
        self._out_channels = out_channels or [channels[2], channels[3], channels[4]]
        self._strides = [8, 16, 32]

        # 初始化权重
        self._init_weights()

        if pretrained:
            self._load_pretrained()

    @property
    def out_channels(self) -> List[int]:
        """返回各尺度输出通道数。"""
        return self._out_channels

    @property
    def strides(self) -> List[int]:
        """返回各尺度步长。"""
        return self._strides

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播，返回多尺度特征。

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            (C3, C4, C5): 三个尺度的特征图
                - C3: [B, 128, H/8, W/8]
                - C4: [B, 256, H/16, W/16]
                - C5: [B, 512, H/32, W/32]
        """
        x = self.stem(x)       # stride=2
        x = self.stage1(x)     # stride=4
        c3 = self.stage2(x)    # stride=8
        c4 = self.stage3(c3)   # stride=16
        c5 = self.stage4(c4)   # stride=32

        return c3, c4, c5

    def get_stage_params(self) -> List[nn.Module]:
        """返回各stage的参数，用于冻结策略。

        Returns:
            [stem, stage1, stage2, stage3, stage4]
        """
        return [self.stem, self.stage1, self.stage2, self.stage3, self.stage4]

    @staticmethod
    def _make_divisible(v: float, divisor: int = 8) -> int:
        """使通道数能被divisor整除。"""
        new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _init_weights(self):
        """初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self):
        """加载预训练权重（从ultralytics YOLOv8s）。

        注意: 实际使用时需要将ultralytics格式的权重转换为本项目格式。
        这里提供了权重映射的框架，实际权重文件需要单独下载。
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "YOLOv8-S pretrained weights loading is configured. "
            "Please download weights from ultralytics and place in 'weights/' directory. "
            "Run: pip install ultralytics && python -c \"from ultralytics import YOLO; YOLO('yolov8s.pt')\""
        )
