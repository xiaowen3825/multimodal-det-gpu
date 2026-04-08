from .detector import MultiModalDetector
from .backbone import YOLOv8Backbone
from .text_encoder import CLIPTextEncoder, LightTextEncoder
from .neck import AGCMAModule, AGCMA_PAN, RepVL_PAN
from .head import DecoupledHead


def build_model(cfg: dict) -> MultiModalDetector:
    """根据配置构建完整检测器。

    Args:
        cfg: 模型配置字典

    Returns:
        MultiModalDetector 实例
    """
    model_cfg = cfg["model"]

    # 构建各组件
    backbone = YOLOv8Backbone(**model_cfg["backbone"])
    text_encoder = _build_text_encoder(model_cfg["text_encoder"])
    neck = _build_neck(model_cfg["neck"])
    head = DecoupledHead(**model_cfg["head"])

    # 组装检测器
    detector = MultiModalDetector(
        backbone=backbone,
        text_encoder=text_encoder,
        neck=neck,
        head=head,
        freeze_backbone_stages=model_cfg.get("freeze", {}).get("backbone_stages", 0),
        freeze_text_encoder=model_cfg.get("freeze", {}).get("text_encoder", False),
    )

    return detector


def _build_text_encoder(cfg: dict):
    """构建文本编码器。"""
    enc_type = cfg.pop("type", "LightTextEncoder")
    if enc_type == "CLIPTextEncoder":
        return CLIPTextEncoder(**cfg)
    elif enc_type == "LightTextEncoder":
        return LightTextEncoder(**cfg)
    else:
        raise ValueError(f"Unknown text encoder type: {enc_type}")


def _build_neck(cfg: dict):
    """构建融合网络。"""
    neck_type = cfg.pop("type", "AGCMA_PAN")
    if neck_type == "AGCMA_PAN":
        return AGCMA_PAN(**cfg)
    elif neck_type == "RepVL_PAN":
        return RepVL_PAN(**cfg)
    else:
        raise ValueError(f"Unknown neck type: {neck_type}")
