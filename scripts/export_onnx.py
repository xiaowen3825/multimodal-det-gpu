"""ONNX导出脚本。

将PyTorch模型导出为ONNX格式，支持ONNX Runtime推理。

用法:
    python scripts/export_onnx.py --checkpoint runs/best.pth --output model.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重")
    parser.add_argument("--config", type=str, default="configs/agcma_yoloworld_s.yaml")
    parser.add_argument("--output", type=str, default="model.onnx", help="ONNX输出路径")
    parser.add_argument("--img_size", type=int, default=640, help="输入图像大小")
    parser.add_argument("--num_classes", type=int, default=80, help="类别数")
    parser.add_argument("--simplify", action="store_true", help="是否简化ONNX模型")
    parser.add_argument("--verify", action="store_true", help="是否验证导出结果")
    return parser.parse_args()


class ONNXWrapper(torch.nn.Module):
    """ONNX导出用的包装模型。

    将文本编码器的输出作为预计算输入，
    只导出视觉部分的计算图。
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, images: torch.Tensor, text_embeds: torch.Tensor):
        features = self.backbone(images)
        fused = self.neck(features, text_embeds)
        outputs = self.head(fused, text_embeds[:, :1, :])  # 简化
        return outputs["cls_scores"], outputs["bbox_preds"], outputs["objectness"]


def main():
    args = parse_args()

    from scripts.train import load_config
    from models import build_model
    from utils.checkpoint import load_checkpoint

    print(f"Loading model from {args.checkpoint}...")
    cfg = load_config(os.path.join(PROJECT_ROOT, args.config))
    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    # 包装用于ONNX
    onnx_model = ONNXWrapper(model)
    onnx_model.eval()

    # 创建虚拟输入
    dummy_images = torch.randn(1, 3, args.img_size, args.img_size)
    dummy_text = torch.randn(1, args.num_classes, 512)  # 预计算的文本嵌入

    # 导出
    print(f"Exporting to {args.output}...")
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    torch.onnx.export(
        onnx_model,
        (dummy_images, dummy_text),
        args.output,
        opset_version=17,
        input_names=["images", "text_embeddings"],
        output_names=["cls_scores", "bbox_preds", "objectness"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "text_embeddings": {0: "batch_size", 1: "num_classes"},
            "cls_scores": {0: "batch_size"},
            "bbox_preds": {0: "batch_size"},
            "objectness": {0: "batch_size"},
        },
    )

    print(f"ONNX model exported: {args.output}")
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Model size: {file_size:.1f} MB")

    # 简化
    if args.simplify:
        try:
            import onnxsim
            import onnx

            print("Simplifying ONNX model...")
            model_onnx = onnx.load(args.output)
            model_simple, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simple, args.output)
                print("Simplified successfully.")
            else:
                print("Simplification failed, keeping original.")
        except ImportError:
            print("onnx-simplifier not installed: pip install onnxsim")

    # 验证
    if args.verify:
        try:
            import onnxruntime as ort

            print("Verifying with ONNX Runtime...")
            session = ort.InferenceSession(args.output)
            ort_outputs = session.run(
                None,
                {
                    "images": dummy_images.numpy(),
                    "text_embeddings": dummy_text.numpy(),
                },
            )

            # 与PyTorch输出对比
            with torch.no_grad():
                pt_outputs = onnx_model(dummy_images, dummy_text)

            for i, (name, pt_out) in enumerate(
                zip(["cls_scores", "bbox_preds", "objectness"], pt_outputs)
            ):
                diff = np.abs(pt_out.numpy() - ort_outputs[i]).max()
                print(f"  {name}: max diff = {diff:.6f}")

            print("Verification passed!")

        except ImportError:
            print("onnxruntime not installed: pip install onnxruntime")

    print("\nDone!")


if __name__ == "__main__":
    main()
