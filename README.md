# 基于自适应门控跨模态注意力的轻量级开放词汇目标检测

> 硕士毕业设计：基于多模态目标检测的方法研究

## 方法概述

本项目提出 **AGCMA（Adaptive Gated Cross-Modal Attention）** 模块，一种面向CPU部署的轻量级视觉-语言融合机制，用于开放词汇目标检测任务。

### 核心创新

- **AGCMA融合模块**：深度可分离卷积局部增强 + 通道亲和力注意力 + 自适应门控融合
- **AGCMA-PAN**：将AGCMA集成到特征金字塔网络，替代YOLO-World的RepVL-PAN
- **轻量文本编码器蒸馏**：从CLIP ViT-B/32蒸馏4层Transformer文本编码器

### 基线模型

基于 **YOLO-World-S**（CVPR 2024），保留YOLOv8-S视觉backbone和检测头，重点改进跨模态融合机制。

## 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

## 训练

```bash
# 使用AGCMA模型训练
python scripts/train.py --config configs/agcma_yoloworld_s.yaml

# 文本编码器蒸馏
python scripts/train.py --config configs/distill_text_encoder.yaml
```

## 评估

```bash
# COCO开放词汇评估
python scripts/eval.py --config configs/eval_open_vocab.yaml --checkpoint path/to/best.pth

# 推理速度测试
python scripts/eval.py --config configs/eval_open_vocab.yaml --benchmark
```

## 演示

```bash
# 交互式检测演示
python scripts/demo.py --checkpoint path/to/best.pth --image path/to/image.jpg --text "person, car, dog"
```

## 导出ONNX

```bash
python scripts/export_onnx.py --checkpoint path/to/best.pth --output model.onnx
```

## 消融实验

```bash
# 运行全部消融实验
python scripts/ablation.py --config configs/agcma_yoloworld_s.yaml
```

## 项目结构

```
multimodal-det/
├── configs/          # 配置文件
├── data/             # 数据集加载与增强
├── models/           # 模型定义
│   ├── backbone/     # YOLOv8-S视觉backbone
│   ├── text_encoder/ # 文本编码器（CLIP + 轻量蒸馏版）
│   ├── neck/         # 融合网络（AGCMA-PAN + RepVL-PAN）
│   └── head/         # 检测头
├── engine/           # 训练/评估引擎
├── utils/            # 工具函数
├── scripts/          # 入口脚本
└── experiments/      # 实验记录
```

## 参考文献

1. YOLO-World: Real-Time Open-Vocabulary Object Detection (CVPR 2024)
2. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection (ECCV 2024)
3. CLIP-KD: An Empirical Study of CLIP Model Distillation (CVPR 2024)
4. CLIP: Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)
