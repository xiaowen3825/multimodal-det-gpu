# 🚀 完整运行指南 —— AGCMA 多模态目标检测

> 本指南按照**严格的执行顺序**编排，请从第一步开始逐步执行。

---

## 📋 总体运行顺序

```
Step 0: 环境准备（Python虚拟环境 + 依赖安装）
Step 1: 数据集下载与准备（COCO 2017 + CC3M文本子集）
Step 2: 文本编码器蒸馏训练（Teacher: CLIP → Student: LightTextEncoder）
Step 3: 检测模型训练（AGCMA + YOLO-World-S）
Step 4: 模型评估（COCO AP + 推理速度Benchmark）
Step 5: 消融实验（5种变体对比）
Step 6: 演示推理（单图/批量检测）
Step 7: ONNX导出（部署用）
```

---

## Step 0: 环境准备

### 0.1 创建虚拟环境

```bash
# 进入项目目录
cd multimodal-det

# 创建Python虚拟环境（推荐Python 3.10+）
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 0.2 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 验证关键库版本
python -c "
import torch
import torchvision
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CPU threads: {torch.get_num_threads()}')
print(f'BF16 support: {torch.cpu.is_available()}')
"
```

**期望输出**：PyTorch >= 2.1.0，Transformers >= 4.35.0

---

## Step 1: 数据集下载与准备

### 1.1 创建数据目录

```bash
mkdir -p datasets/coco
cd datasets/coco
```

### 1.2 下载 COCO 2017 数据集

```bash
# 下载训练集图片（约18GB，如果网络慢可以考虑只用验证集先跑通）
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 解压
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# 清理zip文件（可选）
rm -f train2017.zip val2017.zip annotations_trainval2017.zip
```

> **⚠️ 国内下载较慢的替代方案**：可以使用以下镜像源
> ```bash
> # 使用学术镜像（如果可用）
> # 或者通过百度网盘/阿里云盘等方式获取COCO 2017数据集
> ```

### 1.3 验证数据集结构

```bash
# 回到项目根目录
cd ../../

# 验证目录结构
ls -la datasets/coco/
# 期望看到:
#   annotations/
#   train2017/
#   val2017/

# 验证标注文件
ls datasets/coco/annotations/
# 期望看到:
#   instances_train2017.json
#   instances_val2017.json
#   ... (其他文件)

# 统计图片数量
echo "训练集图片数: $(ls datasets/coco/train2017/ | wc -l)"
echo "验证集图片数: $(ls datasets/coco/val2017/ | wc -l)"
# 期望: 训练集 ~118287, 验证集 ~5000
```

### 1.4 准备蒸馏用文本数据（CC3M子集）

蒸馏需要一个纯文本文件（每行一条图像描述文本）。你可以：

**方案A：用COCO自带的captions生成（推荐，无需额外下载）**

```bash
python -c "
import json

# 从COCO captions中提取文本
with open('datasets/coco/annotations/captions_train2017.json', 'r') as f:
    data = json.load(f)

texts = [ann['caption'].strip() for ann in data['annotations']]

# 保存为文本文件（取前10万条）
with open('datasets/cc3m_texts.txt', 'w') as f:
    for text in texts[:100000]:
        f.write(text + '\n')

print(f'生成文本数据: {min(len(texts), 100000)} 条')
"
```

**方案B：如果没有captions文件，用COCO类别名生成模拟数据**

```bash
python -c "
import random

# COCO 80个类别
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

templates = [
    'a photo of a {}',
    'a picture of a {}',
    'an image showing a {}',
    'a {} in the scene',
    'a close-up photo of a {}',
    'a bright photo of a {}',
    'a dark photo of a {}',
    'a photo of a small {}',
    'a photo of a large {}',
    'a photo of the {}',
]

random.seed(42)
texts = []
for _ in range(100000):
    cls = random.choice(classes)
    tmpl = random.choice(templates)
    texts.append(tmpl.format(cls))

with open('datasets/cc3m_texts.txt', 'w') as f:
    for t in texts:
        f.write(t + '\n')

print(f'生成模拟文本数据: {len(texts)} 条')
"
```

### 1.5 最终目录结构验证

```bash
# 验证完整目录结构
find datasets/ -maxdepth 2 -type f | head -20
# 期望:
#   datasets/coco/annotations/instances_train2017.json
#   datasets/coco/annotations/instances_val2017.json
#   datasets/coco/train2017/000000000009.jpg  (大量jpg)
#   datasets/coco/val2017/000000000139.jpg    (大量jpg)
#   datasets/cc3m_texts.txt
```

---

## Step 2: 文本编码器蒸馏训练

> **这一步做什么？** 将大型CLIP文本编码器（Teacher, ~63M参数）的知识蒸馏到轻量4层Transformer（Student, ~12M参数），使其在CPU上能快速运行。

### 2.1 运行蒸馏训练

```bash
python scripts/train.py --config configs/distill_text_encoder.yaml
```

**运行参数说明：**
| 参数 | 值 | 说明 |
|------|------|------|
| Teacher | CLIP ViT-B/32 | 自动从HuggingFace下载（首次需联网） |
| Student | 4层Transformer | ~12M参数，512维嵌入 |
| Epochs | 30 | 约3-6小时（CPU） |
| Batch size | 64 | 纯文本数据，batch可以大 |
| 文本数据量 | 100,000条 | 从CC3M/COCO captions提取 |

**训练日志示例：**
```
[INFO] === Text Encoder Distillation Mode ===
[INFO] Student params: 12.3M
[INFO] Epoch [1/30] Step [100/1562] Loss: 0.8234 Feat: 0.6512 Align: 0.3444 LR: 0.000998
...
[INFO] Epoch [30/30] Avg Loss: 0.1245 Feat: 0.0834 Align: 0.0822
[INFO] Best Checkpoint saved: ./runs/distill/best_text_encoder.pth
```

### 2.2 验证蒸馏结果

```bash
# 检查输出权重文件
ls -lh runs/distill/
# 期望看到:
#   best_text_encoder.pth   (~50MB)
#   text_encoder_epoch_5.pth
#   text_encoder_epoch_10.pth
#   ...
```

### 2.3 （可选）跳过蒸馏——直接用CLIP

如果想快速跑通，可以跳过蒸馏，直接修改配置使用CLIP：

```bash
# 修改 configs/agcma_yoloworld_s.yaml 中的 text_encoder 部分:
#   text_encoder:
#     type: "CLIPTextEncoder"       # 改为CLIP
#     model_name: "openai/clip-vit-base-patch32"
#     embed_dim: 512
#     freeze: true
```

> **注意**：直接用CLIP会导致推理更慢（~63M vs ~12M），但可以先验证整体流程。

---

## Step 3: 检测模型训练

> **这一步做什么？** 使用AGCMA融合模块 + YOLOv8-S backbone + 蒸馏后的文本编码器，在COCO数据集上训练多模态目标检测模型。

### 3.1 更新配置（指向蒸馏权重）

在 `configs/agcma_yoloworld_s.yaml` 中指定蒸馏好的文本编码器权重：

```yaml
# 找到 text_encoder 部分，修改 pretrained 字段:
text_encoder:
  type: "LightTextEncoder"
  pretrained: "./runs/distill/best_text_encoder.pth"   # ← 指向Step 2的输出
  ...
```

你可以用以下命令快速修改：

```bash
# macOS/Linux 用 sed 修改
sed -i.bak 's|pretrained: ""|pretrained: "./runs/distill/best_text_encoder.pth"|' configs/agcma_yoloworld_s.yaml
```

### 3.2 开始训练

```bash
python scripts/train.py --config configs/agcma_yoloworld_s.yaml
```

**运行参数说明：**
| 参数 | 值 | 说明 |
|------|------|------|
| Backbone | YOLOv8-S | ~6M参数，冻结前3个stage |
| Text Encoder | LightTextEncoder | ~12M参数，冻结 |
| Neck (AGCMA-PAN) | **核心创新** | ~2M参数，**可训练** |
| Head | DecoupledHead | ~3M参数，**可训练** |
| 训练数据 | COCO 20%子集 | ~23,600张图片（CPU友好） |
| Epochs | 50 | 约24-72小时（CPU i9） |
| 实际batch | 2 × 8累积 = 16 | 梯度累积模拟大batch |
| 图像大小 | 640×640 | 标准YOLO输入 |

**训练日志示例：**
```
[INFO] === Detection Training Mode ===
[INFO] Model built: MultiModalDetector
[INFO] Train: 23657 images
[INFO] Val: 5000 images
[INFO] CPU threads: 16
[INFO] torch.compile enabled (mode=reduce-overhead)
[INFO] Starting training for 50 epochs
[INFO]   Gradient accumulation: 8
[INFO]   Effective batch size: 16
[INFO]   [1][50/11828] loss=12.3456 cls=5.2341 box=6.1234 lr=0.000100
...
[INFO] Epoch [1/50] completed in 3456.2s | Loss: 8.2341 | CLS: 3.4521 | BOX: 4.1234
...
[INFO] Epoch [50/50] completed in 2891.1s | Loss: 1.2345 | CLS: 0.5432 | BOX: 0.6789
[INFO] Training finished. Best AP: 0.1823
```

### 3.3 从断点恢复训练

如果训练中断（电脑重启、进程被杀等），可以恢复：

```bash
# 从最近的checkpoint恢复
python scripts/train.py \
    --config configs/agcma_yoloworld_s.yaml \
    --resume runs/epoch_25.pth
```

### 3.4 查看训练曲线（TensorBoard）

```bash
# 启动TensorBoard
tensorboard --logdir runs/logs --port 6006

# 然后在浏览器中打开: http://localhost:6006
```

### 3.5 训练输出文件

```bash
ls -lh runs/
# 期望看到:
#   best.pth          (~100MB, 最优模型)
#   epoch_5.pth
#   epoch_10.pth
#   ...
#   epoch_50.pth
#   logs/              (TensorBoard日志)
```

---

## Step 4: 模型评估

> **这一步做什么？** 在COCO验证集上计算AP指标，测试推理速度，生成可视化结果。

### 4.1 基础评估（COCO AP指标）

```bash
python scripts/eval.py \
    --config configs/eval_open_vocab.yaml \
    --checkpoint runs/best.pth
```

**输出示例：**
```
Parameters: 23.45M (trainable: 5.12M)

╔══════════════════════════════════╗
║    COCO Evaluation Results       ║
╠══════════════════════════════════╣
║  AP      : 0.1823               ║
║  AP50    : 0.3245               ║
║  AP75    : 0.1567               ║
║  AP_base : 0.2134               ║
║  AP_novel: 0.0891               ║
║  AR100   : 0.2567               ║
╚══════════════════════════════════╝
```

### 4.2 完整评估 + 推理速度 Benchmark

```bash
python scripts/eval.py \
    --config configs/eval_open_vocab.yaml \
    --checkpoint runs/best.pth \
    --benchmark
```

**额外输出：**
```
Running inference benchmark...
╔══════════════════════════════════╗
║    Inference Benchmark           ║
╠══════════════════════════════════╣
║  Latency(ms): 245.3             ║
║  FPS        : 4.08              ║
║  Params(M)  : 23.45             ║
║  GFLOPs     : 12.67             ║
╚══════════════════════════════════╝
```

### 4.3 评估 + 可视化

```bash
python scripts/eval.py \
    --config configs/eval_open_vocab.yaml \
    --checkpoint runs/best.pth \
    --benchmark \
    --vis \
    --vis_dir runs/visualizations
```

```bash
# 查看可视化结果
ls runs/visualizations/
# vis_0000.jpg  vis_0001.jpg  ...  vis_0049.jpg
```

---

## Step 5: 消融实验

> **这一步做什么？** 对比5种模型变体，验证AGCMA每个组件的贡献，这是论文中消融实验表格的数据来源。

### 5.1 仅对比参数量（快速，无需训练）

```bash
python scripts/ablation.py \
    --config configs/agcma_yoloworld_s.yaml \
    --eval_only
```

**输出示例：**
```
============================================================
Running ablation study with variants: ['full_agcma', 'no_gate', 'no_dwconv', 'spatial_attn', 'repvl_pan']
============================================================

Variant              Description                          Params(M)  Trainable(M)
----------------------------------------------------------------------------------
full_agcma           完整AGCMA模型                          23.45        5.12
no_gate              去掉自适应门控 (直接相加)                 23.41        5.08
no_dwconv            去掉深度可分离卷积局部增强                 23.39        5.06
spatial_attn         标准空间Cross-Attention (对比)           24.12        5.79
repvl_pan            RepVL-PAN基线 (YOLO-World)              23.21        4.88

Results saved to ./runs/ablation/ablation_results.json
```

### 5.2 完整消融实验（需要训练每个变体）

```bash
# 训练全部5个变体（耗时较长，约5×50 epochs）
python scripts/ablation.py \
    --config configs/agcma_yoloworld_s.yaml \
    --epochs 50

# 或者只选择部分变体
python scripts/ablation.py \
    --config configs/agcma_yoloworld_s.yaml \
    --variants full_agcma no_gate repvl_pan \
    --epochs 30
```

### 5.3 消融实验论文表格格式

运行完毕后，`runs/ablation/ablation_results.json` 中的数据可以直接填入论文表格：

| 方法 | 说明 | Params(M) | AP | AP50 | AP_novel |
|------|------|-----------|-----|------|----------|
| Full AGCMA | 完整模型（Ours） | 23.45 | — | — | — |
| w/o Gate | 去掉自适应门控 | 23.41 | — | — | — |
| w/o DWConv | 去掉局部增强 | 23.39 | — | — | — |
| Spatial Attn | 标准Cross-Attention | 24.12 | — | — | — |
| RepVL-PAN | YOLO-World基线 | 23.21 | — | — | — |

> AP数值需要训练后填入。

---

## Step 6: 演示推理

> **这一步做什么？** 用训练好的模型对新图片进行检测，支持自定义文本prompt。

### 6.1 单张图片检测

```bash
python scripts/demo.py \
    --checkpoint runs/best.pth \
    --image test.jpg \
    --text "person, car, dog, cat" \
    --score_thr 0.3
```

**输出：**
```
Detecting classes: ['person', 'car', 'dog', 'cat']
Processing 1 images...
  test.jpg: 5 detections, 267.3ms -> ./runs/demo/test_det.jpg
Done! Results saved to ./runs/demo
```

### 6.2 批量图片检测

```bash
# 准备一个测试图片文件夹
mkdir -p test_images
# 放入若干jpg/png图片...

python scripts/demo.py \
    --checkpoint runs/best.pth \
    --image_dir test_images \
    --text "person, bicycle, car, bus, truck" \
    --score_thr 0.25 \
    --output_dir runs/demo_batch
```

### 6.3 开放词汇检测（任意文本描述）

```bash
# 这就是开放词汇检测的魅力——你可以输入训练集中没有的类别！
python scripts/demo.py \
    --checkpoint runs/best.pth \
    --image test.jpg \
    --text "red sports car, golden retriever, child wearing hat" \
    --score_thr 0.2
```

---

## Step 7: ONNX 导出（部署用）

> **这一步做什么？** 将PyTorch模型导出为ONNX格式，用于C++/ONNX Runtime部署，进一步加速CPU推理。

### 7.1 基础导出

```bash
python scripts/export_onnx.py \
    --checkpoint runs/best.pth \
    --output runs/model.onnx
```

**输出：**
```
Loading model from runs/best.pth...
Exporting to runs/model.onnx...
ONNX model exported: runs/model.onnx
Model size: 45.2 MB
```

### 7.2 导出 + 简化 + 验证

```bash
# 需要额外安装: pip install onnxsim
python scripts/export_onnx.py \
    --checkpoint runs/best.pth \
    --output runs/model.onnx \
    --simplify \
    --verify
```

**输出：**
```
Loading model from runs/best.pth...
Exporting to runs/model.onnx...
ONNX model exported: runs/model.onnx
Model size: 45.2 MB
Simplifying ONNX model...
Simplified successfully.
Verifying with ONNX Runtime...
  cls_scores: max diff = 0.000012
  bbox_preds: max diff = 0.000008
  objectness: max diff = 0.000003
Verification passed!
Done!
```

---

## 📊 完整文件输出一览

训练完成后，你的 `runs/` 目录应该长这样：

```
runs/
├── best.pth                       # 最优检测模型权重
├── epoch_5.pth                    # 定期checkpoint
├── epoch_10.pth
├── ...
├── epoch_50.pth
├── eval_results.json              # 评估结果（AP等指标）
├── model.onnx                     # ONNX导出模型
│
├── distill/                       # 蒸馏相关
│   ├── best_text_encoder.pth      # 最优蒸馏文本编码器
│   ├── text_encoder_epoch_5.pth
│   └── ...
│
├── ablation/                      # 消融实验
│   └── ablation_results.json      # 消融实验结果汇总
│
├── logs/                          # TensorBoard日志
│   └── agcma_yoloworld_s/
│       └── events.out.tfevents...
│
├── demo/                          # 演示输出
│   ├── test_det.jpg
│   └── ...
│
└── visualizations/                # 评估可视化
    ├── vis_0000.jpg
    └── ...
```

---

## ⏱️ 预估时间表（i9 CPU, 32GB内存）

| 步骤 | 预估时间 | 备注 |
|------|---------|------|
| Step 0: 环境安装 | 10-20分钟 | 取决于网络速度 |
| Step 1: 数据下载 | 1-3小时 | COCO约20GB，取决于网速 |
| Step 2: 文本蒸馏 | 3-6小时 | 30 epochs, 纯文本，较快 |
| Step 3: 检测训练 | 24-72小时 | 50 epochs, 20%子集 |
| Step 4: 评估 | 30-60分钟 | 5000张验证图 |
| Step 5: 消融(仅参数) | 1-2分钟 | 不训练，仅统计参数 |
| Step 5: 消融(完整) | 5×24-72小时 | 5个变体各训练一遍 |
| Step 6: 演示 | 几秒-几分钟 | 取决于图片数量 |
| Step 7: ONNX导出 | 1-2分钟 | 快速 |

> **快速跑通建议**：先用 `--epochs 5` 跑通全流程，确认无误后再跑完整50 epochs。

---

## 🛠️ 常见问题 & 故障排除

### Q1: 首次运行CLIP下载超时？
```bash
# 设置HuggingFace镜像（国内推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型后指定本地路径
# 在 configs/distill_text_encoder.yaml 中修改:
#   model_name: "/path/to/local/clip-vit-base-patch32"
```

### Q2: 内存不足 (OOM)？
```bash
# 减小batch_size
# 在 configs/agcma_yoloworld_s.yaml 中修改:
#   batch_size: 1              # 从2改为1
#   gradient_accumulation: 16  # 同步增大累积步数保持等效batch

# 减小图像尺寸
#   img_size: 416              # 从640改为416
```

### Q3: 训练太慢？
```bash
# 方案1: 减少数据量
# 在 configs/default.yaml 中修改:
#   subset_ratio: 0.1          # 从0.2改为0.1（只用10%数据）

# 方案2: 减少epochs
python scripts/train.py --config configs/agcma_yoloworld_s.yaml
# 在配置中修改 epochs: 30  # 从50改为30

# 方案3: 用更小的输入
# 在 configs/default.yaml 中修改:
#   img_size: 416
```

### Q4: 想先快速验证整个流程？
```bash
# 使用极小配置跑通全流程（约30分钟）
# 临时修改 configs/default.yaml:
#   subset_ratio: 0.02   # 只用2%数据
#   img_size: 320        # 小图
#   epochs: 3            # 3个epoch

# 然后依次跑 Step 2-7
```

---

## 📝 论文写作时需要的数据

完成以上所有步骤后，你的论文需要的关键数据：

1. **方法描述**：AGCMA模块的三个组件 → 见 `models/neck/agcma_module.py`
2. **参数量对比**：消融实验 → `runs/ablation/ablation_results.json`
3. **AP指标**：COCO评估 → `runs/eval_results.json`
4. **推理速度**：Benchmark → eval.py 的 `--benchmark` 输出
5. **可视化图**：检测结果 → `runs/visualizations/` 和 `runs/demo/`
6. **训练曲线**：TensorBoard → `runs/logs/`（截图）
7. **门控权重热力图**：体现自适应融合 → 见 `utils/visualize.py` 中的 `visualize_gate_weights`

祝毕设顺利！🎓
