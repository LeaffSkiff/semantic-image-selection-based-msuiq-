# 基于 MUSIQ 的语义图像选择系统 (Semantic Image Selection based on MUSIQ)

## 项目简介

本项目基于 MUSIQ（Multi-scale Image Quality Transformer）实现语义图像选择系统，通过引入 SAM（Segment Anything）生成的语义信息，增强图像质量评估的语义感知能力。

## 项目结构

```
semantic-image-selection-based-msuiq/
├── IQA-PyTorch/                 # PyTorch 图像质量评估工具箱
│   └── pyiqa/
│       └── archs/
│           └── musiq_arch.py    # 修改：添加语义嵌入功能
├── datasets/
│   └── KonIQ-10k/               # KonIQ-10k 数据集
├── tutorials/                   # 学习教程（分阶段示例）
│   ├── 01_patch_visualization/
│   ├── 02_sam_generation/
│   ├── 03_overlap_computation/
│   ├── 04_semantic_embedding/
│   ├── 05_semantic_fusion/
│   └── 06_cross_scale_consistency/
├── docs/                        # 文档
│   ├── 改进方案.md
│   └── 改进方案 v2.md
└── README.md
```

## 核心功能

- **MUSIQ 模型**：多尺度图像质量评估 Transformer
- **SAM 集成**：Segment Anything 生成语义 mask
- **语义融合**：将语义信息融入 MUSIQ 特征
- **跨尺度一致性**：计算不同尺度间的语义一致性分数

## 快速开始

```python
import pyiqa

# 创建 MUSIQ 模型
model = pyiqa.create_metric("musiq", device="cuda")

# 评估图像质量（输出 0-100 分数）
score = model(image_path)
```

---

# IQA-PyTorch 原始文档
