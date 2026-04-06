# 基于 MUSIQ 的语义图像选择系统 (Semantic Image Selection based on MUSIQ)

## 项目简介

本项目基于 MUSIQ（Multi-scale Image Quality Transformer）实现语义图像选择系统，通过引入 SAM（Segment Anything）生成的语义信息，增强图像质量评估的语义感知能力。

## 项目结构

```
semantic-image-selection-based-msuiq/
├── IQA-PyTorch/                 # pyiqa，仅保留了 musiq
│   └── pyiqa/
│       └── archs/
│           └── musiq_arch.py    # 添加语义嵌入功能
├── semantic_musiq/              # 语义向量提取器
└── README.md
```

## 核心功能

- **MUSIQ 模型**：多尺度图像质量评估 Transformer
- **SAM 集成**：Segment Anything 生成语义 mask
- **语义融合**：将语义信息融入 MUSIQ 特征
- **跨尺度一致性**：计算不同尺度间的语义一致性分数

## 快速开始

```bash
# 开始试用前，请现在 IQA-PyTorch 文件夹下执行命令以安装修改后的 pyiqa
pip install -e .
```

```python
import pyiqa

# 创建 MUSIQ 模型
model = pyiqa.create_metric("musiq", device="cuda")

# 评估图像质量（输出 0-100 分数）
score = model(image_path)
```
