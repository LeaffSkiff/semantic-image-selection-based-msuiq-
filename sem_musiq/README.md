# Sem-MUSIQ

精简版 MUSIQ（Multi-scale Image Quality Transformer），支持语义嵌入功能。

## 安装依赖

```bash
pip install torch torchvision huggingface_hub tqdm
```

## 快速开始

### 基本使用（不使用语义嵌入）

```python
import torch
from PIL import Image
import sem_musiq

# 加载图像
img = Image.open('image.jpg').convert('RGB')
img_tensor = transforms.ToTensor()(img).unsqueeze(0)

# 创建模型
model = sem_musiq.MUSIQ(pretrained='koniq10k')
model.eval()

# 推理
with torch.no_grad():
    score = model(img_tensor)
print(f"MUSIQ 分数：{score.item():.4f}")
```

### 使用语义嵌入

```python
import torch
from PIL import Image
import sem_musiq

# 加载图像
img = Image.open('image.jpg').convert('RGB')
img_tensor = transforms.ToTensor()(img).unsqueeze(0)

# 创建模型（启用语义嵌入）
model = sem_musiq.MUSIQ(
    pretrained='koniq10k',
    use_semantic=True,
    semantic_input_dim=5,  # SAM mask 数量
)
model.eval()

# 准备语义向量（从 SAM 计算的 patch-mask 重叠率）
# semantic_vectors 形状：[batch_size, num_patches, k]
semantic_vectors = ...  # 你的语义向量

# 推理
with torch.no_grad():
    score = model(img_tensor, semantic_vectors=semantic_vectors)
print(f"MUSIQ 分数：{score.item():.4f}")
```

## 预训练模型

| 模型名称 | 数据集 | 分数范围 |
|----------|--------|----------|
| `koniq10k` | KonIQ-10k | 0-100 |
| `ava` | AVA | 1-10 |
| `spaq` | SPAQ | 0-100 |
| `paq2piq` | PAQ2PIQ | 0-100 |

## 目录结构

```
sem-musiq/
├── __init__.py
├── archs/
│   ├── __init__.py
│   ├── arch_util.py      # 工具函数
│   └── musiq_arch.py     # MUSIQ 模型
├── data/
│   ├── __init__.py
│   └── multiscale_trans_util.py  # 多尺度处理
├── matlab_utils/
│   ├── __init__.py
│   └── padding.py        # 填充工具
├── utils/
│   ├── __init__.py
│   ├── download_util.py  # 下载工具
│   └── misc.py           # 杂项工具
└── example.py            # 使用示例
```

## 许可证

基于 IQA-PyTorch 精简修改。
