# 基于 MUSIQ 的语义图像评估系统 (Semantic Image Selection based on MUSIQ)

## 项目简介

本项目基于 MUSIQ（Multi-scale Image Quality Transformer）实现语义图像选择系统，通过引入 SAM（Segment Anything）生成的语义信息，增强图像质量评估的语义感知能力。

### 核心公式

最终分数由三部分组成：
- **Q（质量分数）**：MUSIQ 预测的图像质量得分 (0-100)
- **SIM（语义一致性）**：跨尺度语义相似度 (0-1)
- **F（融合分数）**：`F = λ_q × Q + λ_sim × (SIM × 100)`

其中 `λ_q + λ_sim = 1`，权重可通过学习优化。

---

## 主要修改与实现

### 1. MUSIQ 模型修改 (`sem_musiq/archs/musiq_arch.py`)

**新增语义嵌入层 `AddSemanticEmbs`：**
```python
class AddSemanticEmbs(nn.Module):
    """将语义向量投影到 Transformer 特征空间"""
    def __init__(self, semantic_input_dim, dim):
        self.semantic_proj = nn.Parameter(torch.randn(dim, semantic_input_dim))
    
    def forward(self, inputs, semantic_vectors):
        e_sem = F.linear(semantic_vectors, self.semantic_proj)
        return inputs + e_sem
```

**修改 `TransformerEncoder`：**
- 添加 `use_semantic` 和 `semantic_input_dim` 参数
- 在特征输入后、Transformer 编码前叠加语义嵌入

**修改 `MUSIQ.forward()`：**
- 新增 `semantic_vectors` 参数，形状 `[B, N, K]`
- 将语义向量传递给 `TransformerEncoder`

---

### 2. 语义向量生成器 (`sem_musiq/semantic/vector_generator.py`)

**核心功能：**
1. 使用 SAM 生成 top-k 个语义 mask
2. 计算每个 patch 与 mask 的重叠率（overlap ratio）
3. 生成语义向量 `s_i = [o_1, o_2, ..., o_k]`
4. 计算跨尺度语义一致性分数（SIM）

**SIM 计算：**
```python
# 余弦相似度计算跨尺度语义一致性
cos_sim = nn.CosineSimilarity(dim=-1)
sim_score = cos_sim(v1, v2).mean().item()  # [0, 1]
```

**Padding 处理：**
- 当 SAM 返回的 mask 数量少于 `top_k` 时，自动 padding 零向量
- 与 MUSIQ 的 patch padding 逻辑保持一致

---

### 3. 可学习融合模块 (`sem_musiq/fusion.py`)

**`QualityScoreFusion` 类：**
```python
class QualityScoreFusion(nn.Module):
    """可学习的质量分数融合"""
    def __init__(self, lambda_q_init=0.7, lambda_sim_init=0.3):
        self.lambda_q_raw = nn.Parameter(torch.tensor(lambda_q_init))
        self.lambda_sim_raw = nn.Parameter(torch.tensor(lambda_sim_init))
    
    def forward(self, quality_score, sim_score):
        weights = torch.softmax(torch.stack([self.lambda_q_raw, self.lambda_sim_raw]), dim=0)
        return weights[0] * quality_score + weights[1] * (sim_score * 100)
```

**特点：**
- 权重通过 `nn.Parameter` 声明为可学习参数
- 使用 `softmax` 约束权重和为 1
- 可通过 `train_fusion_only()` 方法训练

---

### 4. 端到端融合模型 (`sem_musiq/fusion.py`)

**`SemMUSIQFusion` 类整合了：**
- `SemanticVectorGenerator`：语义向量生成
- `MUSIQ`：质量分数预测
- `QualityScoreFusion`：可学习权重融合

**一行代码完成全部流程：**
```python
model = sem_musiq.SemMUSIQFusion(
    musiq_pretrained='koniq10k',
    sam_checkpoint='checkpoints/sam_vit_b.pth',
    top_k=5,
)
result = model.predict('image.jpg')
# result = {quality_score, sim_score, final_score, weights}
```

---

## 项目结构

```
semantic-image-selection-based-msuiq/
├── sem_musiq/                     # 精简版 MUSIQ 模块（核心）
│   ├── __init__.py                # 导出接口
│   ├── archs/
│   │   ├── __init__.py
│   │   ├── arch_util.py           # 工具函数（dist_to_mos, 加载预训练）
│   │   └── musiq_arch.py          # MUSIQ 模型（新增语义嵌入）
│   ├── data/
│   │   ├── __init__.py
│   │   └── multiscale_trans_util.py  # 多尺度 patch 提取
│   ├── matlab_utils/
│   │   ├── __init__.py
│   │   └── padding.py             # 精确 padding（兼容 MATLAB）
│   ├── semantic/
│   │   ├── __init__.py
│   │   └── vector_generator.py    # 语义向量生成器（SAM + SIM）
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── download_util.py       # 下载工具
│   │   └── misc.py                # 杂项工具
│   ├── fusion.py                  # 可学习融合模块（新增）
│   ├── USAGE.md                   # 详细使用文档
│   └── example.py                 # 使用示例
├── checkpoints/
│   └── sam_vit_b_01ec64.pth       # SAM 模型权重
├── datasets/KonIQ-10k/            # KonIQ-10k 数据集
├── example_usage.py               # 模块外调用示例
└── README.md                      # 本文档
```

---

## 与原始 MUSIQ 的对比

| 特性 | 原始 MUSIQ | 本项目的 Sem-MUSIQ |
|------|------------|---------------------|
| 代码量 | 1500+ 行（完整 IQA-PyTorch） | 9 个核心文件 |
| 语义嵌入 | ❌ | ✅ `AddSemanticEmbs` 层 |
| 语义向量生成 | ❌ | ✅ SAM + Patch-Mask 重叠率 |
| 跨尺度一致性 | ❌ | ✅ SIM 分数（余弦相似度） |
| 可学习融合 | ❌ | ✅ `QualityScoreFusion` |
| 端到端调用 | ❌ | ✅ `SemMUSIQFusion` |

---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision pillow huggingface_hub tqdm
pip install segment-anything  # 可选，用于语义功能
```

### 下载 SAM 权重

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 最简单的用法

```python
import sem_musiq

# 初始化模型
model = sem_musiq.SemMUSIQFusion(
    musiq_pretrained='koniq10k',
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
    top_k=5,
)

# 预测
result = model.predict('your_image.jpg')
print(f"质量分数 Q = {result['quality_score']:.4f}")
print(f"语义一致性 SIM = {result['sim_score']:.4f}")
print(f"融合分数 F = {result['final_score']:.4f}")
```

### 分离调用（更灵活）

```python
import torch
import sem_musiq
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# 1. 生成语义向量
sem_gen = sem_musiq.SemanticVectorGenerator(
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
    top_k=5,
)

img = Image.open('image.jpg').convert('RGB')
sem_result = sem_gen(img, return_details=False)
semantic_vectors = sem_result['semantic_vectors']
sim_score = sem_result['consistency_score']

# 2. MUSIQ 预测
musiq = sem_musiq.MUSIQ(
    pretrained='koniq10k',
    use_semantic=True,
    semantic_input_dim=5,
)

img_tensor = transforms.ToTensor()(img).unsqueeze(0)
all_semantic_vectors = np.concatenate(semantic_vectors, axis=0)
semantic_tensor = torch.from_numpy(all_semantic_vectors).float().unsqueeze(0)

with torch.no_grad():
    quality_score = musiq(img_tensor, semantic_vectors=semantic_tensor)

# 3. 融合分数
fusion = sem_musiq.QualityScoreFusion(lambda_q_init=0.7, lambda_sim_init=0.3)
sim_tensor = torch.tensor(sim_score).float()
final_score = fusion(quality_score, sim_tensor)

print(f"Q = {quality_score.item():.4f}, SIM = {sim_score:.4f}, F = {final_score.item():.4f}")
```

---

## 训练融合权重

如果有带标注的数据集（图像 + ground truth MOS），可以训练融合权重：

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, image_paths, mos_scores):
        self.paths = image_paths
        self.mos = mos_scores
    
    def __getitem__(self, idx):
        return {'image': self.paths[idx], 'mos': torch.tensor([self.mos[idx]])}
    
    def __len__(self):
        return len(self.paths)

# 准备数据
dataset = MyDataset(['img1.jpg', 'img2.jpg'], [75.0, 82.0])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 初始化模型
model = sem_musiq.SemMUSIQFusion(
    musiq_pretrained='koniq10k',
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
    top_k=5,
)

# 训练（冻结 MUSIQ 和 SAM，仅优化融合权重）
model.train_fusion_only(
    dataloader=dataloader,
    criterion=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam(model.fusion.parameters(), lr=0.01),
    num_epochs=10,
)

# 查看训练后的权重
lambda_q, lambda_sim = model.get_current_weights()
print(f"训练后权重：Q={lambda_q:.4f}, SIM={lambda_sim:.4f}")
```

---

## 许可证

基于 IQA-PyTorch 精简修改，保留原许可证。
