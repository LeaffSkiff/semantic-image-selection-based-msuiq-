# 基于 MUSIQ 的语义图像选择系统 (Semantic Image Selection based on MUSIQ)

## 项目简介

本项目实现了一个**双分支架构**的图像质量评估系统：
- **语义分支**：SAM 特征提取 → Transformer 编码 → 语义一致性分数 (SIM)
- **MUSIQ 分支**：多尺度 patch 编码 → Transformer 主干 → 质量分数 (Q)
- **融合模块**：可学习权重融合 `F = λ_q × Q + λ_sim × (SIM × 100)`

### 核心公式

| 分数 | 范围 | 说明 |
|------|------|------|
| **Q（质量分数）** | 0-100 | MUSIQ 预测的图像质量 |
| **SIM（语义一致性）** | 0-1 | 同尺度内 patch 的余弦相似度 |
| **F（融合分数）** | 0-100 | `F = λ_q × Q + λ_sim × (SIM × 100)` |

其中 `λ_q + λ_sim = 1`，权重可通过学习优化。

---

## 系统架构

```
输入图像
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
┌─────────────────────┐              ┌─────────────────────┐
│   语义分支          │              │   MUSIQ 分支        │
│                     │              │                     │
│ 1. SAM 特征提取     │              │ 1. 多尺度 patch 编码 │
│    - 3 尺度缩放      │              │    - 原图 + 2 缩放    │
│    - ViT-B embedding│              │    - patch_size=32   │
│    - [N, 256]       │              │    - [N, 384]        │
│                     │              │                     │
│ 2. Transformer 编码 │              │ 2. 14 层 Transformer │
│    - 6 层 Encoder    │              │    - 输出质量分布   │
│    - [N, 256]→[N,384]│             │    - MOS: 0-100     │
│                     │              │                     │
│ 3. 一致性打分       │              │                     │
│    - 余弦相似度      │              │                     │
│    - SIM: 0-1       │              │                     │
└─────────────────────┘              └─────────────────────┘
    │                                          │
    └──────────────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  QualityScoreFusion │
              │  F = λ·Q + (1-λ)·SIM│
              └─────────────────┘
                       │
                       ▼
                  最终分数 F
```

---

## 核心模块

### 1. SAM 特征提取器 (`sem_musiq/semantic/sam_feature_extractor.py`)

**功能**：从 SAM ViT-B 提取多尺度 patch 的语义 embedding

**多尺度配置**：
| 尺度 | patch_size | 图像长边 | 输出 | 用途 |
|------|------------|----------|------|------|
| 细粒度 | 16×16 | 224 | [N₁, 256] | 局部细节 |
| 中等 | 32×32 | 384 | [N₂, 256] | 区域语义 |
| 粗粒度 | 64×64 | 512 | [N₃, 256] | 全局上下文 |

**使用示例**：
```python
from sem_musiq import SAMFeatureExtractor

extractor = SAMFeatureExtractor(
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
)
result = extractor(image_path)
sam_embeddings = result['sam_embeddings']  # List of [N_s, 256]
```

---

### 2. 语义 Transformer 编码器 (`sem_musiq/transformer/semantic_transformer.py`)

**功能**：将 SAM embedding 投影到 Transformer 特征空间

**架构**：
- 输入：`List[[N_s, 256]]` - 各尺度 SAM embedding
- 输出：`List[[N_s, 384]]` - 编码后的语义特征
- 6 层 Transformer Encoder
- 每个尺度独立权重（不共享）

**位置编码**：
- Hash-based 空间位置编码（与 MUSIQ 设计一致）
- 尺度嵌入（scale embedding）

**使用示例**：
```python
from sem_musiq import SemanticTransformerEncoder

encoder = SemanticTransformerEncoder(
    input_dim=256,      # SAM ViT-B 输出维度
    output_dim=384,     # 与 MUSIQ hidden_size 对齐
    num_layers=6,
    num_scales=3,
)
semantic_embeds = encoder(sam_embeddings)
```

---

### 3. 语义一致性打分器 (`sem_musiq/semantic/consistency_scorer.py`)

**功能**：计算同尺度内 patch 之间的余弦相似度

**原理**：
- 同一尺度内，语义相似的 patch 应有较高相似度
- 计算所有 patch 对的余弦相似度，取平均
- 各尺度平均分聚合为最终 SIM 分数

**公式**：
```python
# L2 归一化
embeddings_norm = F.normalize(embeddings, p=2, dim=1)
# 余弦相似度矩阵
similarity_matrix = embeddings_norm @ embeddings_norm.t()
# 平均相似度（排除对角线）
sim_score = (similarity_matrix.sum() - N) / (N * (N-1))
```

**使用示例**：
```python
from sem_musiq import SemanticConsistencyScorer

scorer = SemanticConsistencyScorer(num_scales=3)
sim_score = scorer(semantic_embeds)  # float in [0, 1]
```

---

### 4. MUSIQ 主干 (`sem_musiq/archs/musiq_arch.py`)

**功能**：预测图像质量分数（0-100）

**架构**：
- 多尺度 patch 编码（3 尺度）
- 14 层 Transformer Encoder
- 输出：质量分布 → MOS 分数

**预训练模型**：
| 模型 | 数据集 | 分数范围 |
|------|--------|----------|
| `koniq10k` | KonIQ-10k | 0-100 |
| `ava` | AVA | 1-10 |
| `spaq` | SPAQ | 0-100 |

**使用示例**：
```python
from sem_musiq import MUSIQ

model = MUSIQ(pretrained='koniq10k')
quality_score = model(img_tensor)  # tensor, 0-100
```

---

### 5. 可学习融合模块 (`sem_musiq/fusion.py`)

**QualityScoreFusion 类**：

```python
class QualityScoreFusion(nn.Module):
    def __init__(self, lambda_q_init=0.7, lambda_sim_init=0.3):
        self.lambda_q_raw = nn.Parameter(torch.tensor(lambda_q_init))
        self.lambda_sim_raw = nn.Parameter(torch.tensor(lambda_sim_init))
    
    def forward(self, quality_score, sim_score):
        weights = torch.softmax(torch.stack([self.lambda_q_raw, self.lambda_sim_raw]), dim=0)
        return weights[0] * quality_score + weights[1] * (sim_score * 100)
```

**特点**：
- 权重通过 `nn.Parameter` 可学习
- `softmax` 约束 `λ_q + λ_sim = 1`
- 可调用 `train_fusion_only()` 训练

---

## 项目结构

```
semantic-image-selection-based-msuiq/
├── sem_musiq/                     # 核心模块
│   ├── __init__.py                # 导出接口
│   ├── archs/
│   │   ├── __init__.py
│   │   ├── arch_util.py           # 工具函数
│   │   └── musiq_arch.py          # MUSIQ 模型
│   ├── data/
│   │   ├── __init__.py
│   │   └── multiscale_trans_util.py  # 多尺度 patch 提取
│   ├── transformer/
│   │   ├── __init__.py
│   │   └── semantic_transformer.py   # 语义 Transformer 编码
│   ├── semantic/
│   │   ├── __init__.py
│   │   ├── sam_feature_extractor.py  # SAM 特征提取
│   │   └── consistency_scorer.py     # 语义一致性打分
│   ├── fusion.py                  # 融合模块
│   └── USAGE.md                   # 详细使用文档
├── checkpoints/
│   └── sam_vit_b_01ec64.pth       # SAM 模型权重
├── images/
│   └── example.png                # 示例图像
├── train_semantic_fusion.py       # 训练脚本
├── test_semantic_pipeline.py      # 测试脚本
└── README.md                      # 本文档
```

---

## 快速开始

### 安装依赖

```bash
# 基础依赖
pip install torch torchvision pillow huggingface_hub tqdm

# SAM 依赖（必需）
pip install segment-anything
```

### 下载 SAM 权重

```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 最简单的用法（端到端）

```python
from sem_musiq import SemMUSIQFusion

# 初始化模型
model = SemMUSIQFusion(
    musiq_pretrained='koniq10k',
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
    semantic_transformer_layers=6,
    lambda_q_init=0.7,
)

# 预测
result = model.predict('your_image.jpg')
print(f"质量分数 Q   = {result['quality_score']:.4f}")
print(f"语义一致性 SIM = {result['sim_score']:.4f}")
print(f"融合分数 F   = {result['final_score']:.4f}")
```

### 分离调用（更灵活）

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
from sem_musiq import (
    SAMFeatureExtractor,
    SemanticTransformerEncoder,
    SemanticConsistencyScorer,
    MUSIQ,
    QualityScoreFusion,
)

# 加载图像
img = Image.open('image.jpg').convert('RGB')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. SAM 特征提取
sam_extractor = SAMFeatureExtractor(
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
    device=device,
)
sam_result = sam_extractor(img, return_details=False)
sam_embeddings = sam_result['sam_embeddings']  # List[np.ndarray]

# 2. 语义编码
semantic_encoder = SemanticTransformerEncoder(
    input_dim=256,
    output_dim=384,
    num_layers=6,
    num_scales=3,
)
semantic_encoder.to(device)
semantic_encoder.eval()

sam_embeddings_tensor = [
    torch.from_numpy(emb).float().to(device)
    for emb in sam_embeddings
]
with torch.no_grad():
    semantic_embeds = semantic_encoder(sam_embeddings_tensor)

# 3. 一致性打分
consistency_scorer = SemanticConsistencyScorer(num_scales=3)
consistency_scorer.to(device)
consistency_scorer.eval()

with torch.no_grad():
    sim_score = consistency_scorer(semantic_embeds)

# 4. MUSIQ 质量预测
musiq = MUSIQ(pretrained='koniq10k')
musiq.to(device)
musiq.eval()

img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
with torch.no_grad():
    quality_score = musiq(img_tensor)

# 5. 融合分数
fusion = QualityScoreFusion(lambda_q_init=0.7)
fusion.to(device)
sim_tensor = torch.tensor(sim_score).float().to(device)
with torch.no_grad():
    final_score = fusion(quality_score, sim_tensor)

print(f"Q = {quality_score.item():.4f}")
print(f"SIM = {sim_score:.4f}")
print(f"F = {final_score.item():.4f}")
```

---

## 训练融合权重

如果有带标注的数据集（图像 + ground truth MOS），可以训练融合权重：

```python
from torch.utils.data import Dataset, DataLoader
from sem_musiq import SemMUSIQFusion

# 1. 准备数据集
class MyDataset(Dataset):
    def __init__(self, image_paths, mos_scores):
        self.paths = image_paths
        self.mos = mos_scores
    
    def __getitem__(self, idx):
        return {'image': self.paths[idx], 'mos': torch.tensor([self.mos[idx]])}
    
    def __len__(self):
        return len(self.paths)

dataset = MyDataset(['img1.jpg', 'img2.jpg'], [75.0, 82.0])
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 2. 初始化模型
model = SemMUSIQFusion(
    musiq_pretrained='koniq10k',
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
)

# 3. 训练（冻结 MUSIQ 和 SAM，仅优化融合权重）
model.train_fusion_only(
    dataloader=dataloader,
    criterion=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam(model.fusion.parameters(), lr=0.01),
    num_epochs=10,
)

# 4. 查看训练后的权重
lambda_q, lambda_sim = model.get_current_weights()
print(f"训练后权重：Q={lambda_q:.4f}, SIM={lambda_sim:.4f}")
```

---

## 运行测试

```bash
# 使用 conda 环境（推荐）
C:/Users/28027/miniconda3/envs/iqa_env/python.exe test_semantic_pipeline.py

# 或直接运行（需要自行安装依赖）
python test_semantic_pipeline.py
```

测试脚本会验证：
1. SAM 特征提取
2. 语义 Transformer 编码
3. 一致性打分
4. MUSIQ 质量预测
5. 融合分数计算

---

## 与原始 MUSIQ 的对比

| 特性 | 原始 MUSIQ | 本系统 |
|------|------------|--------|
| 语义感知 | ❌ | ✅ SAM 特征 + Transformer 编码 |
| 一致性评估 | ❌ | ✅ 同尺度 patch 余弦相似度 |
| 可学习融合 | ❌ | ✅ 端到端优化权重 |
| 双分支架构 | ❌ | ✅ 独立语义分支 + MUSIQ 主干 |

---

## 许可证

基于 IQA-PyTorch 精简修改，保留原许可证。

## 致谢

- **MUSIQ 原作者**：Junjie Ke et al.
- **IQA-PyTorch 实现**：Chaofeng Chen
- **SAM**：Meta AI Research
