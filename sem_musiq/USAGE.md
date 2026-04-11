# Sem-MUSIQ 使用文档

## 目录

- [简介](#简介)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细用法](#详细用法)
- [API 参考](#api 参考)
- [常见问题](#常见问题)

---

## 简介

**Sem-MUSIQ** 是一个精简版的 MUSIQ（Multi-scale Image Quality Transformer）图像质量评估模块，支持语义嵌入功能。

### 核心特点

- **轻量精简**：仅保留 MUSIQ 核心推理代码，9 个文件即可运行
- **语义增强**：支持传入 SAM 语义向量，增强图像质量评估的语义感知能力
- **多种预训练模型**：支持 koniq10k、ava、spaq、paq2piq 等预训练权重
- **即插即用**：无需复杂配置，直接导入即可使用

### 分数说明

| 预训练模型 | 数据集 | 分数范围 | 说明 |
|------------|--------|----------|------|
| `koniq10k` | KonIQ-10k | 0-100 | 推荐用于图像质量评估 |
| `ava` | AVA | 1-10 | 美学质量评估 |
| `spaq` | SPAQ | 0-100 | 屏幕内容质量评估 |
| `paq2piq` | PAQ2PIQ | 0-100 | 压缩图像质量评估 |

---

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA（可选，用于 GPU 加速）

### 安装依赖

```bash
pip install torch torchvision huggingface_hub tqdm pillow
```

### 可选：安装 SAM（用于语义功能）

```bash
pip install segment-anything
```

下载 SAM 模型权重：
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## 快速开始

### 1. 基本使用（不使用语义嵌入）

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import sem_musiq

# 加载图像
img = Image.open('image.jpg').convert('RGB')

# 转换为 tensor [1, 3, H, W]
img_tensor = transforms.ToTensor()(img).unsqueeze(0)

# 移动到 GPU（如果有）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_tensor = img_tensor.to(device)

# 创建 MUSIQ 模型（使用 KonIQ-10k 预训练权重）
model = sem_musiq.MUSIQ(pretrained='koniq10k')
model.to(device)
model.eval()

# 推理
with torch.no_grad():
    score = model(img_tensor)

print(f"MUSIQ 分数：{score.item():.4f}")
```

### 2. 使用语义特征（双分支完整流程）

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import sem_musiq

# 加载图像
img = Image.open('image.jpg').convert('RGB')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. 提取 SAM 特征
sam_extractor = sem_musiq.SAMFeatureExtractor(
    sam_checkpoint='sam_vit_b_01ec64.pth',
    device=device,
)
sam_result = sam_extractor(img, return_details=False)
sam_embeddings = sam_result['sam_embeddings']  # List[np.ndarray], 每个 [N_s, 256]

# 2. 语义编码和一致性打分
semantic_encoder = sem_musiq.SemanticTransformerEncoder(
    input_dim=256,
    output_dim=384,
    num_scales=3,
)
consistency_scorer = sem_musiq.SemanticConsistencyScorer(num_scales=3)

# 编码语义特征
semantic_embeds = semantic_encoder(sam_embeddings)

# 计算一致性分数
sim_score = consistency_scorer(semantic_embeds)

# 3. MUSIQ 质量预测
model = sem_musiq.MUSIQ(pretrained='koniq10k')
model.to(device)
model.eval()

img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

with torch.no_grad():
    quality_score = model(img_tensor)

print(f"质量分数 Q = {quality_score.item():.4f}")
print(f"语义一致性 SIM = {sim_score:.4f}")

# 4. 融合分数
final_score = 0.7 * quality_score.item() + 0.3 * (sim_score * 100)
print(f"最终分数 F = {final_score:.4f}")
```

---

## 详细用法

### 模型配置选项

```python
model = sem_musiq.MUSIQ(
    # 基本参数
    patch_size=32,              # Patch 大小（默认 32）
    hidden_size=384,            # Transformer 隐藏层维度（默认 384）
    num_heads=6,                # 注意力头数（默认 6）
    num_layers=14,              # Transformer 层数（默认 14）
    
    # 多尺度配置
    longer_side_lengths=[224, 384],  # 多尺度缩放的长边长度
    max_seq_len_from_original_res=-1, # 原图尺度处理，-1 表示使用全部 patch
    
    # 预训练模型
    pretrained='koniq10k',      # 可选：'koniq10k', 'ava', 'spaq', 'paq2piq', True, False
    
    # 语义嵌入配置
    use_semantic=False,         # 是否启用语义嵌入
    semantic_input_dim=5,       # 语义向量维度（SAM mask 数量）
)
```

### 推理模式

```python
# 仅获取质量分数（MOS）
score = model(img_tensor, return_mos=True, return_dist=False)

# 获取预测分布（用于 AVA 模型）
dist = model(img_tensor, return_mos=False, return_dist=True)

# 同时获取两者
mos, dist = model(img_tensor, return_mos=True, return_dist=True)
```

### 批量推理

```python
# 批量图像（B, 3, H, W）
batch_tensor = torch.stack([img1, img2, img3], dim=0)

# 批量推理
with torch.no_grad():
    scores = model(batch_tensor)

print(f"批量分数：{scores.cpu().numpy()}")
```

### 完整流程示例

```python
"""
完整流程：图像质量评估 + 语义一致性分析
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
import sem_musiq

def evaluate_image(image_path, sam_checkpoint=None, lambda_q=0.7):
    """
    评估单张图像的质量和语义一致性
    
    Args:
        image_path: 图像路径
        sam_checkpoint: SAM 模型路径（可选）
        lambda_q: 质量分数权重（默认 0.7）
    
    Returns:
        dict: 包含各项分数的字典
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # 初始化结果
    result = {
        'image_path': image_path,
        'quality_score': None,
        'sim_score': None,
        'final_score': None,
    }
    
    # 1. 质量评估（不使用语义）
    model = sem_musiq.MUSIQ(pretrained='koniq10k')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        q_score = model(img_tensor)
    result['quality_score'] = q_score.item()
    
    # 2. 语义一致性分析（如果有 SAM）
    if sam_checkpoint:
        sam_extractor = sem_musiq.SAMFeatureExtractor(
            sam_checkpoint=sam_checkpoint,
            device=device,
        )
        sam_result = sam_extractor(img, return_details=False)
        sam_embeddings = sam_result['sam_embeddings']
        
        # 语义编码
        semantic_encoder = sem_musiq.SemanticTransformerEncoder(
            input_dim=256,
            output_dim=384,
            num_scales=3,
        )
        semantic_embeds = semantic_encoder(sam_embeddings)
        
        # 一致性打分
        consistency_scorer = sem_musiq.SemanticConsistencyScorer(num_scales=3)
        result['sim_score'] = consistency_scorer(semantic_embeds)
        
        # 3. 融合分数
        sim_normalized = result['sim_score'] * 100
        lambda_sim = 1 - lambda_q
        result['final_score'] = lambda_q * result['quality_score'] + lambda_sim * sim_normalized
    
    return result


# 使用示例
if __name__ == '__main__':
    result = evaluate_image(
        'image.jpg',
        sam_checkpoint='sam_vit_b_01ec64.pth',
        lambda_q=0.7,
    )
    
    print(f"图像：{result['image_path']}")
    print(f"质量分数：{result['quality_score']:.4f}")
    print(f"语义一致性：{result['sim_score']:.4f}")
    print(f"最终分数：{result['final_score']:.4f}")
```

---

## API 参考

### `sem_musiq.MUSIQ`

MUSIQ 模型主类。

#### 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `patch_size` | int | 32 | Patch 大小 |
| `num_class` | int | 1 | 输出类别数 |
| `hidden_size` | int | 384 | Transformer 隐藏层维度 |
| `mlp_dim` | int | 1152 | MLP 层维度 |
| `attention_dropout_rate` | float | 0.0 | Attention dropout |
| `dropout_rate` | float | 0 | Dropout 率 |
| `num_heads` | int | 6 | 注意力头数 |
| `num_layers` | int | 14 | Transformer 层数 |
| `num_scales` | int | 3 | 尺度数量 |
| `spatial_pos_grid_size` | int | 10 | 空间位置网格大小 |
| `use_scale_emb` | bool | True | 是否使用尺度嵌入 |
| `pretrained` | bool/str | True | 预训练模型名或路径 |
| `longer_side_lengths` | list | [224, 384] | 多尺度长边长度列表 |
| `max_seq_len_from_original_res` | int | -1 | 原图尺度最大序列长度 |
| `use_semantic` | bool | False | 是否使用语义嵌入 |
| `semantic_input_dim` | int | 5 | 语义输入维度 |

#### 方法

**`forward(target, return_mos=True, return_dist=False, semantic_vectors=None)`**

前向传播。

- **参数**：
  - `target` (torch.Tensor): 输入图像 [B, 3, H, W]
  - `return_mos` (bool): 是否返回 MOS 分数
  - `return_dist` (bool): 是否返回分布
  - `semantic_vectors` (torch.Tensor, optional): 语义向量 [B, N, K]

- **返回**：
  - MOS 分数或分布，或两者元组

---

### `sem_musiq.AddSemanticEmbs`

语义嵌入层，将语义向量投影到特征空间。

#### 参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `semantic_input_dim` | int | 语义输入维度 |
| `dim` | int | 输出特征维度 |

#### 方法

**`forward(inputs, semantic_vectors)`**

- **参数**：
  - `inputs` (torch.Tensor): 输入特征 [B, N, dim]
  - `semantic_vectors` (torch.Tensor): 语义向量 [B, N, semantic_input_dim]

- **返回**：
  - 融合后的特征 [B, N, dim]

---

## 常见问题

### Q1: 为什么分数和官方实现不一样？

A: 请检查以下几点：
1. 确保使用相同的预训练模型（`koniq10k` vs `ava`）
2. 确保图像归一化到 [0, 1] 范围
3. 确保模型处于 `eval()` 模式

### Q2: 如何使用自己的预训练权重？

A: 传入 `pretrained_model_path` 参数：
```python
model = sem_musiq.MUSIQ(
    pretrained=False,
    pretrained_model_path='path/to/your/weights.pth',
)
```

### Q3: 如何提取语义特征？

A: 使用 `sem_musiq.SAMFeatureExtractor`：
```python
from sem_musiq import SAMFeatureExtractor

extractor = SAMFeatureExtractor(
    sam_checkpoint='sam_vit_b.pth',
)
result = extractor(image_path)
sam_embeddings = result['sam_embeddings']  # List[np.ndarray], 每个 [N_s, 256]
```

### Q4: 如何在 CPU 上运行？

A: 设置 `device='cpu'`：
```python
device = 'cpu'
model = sem_musiq.MUSIQ(pretrained='koniq10k')
model.to(device)
```

### Q5: 如何提高推理速度？

A: 建议使用 GPU，并可使用半精度推理：
```python
model = sem_musiq.MUSIQ(pretrained='koniq10k').half().cuda()
img_tensor = img_tensor.half().cuda()

with torch.no_grad():
    score = model(img_tensor)
```

### Q6: 如何使用端到端的融合模型？

A: 使用 `SemMUSIQFusion` 类，一行代码完成全部流程：
```python
import sem_musiq

model = sem_musiq.SemMUSIQFusion(
    musiq_pretrained='koniq10k',
    sam_checkpoint='sam_vit_b.pth',
    top_k=5,
)

result = model.predict('image.jpg')
print(f"质量分数：{result['quality_score']}")
print(f"语义一致性：{result['sim_score']}")
print(f"融合分数：{result['final_score']}")
```

### Q7: 如何训练融合权重？

A: 使用 `train_fusion_only()` 方法，仅训练融合参数（冻结 MUSIQ 和 SAM）：
```python
# 准备你的数据集
dataloader = your_data_loader  # 每个 batch 包含 'image' 和 'mos'

# 训练
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

## 高级 API

### `sem_musiq.SemMUSIQFusion`

端到端融合模型，整合 SAM、MUSIQ 和可学习融合权重。

#### 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `musiq_pretrained` | str | 'koniq10k' | MUSIQ 预训练模型 |
| `sam_checkpoint` | str | None | SAM 模型权重路径 |
| `semantic_transformer_layers` | int | 6 | 语义 Transformer 层数 |
| `semantic_output_dim` | int | 384 | 语义嵌入输出维度 |
| `lambda_q_init` | float | 0.7 | 质量分数初始权重 |
| `device` | str | None | 运行设备 |

#### 方法

**`predict(image, return_details=False)`**

端到端预测。

- **参数**：
  - `image`: 输入图像（路径、PIL 或 numpy）
  - `return_details`: 是否返回详细信息

- **返回**：
  - dict，包含 quality_score, sim_score, final_score, weights

**`forward(img_tensor, sam_embeddings, sim_score, return_all=True)`**

前向传播（需要预先提取 SAM 特征）。

**`train_fusion_only(dataloader, criterion, optimizer, num_epochs)`**

仅训练融合权重（冻结 MUSIQ 和 SAM）。

**`get_current_weights()`**

获取当前融合权重。

---

### `sem_musiq.SAMFeatureExtractor`

SAM 特征提取器，提取多尺度 patch 的语义 embedding。

#### 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `sam_checkpoint` | str | None | SAM 模型权重路径 |
| `sam_model_type` | str | 'vit_b' | SAM 模型类型 |
| `device` | str | None | 运行设备 |
| `scale_longer_sides` | list | [224, 384, 512] | 各尺度长边长度 |

#### 方法

**`__call__(image, return_details=False)`**

提取 SAM 特征。

- **返回**：
  - `sam_embeddings`: List[np.ndarray] - 各尺度 embedding [N_s, 256]
  - `scale_info`: List[Dict] - 各尺度信息

---

### `sem_musiq.SemanticTransformerEncoder`

语义 Transformer 编码器，将 SAM embedding 投影到 Transformer 特征空间。

#### 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `input_dim` | int | 256 | 输入维度（SAM ViT-B 输出） |
| `output_dim` | int | 384 | 输出维度（与 MUSIQ hidden_size 对齐） |
| `num_layers` | int | 6 | Transformer 层数 |
| `num_scales` | int | 3 | 尺度数量 |

---

### `sem_musiq.SemanticConsistencyScorer`

语义一致性打分器，计算同尺度内 patch 之间的余弦相似度。

#### 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `num_scales` | int | 3 | 尺度数量 |
| `similarity_type` | str | 'mean' | 相似度聚合方式 |
| `temperature` | float | 1.0 | 温度参数 |

---

### `sem_musiq.QualityScoreFusion`

可学习的分数融合模块（独立组件）。

#### 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `lambda_q_init` | float | 0.7 | 质量分数初始权重 |
| `lambda_sim_init` | float | 0.3 | 语义分数初始权重 |

#### 方法

**`forward(quality_score, sim_score)`**

融合分数。

**`get_weights()`**

返回当前权重值。

---

## 许可证

基于 IQA-PyTorch 精简修改。

## 致谢

- MUSIQ 原作者：Junjie Ke et al.
- IQA-PyTorch 实现：Chaofeng Chen
