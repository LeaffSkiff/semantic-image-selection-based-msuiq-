# 快速参考卡

## 核心公式

### 重叠率
```
o_ij = Area(Patch_i ∩ Mask_j) / Area(Patch_i)
```

### 语义向量
```
s_i = [o_i1, o_i2, ..., o_iK] ∈ R^K
```

### 语义嵌入
```
e_i^sem = W_sem @ s_i + b ∈ R^384
```

### 融合
```
z_i = x_i + e_i^sem + e_i^pos + e_i^scale
```

### 质量分数
```
Q = Transformer(z_1, ..., z_N) → [0-100]
```

### 语义一致性
```
SIM = cosine_similarity(mean(E_sem^(a)), mean(E_sem^(b)))
```

### 最终筛选分
```
F = 0.7 × Q + 0.3 × SIM
```

---

## 文件位置

| 文件 | 位置 |
|------|------|
| patch 可视化 | `tutorials/01_patch_visualization/` |
| SAM mask | `tutorials/02_sam_generation/` |
| 语义向量 | `tutorials/03_overlap_computation/` |

---

##  Shapes 参考

```
原始图像：1024×768
Patch 大小：32×32

尺度 1 (224):  42 个 patch  →  [42, K]
尺度 2 (384):  108 个 patch →  [108, K]
原始分辨率：768 个 patch →  [768, K]

语义嵌入：[N_patches, 384]
```

---

## 常用命令

```bash
# 阶段 1
python tutorials/01_patch_visualization/visualize_patches.py

# 阶段 2
python tutorials/02_sam_generation/run_sam.py

# 阶段 3
python tutorials/03_overlap_computation/compute_overlap.py
```
