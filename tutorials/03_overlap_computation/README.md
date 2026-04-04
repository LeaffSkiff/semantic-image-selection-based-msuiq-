# 阶段 3：计算 patch-mask 重叠率

## 目标
生成语义向量 `s_i = [重叠率 1, ..., 重叠率 K]`。

## 运行

```bash
cd tutorials/03_overlap_computation
python compute_overlap.py
```

**输入**：
- `../02_sam_generation/sam_masks.pkl` - 需要先从阶段 2 复制过来
- `images/img.png` - 测试图像

**输出**：
- `semantic_vectors.pkl` - 语义向量
- `patch_mask_overlap_visualization.png` - 可视化

## 输入
- `sam_masks.pkl` - SAM 生成的 mask（来自阶段 2）

## 输出
- `semantic_vectors.pkl` - 语义向量
- `patch_mask_overlap_visualization.png` - 可视化

## 关键概念

### 重叠率计算公式

```
重叠率 = Area(Patch ∩ Mask) / Area(Patch)

= patch 与 mask 重叠的像素数 / patch 的总像素数
```

### 语义向量

```python
s_i = [重叠率_1, 重叠率_2, ..., 重叠率_K]

例如：s_i = [0.9, 0.1, 0.0, 0.0, 0.0]
       ↑    ↑    ↑    ↑    ↑
       这个 patch 90% 在 Mask 1 内，10% 在 Mask 2 内
```

## 输出解读

```
Scale 0 (224):
  Patch 0: [0.97, 0.0]  ← 这个 patch 几乎全是狗
  Patch 9: [0.22, 0.0]  ← 这个 patch 只有 22% 是狗

Scale Original:
  Patch 100: [0.1, 0.9] ← 这个 patch 主要是草地
```

## 语义向量的用途

阶段 4 会把它映射到 384 维，然后加到 MUSIQ 的 patch token 上：

```python
z_i = x_i + e_i^sem + e_i^pos + e_i^scale
      ↑      ↑
    像素   语义
```

## 下一步
进入 `04_semantic_embedding/`，实现语义嵌入矩阵。
