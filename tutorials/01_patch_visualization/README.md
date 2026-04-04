# 阶段 1：理解 MUSIQ patch 提取

## 目标
看懂一张图像是如何变成一堆 patch 的。

## 运行

```bash
cd tutorials/01_patch_visualization
python visualize_patches.py
```

**输出**：`patch_visualization.png` 会生成在当前目录。

## 关键概念

### Patch
把大图切成 32×32 的小块，就像拼图打散。

### 多尺度
同一张图，缩放到不同大小（224, 384, 原始），提取不同粒度的 patch。

### Position Embedding
告诉模型每个 patch 在原图的什么位置（左上？右下？）。

### Scale Embedding
告诉模型每个 patch 来自哪个尺度（224 还是 384）。

### Mask
标识 patch 是否有效（1=有效，0=padding）。

## 输出解读

```
尺度 1 (224): 42 个有效 + 43 个 padding = 85 总计
尺度 2 (384): 108 个有效 + 0 个 padding = 108 总计
原始分辨率：768 个有效 + 0 个 padding = 768 总计
```

**Padding 来源**：序列长度对齐，Transformer 要求固定长度输入。

## 下一步
进入 `02_sam_generation/`，学习 SAM 如何生成语义 mask。
