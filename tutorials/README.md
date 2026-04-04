# 学习教程 - 基于 MUSIQ+SAM 的语义图像选择系统

本文件夹包含完整的分步学习教程，带你从零开始实现语义图像选择系统。

---

## 前置要求

```bash
# 环境
Python 3.8+
PyTorch
MUSIQ (IQA-PyTorch)

# SAM 依赖（二选一）
pip install segment-anything      # 原版 SAM，效果好，权重大 (~2.5GB)
pip install mobile-sam            # 轻量版，速度快，权重小 (~40MB)
```

---

## 目录结构

```
tutorials/
├── 01_patch_visualization/       # 阶段 1：理解 MUSIQ patch 提取
│   ├── visualize_patches.py      # 可视化脚本
│   └── patch_visualization.png   # 输出示例
│
├── 02_sam_generation/            # 阶段 2：SAM 生成 mask
│   ├── run_sam.py                # SAM 运行脚本
│   ├── sam_install.txt           # 安装说明
│   ├── sam_masks_visualization.png
│   └── sam_masks.pkl
│
├── 03_overlap_computation/       # 阶段 3：计算 patch-mask 重叠率
│   ├── compute_overlap.py        # 重叠率计算脚本
│   ├── patch_mask_overlap_visualization.png
│   └── semantic_vectors.pkl
│
├── 04_semantic_embedding/        # 阶段 4：语义嵌入矩阵（待完成）
│   └── ...
│
├── 05_fusion/                    # 阶段 5：语义特征融合（待完成）
│   └── ...
│
├── 06_consistency/               # 阶段 6：跨尺度语义一致性（待完成）
│   └── ...
│
└── README.md                     # 本文件
```

---

## 学习路线

### 阶段 1：理解 MUSIQ patch 提取

**目标**：看懂一张图像是如何变成 patch 序列的。

**运行**：
```bash
cd tutorials/01_patch_visualization
python visualize_patches.py
```

**学习要点**：
- 多尺度是如何实现的
- patch 是怎么提取的
- position embedding 的作用
- padding 是怎么来的

**预计时间**：1-2 小时

---

### 阶段 2：SAM 生成 mask

**目标**：跑通 SAM，生成语义区域 mask。

**运行**：
```bash
cd tutorials/02_sam_generation
python run_sam.py
```

**学习要点**：
- SAM 的工作原理
- mask 的含义
- 如何筛选 Top-K mask

**预计时间**：1-2 小时（不含下载权重时间）

---

### 阶段 3：计算 patch-mask 重叠率

**目标**：生成语义向量 `s_i = [重叠率 1, ..., 重叠率 K]`。

**运行**：
```bash
cd tutorials/03_overlap_computation
python compute_overlap.py
```

**学习要点**：
- 重叠率的计算公式
- 语义向量的含义
- 不同尺度 patch 的对应关系

**预计时间**：1-2 小时

---

### 阶段 4：语义嵌入矩阵（待完成）

**目标**：将 K 维语义向量映射到 384 维。

**预计时间**：1-2 小时

---

### 阶段 5：语义特征融合（待完成）

**目标**：将语义嵌入融合到 MUSIQ patch token。

**预计时间**：2-3 小时

---

### 阶段 6：跨尺度语义一致性（待完成）

**目标**：计算 SIM 分数。

**预计时间**：1-2 小时

---

### 阶段 7：完整系统（待完成）

**目标**：整合所有模块，实现图像质量评估与筛选。

**预计时间**：2-3 小时

---

## 常见问题

### Q1: SAM 权重下载太慢
**A**: 使用 MobileSAM，或者从国内镜像站下载。

### Q2: 显存不够
**A**: 用 MobileSAM，或者减小图像分辨率。

### Q3: mask 数量太少/太多
**A**: 调整 `run_sam.py` 中的 `TOP_K` 和 `min_area` 参数。

### Q4: 重叠率都是 0 或 1
**A**: 正常，说明 patch 要么完全在 mask 内，要么完全在外面。

---

## 参考资源

- MUSIQ 论文：https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_ICCV_2021_paper.html
- SAM 论文：https://arxiv.org/abs/2304.02643
- SAM 代码：https://github.com/facebookresearch/segment-anything

---

**最后更新**：2026-04-04
