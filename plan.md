# 基于 MUSIQ + SAM 的语义图像选择系统 - 实现计划

**核心思路**：MUSIQ 主干 + SAM 语义引导支路 —— 不动主干结构，只加一条语义增强分支。

---

## 系统架构图（文字版）

```
输入图像
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  模块 1: 多尺度图像生成                                       │
│  - 原图 I^(1)                                                │
│  - 缩放图 I^(2) (长边 75%)                                   │
│  - 裁剪图 I^(3) (中心裁剪)                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ├──► 模块 2: MUSIQ patch 编码 ──► patch token X^(s)
    │
    └──► 模块 3: SAM 语义引导 ──► mask {M1...M5}
                                      │
                                      ▼
                              模块 4: 语义嵌入矩阵
                              - patch-mask 重叠率
                              - 线性映射到 d 维
                                      │
                                      ▼
                              模块 5: 融合
                              z_i = x_i + e_sem + e_pos + e_scale
                                      │
                                      ▼
                              模块 6: 双输出
                              - 质量分 Q
                              - 语义一致性 SIM
                              - 最终分 F = λQ + (1-λ)SIM
```

---

## 任务清单

### Phase 1: 基础准备

- [ ] **Task 1.1**: 多尺度图像生成模块
  - 文件：`pyiqa/data/multiscale_trans_util.py`
  - 功能：生成原图 + 缩放图 + 裁剪图（3 尺度）
  - 输出：`{I^(1), I^(2), I^(3)}`

- [ ] **Task 1.2**: SAM 集成与环境配置
  - 文件：新建 `pyiqa/archs/semantic_branch.py`
  - 依赖：`pip install segment-anything`
  - 功能：加载 SAM 模型，生成 K=5 个主体 mask

---

### Phase 2: 语义嵌入模块

- [ ] **Task 2.1**: Patch-Mask 重叠率计算
  - 文件：`pyiqa/archs/semantic_branch.py`
  - 功能：对每个 patch 计算与 5 个 mask 的重叠率
  - 输出：$s_i = [o_{i,1}, ..., o_{i,5}] \in \mathbb{R}^5$

- [ ] **Task 2.2**: 语义嵌入矩阵生成
  - 文件：`pyiqa/archs/semantic_branch.py`
  - 功能：线性层映射 $e_i^{sem} = W_{sem}s_i + b \in \mathbb{R}^d$
  - 输出：$E_{sem}^{(s)} \in \mathbb{R}^{N_s \times d}$

---

### Phase 3: 融合与编码器

- [ ] **Task 3.1**: 融合模块实现
  - 文件：修改 `pyiqa/archs/musiq_arch.py`
  - 功能：$z_i = x_i + e_i^{sem} + e_i^{pos} + e_i^{scale}$
  - 关键：先内容融合，再加位置/尺度嵌入

- [ ] **Task 3.2**: Transformer 编码器适配
  - 文件：`pyiqa/archs/musiq_arch.py`
  - 功能：输入维度适配（如需）
  - 保持：14 层 encoder 结构不变

---

### Phase 4: 输出与筛选

- [ ] **Task 4.1**: 跨尺度语义一致性计算
  - 文件：`pyiqa/archs/semantic_branch.py`
  - 功能：全局池化 + 余弦相似度
  - 公式：$SIM = \frac{1}{|P|}\sum_{(a,b)} \frac{\bar{e}^{(a)} \cdot \bar{e}^{(b)}}{||\bar{e}^{(a)}|| \cdot ||\bar{e}^{(b)}||}$

- [ ] **Task 4.2**: 加权融合筛选分数
  - 文件：新建 `pyiqa/metrics/semantic_iqa_metric.py`
  - 功能：$F = \lambda \hat{Q} + (1-\lambda)\hat{SIM}$
  - 参数：$\lambda = 0.7$（可配置）

---

### Phase 5: 集成与测试

- [ ] **Task 5.1**: 完整评估脚本
  - 文件：`demo_semantic.py`
  - 功能：
    - 加载图像
    - 多尺度处理
    - SAM 语义提取
    - 融合推理
    - 输出 Q + SIM + F

- [ ] **Task 5.2**: 批量筛选测试
  - 文件：`batch_selection.py`
  - 功能：
    - 批量处理图像文件夹
    - 按 F 分数排序
    - 输出 Top-K 图像列表

---

## 文件结构（最终）

```
IQA-PyTorch/
├── pyiqa/
│   ├── archs/
│   │   ├── musiq_arch.py          # MUSIQ 主干（修改融合部分）
│   │   └── semantic_branch.py     # 新增：SAM 语义分支
│   ├── data/
│   │   └── multiscale_trans_util.py  # 多尺度处理（增 3 尺度）
│   └── metrics/
│       ├── __init__.py
│       └── semantic_iqa_metric.py    # 新增：加权融合指标
├── demo_semantic.py               # 新增：语义评估演示
├── batch_selection.py             # 新增：批量筛选脚本
└── plan.md                        # 本文件
```

---

## 依赖安装

```bash
# SAM 依赖
pip install segment-anything

# 或使用轻量版（推荐，如果显存有限）
pip install mobile-sam
```

---

## 关键参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 尺度数 | 3 | 原图 + 缩放 + 裁剪 |
| 缩放比例 | 0.75 | 长边缩放 |
| SAM mask 数 K | 5 | 保留前 5 大 mask |
| 语义嵌入维度 d | 384 | 与 MUSIQ hidden_size 对齐 |
| 权重 λ | 0.7 | 质量分占比 70% |

---

## 预期输出

1. **质量分 Q**: 0-100 范围，MUSIQ 预测的图像质量
2. **语义一致性 SIM**: 0-1 范围，跨尺度语义稳定性
3. **最终筛选分 F**: 归一化后 0-100，用于排序

---

## 风险与规避

| 风险 | 规避方案 |
|------|----------|
| SAM 显存占用高 | 使用 mobile-sam 或 CPU 推理 |
| patch-mask 对齐复杂 | 用重叠率近似，不做精确对齐 |
| 融合后性能下降 | 保持 MUSIQ 主干，语义作为增强 |
| 权重选择困难 | 先用λ=0.7，后续做消融实验 |

---

## 进度追踪

- [ ] Phase 1 完成：____/__/__
- [ ] Phase 2 完成：____/__/__
- [ ] Phase 3 完成：____/__/__
- [ ] Phase 4 完成：____/__/__
- [ ] Phase 5 完成：____/__/__

---

**最后更新**: 2026-04-03
**方案版本**: MVP v1.0（3 尺度 + 5 mask + 简单融合）
