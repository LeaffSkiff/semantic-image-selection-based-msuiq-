# 阶段 2：SAM 生成 mask

## 目标
跑通 SAM，生成语义区域 mask。

## 安装 SAM

```bash
# 二选一
pip install segment-anything      # 原版，效果好，权重~2.5GB
pip install mobile-sam            # 轻量版，速度快，权重~40MB
```

## 下载权重

**原版 SAM (vit_h)**：
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**MobileSAM**：
```
https://github.com/ChaoningZhang/MobileSAM
```

## 运行

```bash
cd tutorials/02_sam_generation
python run_sam.py
```

**注意**：首次运行前需要把 SAM 权重文件放在 `02_sam_generation/` 目录下。

## 输出
- `sam_masks_visualization.png` - mask 可视化
- `sam_masks.pkl` - mask 数据

## 关键概念

### SAM 是什么
Segment Anything Model，能自动找出图像中所有"有意义的物体/区域"。

### Mask
二值图像，白色表示"这个区域是一个物体"。

### Top-K 筛选
只保留前 K 个最大的 mask（默认 K=5），避免噪声。

## 输出解读

```
Mask 1: Score=0.95, Area=200000  ← 天空
Mask 2: Score=0.89, Area=150000  ← 房子
Mask 3: Score=0.82, Area=80000   ← 树木
```

## 下一步
进入 `03_overlap_computation/`，计算 patch-mask 重叠率。
