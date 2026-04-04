import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
阶段 3：计算 patch-mask 重叠率

目标：
1. 读取 MUSIQ 提取的 patch 边界坐标
2. 读取 SAM 生成的 mask
3. 计算每个 patch 与每个 mask 的重叠率
4. 生成语义向量 s_i = [o_i1, o_i2, ..., o_iK]
5. 可视化验证

输出：
- patch_mask_overlap.png (可视化)
- semantic_vectors.npy (语义向量，供下一阶段使用)
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

# ================= 配置 =================
img_path = r".\tutorials\images\img.png"  # 图片路径：tutorials/images/img.png
mask_path = r".\tutorials\02_sam_generation\sam_masks.pkl"  # SAM 输出的 mask 数据（来自阶段 2）
patch_size = 32
longer_side_lengths = [224, 384]
TOP_K = 5  # 与 run_sam.py 保持一致

print("=" * 60)
print("阶段 3：Patch-Mask 重叠率计算")
print("=" * 60)

# ================= 加载图像 =================
print("\n步骤 1: 加载图像")
img = Image.open(img_path).convert("RGB")
w, h = img.size
print(f"原始图像尺寸：{w} × {h}")

# ================= 加载 SAM mask =================
print("\n步骤 2: 加载 SAM mask")
with open(mask_path, "rb") as f:
    mask_data = pickle.load(f)

masks = mask_data['masks']      # [K, H, W]
scores = mask_data['scores']    # [K]
areas = mask_data['areas']      # [K]
mask_image_size = mask_data['image_size']  # (W, H)

K = len(masks)
print(f"Mask 数量：{K}")
print(f"Mask 图像尺寸：{mask_image_size}")
print(f"Top-{K} scores: {scores}")

# 确保 mask 尺寸与原始图像一致
assert mask_image_size == (w, h), f"Mask 尺寸 {mask_image_size} 与图像尺寸 {w}×{h} 不匹配!"

# ================= 定义多尺度缩放函数 =================
from pyiqa.data.multiscale_trans_util import resize_preserve_aspect_ratio

img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)

# ================= 计算各尺度的 patch 边界 =================
print("\n步骤 3: 计算各尺度 patch 的边界坐标")

scale_info = []
for i, longer_size in enumerate(longer_side_lengths):
    resized, rh, rw = resize_preserve_aspect_ratio(img_tensor, h, w, longer_size)
    rh, rw = int(rh), int(rw)

    num_patches_h = int(np.ceil(rh / patch_size))
    num_patches_w = int(np.ceil(rw / patch_size))

    scale_info.append({
        'name': f'Scale {i} ({longer_size})',
        'size': (rw, rh),
        'num_patches_h': num_patches_h,
        'num_patches_w': num_patches_w,
    })
    print(f"  {scale_info[-1]['name']}: {rw}×{rh}, {num_patches_w}×{num_patches_h} = {num_patches_w * num_patches_h} patches")

# 原始分辨率
scale_info.append({
    'name': 'Scale Original',
    'size': (w, h),
    'num_patches_h': int(np.ceil(h / patch_size)),
    'num_patches_w': int(np.ceil(w / patch_size)),
})
print(f"  {scale_info[-1]['name']}: {w}×{h}, {scale_info[-1]['num_patches_w']}×{scale_info[-1]['num_patches_h']} = {scale_info[-1]['num_patches_w'] * scale_info[-1]['num_patches_h']} patches")

# ================= 重叠率计算函数 =================
def compute_patch_mask_overlap(patch_x, patch_y, patch_w, patch_h, mask):
    """
    计算一个 patch 与 mask 的重叠率

    Args:
        patch_x, patch_y: patch 左上角坐标（在原图中的位置）
        patch_w, patch_h: patch 的宽高
        mask: 二值 mask [H, W]

    Returns:
        overlap_ratio: 重叠率 = Area(Patch ∩ Mask) / Area(Patch)
    """
    # patch 在 mask 上的区域
    y1, y2 = int(patch_y), int(patch_y + patch_h)
    x1, x2 = int(patch_x), int(patch_x + patch_w)

    # 确保不越界
    y1, y2 = max(0, y1), min(mask.shape[0], y2)
    x1, x2 = max(0, x1), min(mask.shape[1], x2)

    # 提取 mask 区域
    mask_region = mask[y1:y2, x1:x2]

    # 计算重叠率
    overlap_area = mask_region.sum()
    patch_area = (y2 - y1) * (x2 - x1)

    if patch_area == 0:
        return 0.0

    return float(overlap_area) / float(patch_area)


# ================= 对每个尺度计算重叠率 =================
print("\n步骤 4: 计算每个 patch 与每个 mask 的重叠率")

all_semantic_vectors = []  # 存储所有尺度的语义向量

for scale_idx, info in enumerate(scale_info):
    print(f"\n处理 {info['name']}...")

    rw, rh = info['size']
    num_patches_w = info['num_patches_w']
    num_patches_h = info['num_patches_h']

    # 计算缩放比例（从原图到当前尺度）
    scale_x = rw / w
    scale_y = rh / h

    semantic_vectors = []  # 存储当前尺度的语义向量

    for patch_i in range(num_patches_h):
        for patch_j in range(num_patches_w):
            # patch 在当前尺度图像上的坐标
            patch_x_scale = patch_j * patch_size
            patch_y_scale = patch_i * patch_size

            # 映射回原图坐标
            patch_x_orig = patch_x_scale / scale_x
            patch_y_orig = patch_y_scale / scale_y

            # patch 在原图上的尺寸
            patch_w_orig = patch_size / scale_x
            patch_h_orig = patch_size / scale_y

            # 对每个 mask 计算重叠率
            overlap_ratios = []
            for k in range(K):
                ratio = compute_patch_mask_overlap(
                    patch_x_orig, patch_y_orig,
                    patch_w_orig, patch_h_orig,
                    masks[k]
                )
                overlap_ratios.append(ratio)

            # 语义向量 s_i = [o_i1, o_i2, ..., o_iK]
            semantic_vectors.append(overlap_ratios)

    semantic_vectors = np.array(semantic_vectors)  # [N_patches, K]
    all_semantic_vectors.append(semantic_vectors)

    print(f"  语义向量 shape: {semantic_vectors.shape}")
    print(f"  重叠率统计:")
    print(f"    均值: {semantic_vectors.mean():.4f}")
    print(f"    最大值: {semantic_vectors.max():.4f}")
    print(f"最小值：{semantic_vectors.min():.4f}")

# ================= 保存语义向量 =================
print("\n步骤 5: 保存语义向量")

# 按尺度保存
output_data = {
    'semantic_vectors': all_semantic_vectors,  # List of [N_s, K]
    'scale_info': scale_info,
    'mask_scores': scores,
    'K': K,
}

with open(r"tutorials\03_overlap_computation\semantic_vectors.pkl", "wb") as f:
    pickle.dump(output_data, f)

print(f"已保存到 semantic_vectors.pkl")
print(f"  尺度数量：{len(all_semantic_vectors)}")
for i, sv in enumerate(all_semantic_vectors):
    print(f"  尺度 {i}: shape={sv.shape}")

# ================= 可视化 =================
print("\n步骤 6: 可视化")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 原图
axes[0, 0].imshow(img)
axes[0, 0].set_title(f"Original Image\n{w}×{h}")
axes[0, 0].axis("off")

# 2. Mask 叠加
ax_masks = axes[0, 1]
ax_masks.imshow(img)
colors = plt.cm.Set1(np.linspace(0, 1, K))
for k in range(K):
    # 为每个 mask 绘制轮廓
    ax_masks.contour(masks[k], colors=[colors[k]], linewidths=2, alpha=0.7)
ax_masks.set_title(f"SAM Masks (K={K})")
ax_masks.axis("off")

# 3. 语义向量热力图（所有 patch 的平均重叠率）
ax_heatmap = axes[0, 2]
# 拼接所有尺度的语义向量
all_sv_concat = np.vstack(all_semantic_vectors)
mean_overlap = all_sv_concat.mean(axis=0)  # [K]
bars = ax_heatmap.bar(range(K), mean_overlap, color=colors)
ax_heatmap.set_xlabel('Mask Index')
ax_heatmap.set_ylabel('Mean Overlap Ratio')
ax_heatmap.set_title('Mean Overlap Ratio per Mask')
ax_heatmap.set_xticks(range(K))
ax_heatmap.set_xticklabels([f'M{k+1}\n(s={s:.2f})' for k, s in enumerate(scores)], rotation=45)
# 在柱子上标注数值
for bar, val in zip(bars, mean_overlap):
    ax_heatmap.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# 4-6. 各尺度的 patch 语义强度图
for scale_idx in range(min(3, len(scale_info))):
    sv = all_semantic_vectors[scale_idx]
    info = scale_info[scale_idx]

    # 计算每个 patch 的最大重叠率（表示语义强度）
    patch_intensity = sv.max(axis=1)  # [N_patches]

    # 重构成网格
    grid = patch_intensity.reshape(info['num_patches_h'], info['num_patches_w'])

    ax = axes[1, scale_idx]
    im = ax.imshow(grid, cmap='viridis', aspect='auto')
    ax.set_title(f"{info['name']}\nPatch Semantic Intensity")
    ax.set_xlabel('Patch X')
    ax.set_ylabel('Patch Y')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(r".\tutorials\outputImg\patch_mask_overlap_visualization.png", dpi=150, bbox_inches='tight')
print(r"可视化已保存到：.\tutorials\outputImg\patch_mask_overlap_visualization.png")

# ================= 打印示例语义向量 =================
print("\n" + "=" * 60)
print("示例语义向量（前 10 个 patch）")
print("=" * 60)

for scale_idx in range(len(scale_info)):
    print(f"\n{scale_info[scale_idx]['name']}:")
    sv = all_semantic_vectors[scale_idx]
    for i in range(min(10, len(sv))):
        print(f"  Patch {i}: {sv[i]}")

# ================= 总结 =================
print("\n" + "=" * 60)
print("阶段 3 完成！")
print("=" * 60)
print("""
输出文件：
1. semantic_vectors.pkl - 语义向量（供下一阶段使用）
2. patch_mask_overlap_visualization.png - 可视化

语义向量说明：
- 每个 patch 有一个 K 维向量 s_i
- s_i[k] = patch i 与 mask k 的重叠率
- 范围 [0, 1]，0=无重叠，1=完全重叠

下一步：阶段 4 - 语义嵌入矩阵生成
将 K 维语义向量映射到 384 维（与 MUSIQ token 对齐）
""")
print("=" * 60)
