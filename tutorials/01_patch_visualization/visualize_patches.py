import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
可视化 MUSIQ 的 patch 提取过程

目标：看懂一张图像是怎么变成一堆 patch 的，以及每个尺度贡献了多少 patch
"""
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 推荐
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import torch
import numpy as np
from PIL import Image
from pyiqa.data.multiscale_trans_util import get_multiscale_patches, resize_preserve_aspect_ratio

# ================= 配置 =================
img_path = r".\tutorials\images\img.png"  # 图片路径：tutorials/images/img.png
patch_size = 32
longer_side_lengths = [224, 384]  # 两个缩放尺度
max_seq_len_from_original_res = -1  # -1 表示使用原始图的所有 patch

# ================= 加载图像 =================
print("=" * 60)
print("步骤 1: 加载图像")
print("=" * 60)

img = Image.open(img_path).convert("RGB")
w, h = img.size
print(f"原始图像尺寸：宽={w}, 高={h}")

# 计算原始图的 patch 数
orig_patches_w = int(np.ceil(w / patch_size))
orig_patches_h = int(np.ceil(h / patch_size))
orig_num_patches = orig_patches_w * orig_patches_h
print(f"原始图像 patch 数：{orig_patches_w} × {orig_patches_h} = {orig_num_patches} 个")

# 转成 tensor，shape: [1, 3, H, W]
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0)
print(f"Tensor shape: {img_tensor.shape}")
print()

# ================= 多尺度缩放可视化 =================
print("=" * 60)
print("步骤 2: 多尺度缩放 + Patch 数量计算")
print("=" * 60)

scale_info = []  # 存储每个尺度的信息

for i, longer_size in enumerate(longer_side_lengths):
    resized, rh, rw = resize_preserve_aspect_ratio(
        img_tensor, h, w, longer_size
    )
    num_patches_h = int(np.ceil(rh / patch_size))
    num_patches_w = int(np.ceil(rw / patch_size))
    num_patches = num_patches_h * num_patches_w

    # 计算该尺度的 max_seq_len
    max_seq_len = int(np.ceil(longer_size / patch_size) ** 2)

    scale_info.append({
        'name': f'尺度 {i+1} ({longer_size})',
        'size': (rw, rh),
        'num_patches_h': num_patches_h,
        'num_patches_w': num_patches_w,
        'num_patches': num_patches,
        'max_seq_len': max_seq_len,
        'image': resized,
    })

    print(f"{scale_info[-1]['name']}: 实际尺寸={rw}x{rh}")
    print(f"  → Patch 数量：{num_patches_h}×{num_patches_w} = {num_patches} 个")
    print(f"  → max_seq_len: {max_seq_len}")

# 原始分辨率
scale_info.append({
    'name': '原始分辨率',
    'size': (w, h),
    'num_patches_h': orig_patches_h,
    'num_patches_w': orig_patches_w,
    'num_patches': orig_num_patches,
    'image': img_tensor,
})
print(f"原始分辨率：实际尺寸={w}x{h}")
print(f"  → Patch 数量：{orig_patches_h}×{orig_patches_w} = {orig_num_patches} 个")

# 计算理论总数
theoretical_total = sum(s['num_patches'] for s in scale_info)
print(f"\n理论总 patch 数：{' + '.join(str(s['num_patches']) for s in scale_info)} = {theoretical_total}")
print()

# ================= 提取 patch =================
print("=" * 60)
print("步骤 3: 提取 patch (关键步骤!)")
print("=" * 60)

patches = get_multiscale_patches(
    img_tensor,
    patch_size=patch_size,
    patch_stride=patch_size,
    longer_side_lengths=longer_side_lengths,
    max_seq_len_from_original_res=max_seq_len_from_original_res,
)

print(f"输出 shape: {patches.shape}")
print(f"  - batch_size: {patches.shape[0]}")
print(f"  - 总 patch 数：{patches.shape[1]}")
print(f"  - 每个 patch 的特征维度：{patches.shape[2]}")
print()

# ================= 从 batch 中分离各尺度的 patch =================
print("=" * 60)
print("步骤 4: 从 batch 里读三个尺度的 patch")
print("=" * 60)

# patches shape: [1, num_total_patches, 3075]
# 每个 patch 的最后一维结构：
#   [pixel_values (3072 维), spatial_pos (1 维), scale_pos (1 维), mask (1 维)]

sample_patches = patches[0]  # shape: [num_patches, 3075]
num_total = sample_patches.shape[0]
pixel_dim = 3 * patch_size * patch_size  # 3072

# 读取 mask (最后一维) - 先用 numpy 统计
mask_np = sample_patches[:, -1].numpy()
print(f"实际输出总 patch 数：{num_total}")
print(f"有效 patch (mask=1): {(mask_np == 1).sum()}")
print(f"Padding patch (mask=0): {(mask_np == 0).sum()}")
print()

# 读取每个 patch 的 scale_id (倒数第 2 维)
scale_positions = sample_patches[:, -2].long().numpy()  # shape: [num_patches]

# 统计每个尺度的 patch 数量
print("从输出 tensor 中统计各尺度 patch 数量:")
actual_counts = {}
padding_counts = {}

for scale_idx in range(len(scale_info)):
    scale_mask = (scale_positions == scale_idx)
    count = ((scale_mask) & (mask_np == 1)).sum()
    padding_count = ((scale_mask) & (mask_np == 0)).sum()
    actual_counts[scale_idx] = int(count)
    padding_counts[scale_idx] = int(padding_count)
    print(f"  尺度 {scale_idx + 1} ({scale_info[scale_idx]['name']}): {count} 个有效 + {padding_count} 个 padding = {count + padding_count} 总计")

# 验证 mask (有效 patch)
mask = sample_patches[:, -1].bool()
valid_patches = mask.sum().item()
print(f"\n有效 patch 总数 (mask=1): {valid_patches}")
print(f"无效 patch 总数 (mask=0, padding): {num_total - valid_patches}")

# 检查是否匹配
if valid_patches == theoretical_total:
    print("✓ 有效 patch 数 = 理论总数，完美匹配!")
else:
    print(f"⚠ 有效 patch 数 ({valid_patches}) ≠ 理论总数 ({theoretical_total})")
    print("  可能原因：max_seq_len 限制了原始分辨率的 patch 数量")
print()

# ================= 提取每个尺度的 patch 数据 =================
print("=" * 60)
print("步骤 5: 提取每个尺度的 patch 特征")
print("=" * 60)

scale_patches = {}
for scale_idx in range(len(scale_info)):
    mask_scale = (scale_positions == scale_idx)
    scale_patch_data = sample_patches[mask_scale]
    scale_patches[scale_idx] = scale_patch_data

    if len(scale_patch_data) > 0:
        avg_pixel = scale_patch_data[:, :pixel_dim].mean(dim=1).mean().item()
        print(f"尺度 {scale_idx + 1}: {len(scale_patch_data)} 个 patch, 平均像素值={avg_pixel:.4f}")
print()

# ================= 可视化 =================
print("=" * 60)
print("步骤 6: 可视化")
print("=" * 60)

# 计算需要多少子图
n_scales = len(scale_info)
n_cols = 3
n_rows = (n_scales + 2) // 3 + 2  # 图像行 + 2 行图表

fig = plt.figure(figsize=(20, 12))

# 第 1 行：各尺度图像
for i, info in enumerate(scale_info):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    if i < len(longer_side_lengths):
        # 缩放图
        img_display = info['image'].squeeze(0).permute(1, 2, 0).numpy()
    else:
        # 原始图
        img_display = np.array(img)

    ax.imshow(img_display)
    ax.set_title(f"{info['name']}\n尺寸={info['size'][0]}x{info['size'][1]}\nPatch 数={info['num_patches']}")
    ax.axis("off")

# 第 2 行：各尺度 patch 数量对比
ax_bar = fig.add_subplot(n_rows, n_cols, n_cols + 1)
scale_names = [s['name'] for s in scale_info]
actual_vals = [actual_counts.get(i, 0) for i in range(len(scale_info))]
theoretical_vals = [s['num_patches'] for s in scale_info]

x = np.arange(len(scale_names))
width = 0.35

bars1 = ax_bar.bar(x - width/2, theoretical_vals, width, label='理论值', color='skyblue')
bars2 = ax_bar.bar(x + width/2, actual_vals, width, label='实际值', color='coral')

ax_bar.set_ylabel('Patch 数量')
ax_bar.set_title('各尺度 Patch 数量对比')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(scale_names, rotation=15)
ax_bar.legend()

# 在柱子上标注数值
for bar in bars1:
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

# 第 2 行中间：Scale position 分布
ax_scale = fig.add_subplot(n_rows, n_cols, n_cols + 2)
scale_pos = sample_patches[:, -2].long().numpy()
ax_scale.hist(scale_pos, bins=len(scale_info)+1, edgecolor='black')
ax_scale.set_xlabel('Scale Index')
ax_scale.set_ylabel('Patch 数量')
ax_scale.set_title('Scale Position 分布\n(0=尺度 1, 1=尺度 2, 2=原始分辨率)')
ax_scale.set_xticks(range(len(scale_info)+1))

# 第 2 行右边：有效/无效 patch 对比
ax_valid = fig.add_subplot(n_rows, n_cols, n_cols + 3)
labels = ['有效 patch', 'Padding patch']
sizes = [valid_patches, num_total - valid_patches]
colors = ['#2ecc71', '#95a5a6']
ax_valid.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
ax_valid.set_title('Patch 有效性分布')

# 第 3 行：Patch 平均像素值分布（按尺度分开）
for i in range(len(scale_info)):
    ax = fig.add_subplot(n_rows, n_cols, 2*n_cols + i + 1)
    if i in scale_patches and len(scale_patches[i]) > 0:
        patch_data = scale_patches[i]
        pixel_means = patch_data[:, :pixel_dim].mean(dim=1).numpy()
        ax.hist(pixel_means, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Pixel Mean')
        ax.set_ylabel('Count')
        ax.set_title(f'{scale_info[i]["name"]}\nPatch 像素均值分布')
    else:
        ax.text(0.5, 0.5, 'No patches', ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.savefig(r".\tutorials\outputImg\patch_visualization.png", dpi=150, bbox_inches='tight')
print(r"可视化已保存到：.\tutorials\outputImg\patch_visualization.png")
print()

# ================= 总结 =================
print("=" * 60)
print("总结")
print("=" * 60)
print(f"""
MUSIQ 的 patch 提取流程：

1. 输入图像 → 多尺度缩放 (如 224, 384)
2. 每个尺度 → 切成 32×32 的 patch
3. 所有尺度 patch → 按顺序拼接
4. 每个 patch 附加信息：
   - pixel_values: {pixel_dim} 维 (3 通道×32×32 像素)
   - spatial_pos: 这个 patch 在原图的什么位置 (哈希索引)
   - scale_pos: 这个 patch 来自哪个尺度 (0, 1, 2...)
   - mask: 标识是否是有效 patch (1=有效，0=padding)

5. 最终输出 shape: [batch_size, 总 patch 数，3075]

本示例中各尺度贡献：
""")

for i, info in enumerate(scale_info):
    actual = actual_counts.get(i, 0)
    print(f"   - {info['name']}: {actual} 个 patch (理论:{info['num_patches']})")

total_actual = sum(actual_counts.values())
print(f"\n   总计：{total_actual} 个 patch")
print("=" * 60)
