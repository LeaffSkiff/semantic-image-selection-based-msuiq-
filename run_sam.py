"""
阶段 2：跑通 SAM，生成并可视化 mask

目标：
1. 加载 SAM 模型
2. 对输入图像生成 mask
3. 筛选出 Top-K 个主要区域
4. 可视化 mask 效果
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 推荐
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================= 配置 =================
img_path = r".\images\img.png"  # 改成你的图片路径
TOP_K = 5  # 只保留前 5 个最大的 mask
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"使用设备：{DEVICE}")
print(f"保留 Top-{TOP_K} 个 mask")
print()

# ================= 尝试导入 SAM =================
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_INSTALLED = True
    print("✓ segment-anything 已安装")
except ImportError:
    try:
        from mobile_sam import SamPredictor, sam_model_registry
        SAM_INSTALLED = True
        print("✓ mobile-sam 已安装")
    except ImportError:
        SAM_INSTALLED = False
        print("✗ SAM 未安装，请先运行：pip install segment-anything")
        print("  或 pip install mobile-sam (显存有限时)")
        exit(1)

# ================= 加载 SAM 模型 =================
print("=" * 60)
print("步骤 1: 加载 SAM 模型")
print("=" * 60)

# SAM 模型类型和权重路径
SAM_TYPE = "vit_b"  # 或者 vit_l, vit_b
SAM_CHECKPOINT = r"checkpoints\sam_vit_b_01ec64.pth"  # 改成你的权重路径

# 尝试自动查找权重文件
import os
if not os.path.exists(SAM_CHECKPOINT):
    # 尝试 mobile-sam 权重
    SAM_CHECKPOINT = r".\mobile_sam.pt"
    if not os.path.exists(SAM_CHECKPOINT):
        print("警告：未找到 SAM 权重文件!")
        print("请下载权重文件并放在项目根目录:")
        print("  - 原版 SAM: sam_vit_h_4b8939.pth")
        print("  - MobileSAM: mobile_sam.pt")
        print("\n或者尝试自动下载（如果网络可用）...")

        # 尝试使用默认权重（如果库支持）
        try:
            sam = sam_model_registry[SAM_TYPE](checkpoint=None)
        except:
            print("\n无法自动下载，请手动下载权重文件。")
            exit(1)
    else:
        print(f"找到 MobileSAM 权重：{SAM_CHECKPOINT}")
        SAM_TYPE = "vit_t"  # MobileSAM 使用 vit_t
else:
    print(f"找到 SAM 权重：{SAM_CHECKPOINT}")

# 加载模型
print("正在加载模型...")
sam = sam_model_registry.get(SAM_TYPE, sam_model_registry.get('vit_h', None))
if sam is None:
    # 尝试直接加载
    from segment_anything import sam_model_registry as sam_reg
    sam = sam_reg['vit_h'](checkpoint=SAM_CHECKPOINT)
else:
    sam = sam(checkpoint=SAM_CHECKPOINT)

sam.to(device=DEVICE)
sam.eval()
print(f"✓ 模型加载成功")
print()

# ================= 加载图像 =================
print("=" * 60)
print("步骤 2: 加载图像")
print("=" * 60)

image = Image.open(img_path).convert("RGB")
image_np = np.array(image)
h, w = image_np.shape[:2]
print(f"图像尺寸：{w} × {h}")
print()

# ================= 运行 SAM 预测 =================
print("=" * 60)
print("步骤 3: 运行 SAM 预测（可能需要几秒到几十秒）")
print("=" * 60)

predictor = SamPredictor(sam)
predictor.set_image(image_np)

# 全自动模式生成 mask
masks, scores, logits = predictor.predict(
    multimask_output=True,  # 输出多个 mask
)

print(f"生成了 {len(masks)} 个 mask")
print(f"mask shape: {masks.shape}")
print(f"分数范围：{scores.min():.4f} ~ {scores.max():.4f}")
print()

# ================= 筛选 Top-K 个 mask =================
print("=" * 60)
print("步骤 4: 筛选 Top-K 个 mask")
print("=" * 60)

# 按面积筛选（只保留足够大的 mask，避免噪声）
mask_areas = masks.sum(axis=(1, 2))
min_area = w * h * 0.001  # 最小面积为图像的 0.1%
valid_indices = np.where(mask_areas > min_area)[0]

print(f"有效 mask 数量（面积 > {min_area:.0f}）: {len(valid_indices)}")

if len(valid_indices) == 0:
    print("警告：没有找到足够大的 mask，可能是分割失败。")
    print("尝试降低 min_area 阈值。")
    valid_indices = np.arange(len(masks))

# 按分数排序，取 Top-K
valid_scores = scores[valid_indices]
top_k_indices = valid_indices[np.argsort(valid_scores)[-TOP_K:][::-1]]

print(f"保留 Top-{TOP_K} 个 mask，分数：")
for i, idx in enumerate(top_k_indices):
    area = mask_areas[idx]
    print(f"  Mask {i+1}: 分数={scores[idx]:.4f}, 面积={area:.0f} 像素")
print()

# ================= 可视化 =================
print("=" * 60)
print("步骤 5: 可视化")
print("=" * 60)

# 创建图形
n_masks = len(top_k_indices)
n_cols = 3
n_rows = (n_masks + 2) // 3 + 1

fig = plt.figure(figsize=(20, 12))

# 原图
ax_orig = fig.add_subplot(n_rows, n_cols, 1)
ax_orig.imshow(image)
ax_orig.set_title(f"Original Image\n{w}×{h}")
ax_orig.axis("off")

# 统计信息
ax_stats = fig.add_subplot(n_rows, n_cols, 2)
ax_stats.axis("off")
stats_text = f"Total masks: {len(masks)}\nValid masks: {len(valid_indices)}\nTop-K: {TOP_K}\n\nTop-{TOP_K} Scores:\n"
for i, idx in enumerate(top_k_indices):
    stats_text += f"  {i+1}. {scores[idx]:.4f} (area={mask_areas[idx]})\n"
ax_stats.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
              transform=ax_stats.transAxes)

# 所有 mask 的面积分布
ax_hist = fig.add_subplot(n_rows, n_cols, 3)
ax_hist.hist(mask_areas, bins=50, edgecolor='black')
ax_hist.set_xlabel('Area (pixels)')
ax_hist.set_ylabel('Count')
ax_hist.set_title(f'Mask Area Distribution\n(total={len(masks)})')
ax_hist.axvline(x=min_area, color='r', linestyle='--', label=f'Min area={min_area:.0f}')
ax_hist.legend()

# 可视化每个 Top-K mask
for i, idx in enumerate(top_k_indices):
    ax = fig.add_subplot(n_rows, n_cols, i + 4)

    # 显示原图
    ax.imshow(image)

    # 叠加 mask
    mask = masks[idx]

    # 方法 1：半透明覆盖
    overlay = np.zeros_like(image_np, dtype=float) / 255.0  # 归一化到 0-1
    overlay[mask > 0] = [1, 0, 0]  # 红色
    ax.imshow(overlay, alpha=0.3)

    # 方法 2：轮廓线
    from scipy import ndimage
    mask_edges = ndimage.binary_dilation(mask) ^ mask  # 用异或代替减法
    ax.contour(mask, colors='white', linewidths=1)

    ax.set_title(f"Mask {i+1}\nScore={scores[idx]:.4f}\nArea={mask_areas[idx]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("sam_masks_visualization.png", dpi=150, bbox_inches='tight')
print("可视化已保存到：sam_masks_visualization.png")
print()

# ================= 保存 mask 数据 =================
print("=" * 60)
print("步骤 6: 保存 mask 数据（供后续使用）")
print("=" * 60)

# 保存为 npy 文件
output_data = {
    'masks': masks[top_k_indices],
    'scores': scores[top_k_indices],
    'areas': mask_areas[top_k_indices],
    'image_size': (w, h),
}

import pickle
with open("sam_masks.pkl", "wb") as f:
    pickle.dump(output_data, f)

print(f"已保存 {TOP_K} 个 mask 到 sam_masks.pkl")
print(f"  - masks shape: {output_data['masks'].shape}")
print(f"  - 每个 mask: {w}×{h}")
print()

# ================= 总结 =================
print("=" * 60)
print("总结")
print("=" * 60)
print(f"""
SAM 已成功运行！

关键输出：
1. sam_masks_visualization.png - mask 可视化
2. sam_masks.pkl - mask 数据（后续步骤使用）

下一步：
- 检查可视化，确认 mask 是否合理
- 进入阶段 3：计算 patch-mask 重叠率
""")
print("=" * 60)
