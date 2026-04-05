import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
阶段 5：语义特征融合验证

目标：
1. 加载阶段 4 生成的语义嵌入
2. 将语义嵌入对齐到 MUSIQ patch 序列（处理 padding）
3. 验证修改后的 MUSIQ 能否正确接收并融合语义嵌入
4. 对比加/不加语义嵌入的输出差异

核心公式：
    z_i = x_i + e_i^sem + e_i^pos + e_i^scale

其中：
- x_i: 图像特征（CNN 编码，384 维）
- e_i^sem: 语义嵌入（阶段 4 生成，384 维）
- e_i^pos: 空间位置嵌入（哈希网格，384 维）
- e_i^scale: 尺度嵌入（可学习参数，384 维）

输出：
- 对比 MOS 分数差异
- 验证语义嵌入是否成功影响模型输出
"""

import torch
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from pyiqa.archs.musiq_arch import MUSIQ
from pyiqa.data.multiscale_trans_util import get_multiscale_patches

print("=" * 60)
print("阶段 5：语义特征融合验证")
print("=" * 60)

# ================= 配置 =================
IMG_PATH = r".\tutorials\images\img.png"
SEMANTIC_EMBEDDING_PATH = r".\tutorials\04_semantic_embedding\semantic_embeddings.pkl"
OUTPUT_VIS_PATH = r".\tutorials\outputImg\fusion_test_result.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备：{DEVICE}")

# ================= 步骤 1: 加载 MUSIQ 模型 =================
print("\n步骤 1: 加载 MUSIQ 模型")
model = MUSIQ(pretrained='koniq10k').to(DEVICE)
model.eval()
print(f"模型：MUSIQ (koniq10k 预训练)")
print(f"Hidden size: 384")
print(f"Transformer 层数：14")

# ================= 步骤 2: 加载图像并提取 patch =================
print("\n步骤 2: 加载图像并提取 patch")

img = Image.open(IMG_PATH).convert("RGB")
print(f"图像尺寸：{img.size[0]} × {img.size[1]}")

# 转换为 tensor 并预处理
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
img_tensor = (img_tensor - 0.5) * 2  # 归一化到 [-1, 1]

# 提取多尺度 patch
with torch.no_grad():
    patches = get_multiscale_patches(img_tensor, **model.data_preprocess_opts)

print(f"Patches shape: {patches.shape}")
print(f"  - Batch size: {patches.shape[0]}")
print(f"  - 序列长度：{patches.shape[1]}")
print(f"  - 特征维度：{patches.shape[2]} (pixel + spatial_pos + scale_pos + mask)")

# 分析有效 patch 和 padding
patch_mask = patches[0, :, -1].cpu().numpy()
valid_count = int(patch_mask.sum())
padding_count = len(patch_mask) - valid_count
print(f"  - 有效 patch: {valid_count}")
print(f"  - Padding patch: {padding_count}")

# ================= 步骤 3: 加载语义嵌入 =================
print("\n步骤 3: 加载阶段 4 生成的语义嵌入")

with open(SEMANTIC_EMBEDDING_PATH, "rb") as f:
    sem_data = pickle.load(f)

semantic_embeddings_list = sem_data['semantic_embeddings']  # List of [N_s, 384]
scale_info = sem_data['scale_info']
K = sem_data['K']

print(f"尺度数量：{len(semantic_embeddings_list)}")
for i, se in enumerate(semantic_embeddings_list):
    print(f"  尺度 {i} ({scale_info[i]['name']}): shape={se.shape}")

# 计算总有效 patch 数
total_valid_patches = sum(se.shape[0] for se in semantic_embeddings_list)
print(f"总有效 patch 数：{total_valid_patches}")

# ================= 步骤 4: 对齐语义嵌入到 patch 序列 =================
print("\n步骤 4: 对齐语义嵌入到 MUSIQ patch 序列")

# 找到有效 patch 的索引（mask=1 的位置）
valid_indices = np.where(patch_mask == 1)[0]
print(f"有效 patch 索引范围：[{valid_indices.min()}, {valid_indices.max()}]")

# 创建完整的语义嵌入 tensor（包含 padding 位置）
# padding 位置的语义嵌入设为 0（无语义信息）
full_sem = np.zeros((patches.shape[1], 384), dtype=np.float32)

# 按尺度顺序填充语义嵌入
current_idx = 0
for scale_idx, sem_emb in enumerate(semantic_embeddings_list):
    # 找到当前尺度的 patch 在序列中的索引范围
    scale_valid_indices = valid_indices[current_idx:current_idx + len(sem_emb)]

    for i, seq_idx in enumerate(scale_valid_indices):
        full_sem[seq_idx] = sem_emb[i]

    current_idx += len(sem_emb)
    print(f"  尺度 {scale_idx}: 填充到序列索引 [{scale_valid_indices.min()}, {scale_valid_indices.max()}]")

# 转换为 torch tensor
full_sem = torch.from_numpy(full_sem).float().unsqueeze(0).to(DEVICE)
print(f"完整语义嵌入 shape: {full_sem.shape}")
print(f"  - 包含 padding: {full_sem.shape[1]} 个位置")
print(f"  - 有效语义：{total_valid_patches} 个 patch")

# ================= 步骤 5: 前向传播测试 =================
print("\n步骤 5: 运行前向传播测试")

# 注意：使用 train 模式，因为 patches 已经提取好了
model.train()

with torch.no_grad():
    # 测试 1: 无语义嵌入
    print("  [1] 无语义嵌入...")
    out1 = model(patches, return_mos=True)
    mos1 = out1.item()

    # 测试 2: 有语义嵌入
    print("  [2] 有语义嵌入...")
    out2 = model(patches, semantic_embeddings=full_sem, return_mos=True)
    mos2 = out2.item()

diff = mos2 - mos1
diff_percent = diff / mos1 * 100 if mos1 != 0 else 0

print("\n" + "=" * 60)
print("测试结果")
print("=" * 60)
print(f"  无语义嵌入：MOS = {mos1:.4f}")
print(f"  有语义嵌入：MOS = {mos2:.4f}")
print(f"  绝对差异：{diff:+.4f}")
print(f"  相对差异：{diff_percent:+.2f}%")

# ================= 步骤 6: 结果分析 =================
print("\n" + "=" * 60)
print("结果分析")
print("=" * 60)

if abs(diff) > 1e-4:
    print("[OK] 语义嵌入融合成功！")
    print("  语义嵌入成功影响了 MUSIQ 的输出。")
    print("  当前使用的是随机初始化的 W_sem，分数变化较大是预期行为。")
    print("  下一步需要通过训练学习合适的权重。")
else:
    print("[FAIL] 两次输出几乎相同！")
    print("  可能原因：")
    print("  1. forward() 方法未正确修改")
    print("  2. 语义嵌入未正确传入 TransformerEncoder")
    print("  3. 语义嵌入全为 0 或被 mask 掉了")

# ================= 步骤 7: 可视化 =================
print("\n" + "=" * 60)
print("Step 7: Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 原图
ax1 = axes[0, 0]
ax1.imshow(img)
ax1.set_title(f"Test Image\n{img.size[0]} x {img.size[1]}")
ax1.axis("off")

# 2. Patch 有效性可视化
ax2 = axes[0, 1]
# 按尺度分段显示
scale_names = [f"S{i}\n({se.shape[0]})" for i, se in enumerate(semantic_embeddings_list)]
scale_counts = [se.shape[0] for se in semantic_embeddings_list]

bars = ax2.bar(range(len(scale_counts)), scale_counts, color=['#2ecc71', '#3498db', '#e74c3c'])
ax2.set_xticks(range(len(scale_counts)))
ax2.set_xticklabels(scale_names)
ax2.set_ylabel('Patch Count')
ax2.set_title('Valid Patches per Scale')
ax2.tick_params(axis='x', rotation=0)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=9)

# 3. MOS 分数对比
ax3 = axes[1, 0]
categories = ['No Semantic', 'With Semantic']
mos_values = [mos1, mos2]
colors = ['#95a5a6', '#3498db']
bars = ax3.bar(categories, mos_values, color=colors)
ax3.set_ylabel('MOS Score (0-100)')
ax3.set_title(f'MOS Comparison\nDiff: {diff:+.4f} ({diff_percent:+.2f}%)')
ax3.set_ylim(0, 100)

# 添加数值标签
for bar, mos in zip(bars, mos_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{mos:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. 语义嵌入分布
ax4 = axes[1, 1]
all_se = np.vstack(semantic_embeddings_list)
ax4.hist(all_se.flatten(), bins=50, edgecolor='black', alpha=0.7, color='#9b59b6')
ax4.set_xlabel('Embedding Value')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Semantic Embedding Distribution\nMean={all_se.mean():.4f}, Std={all_se.std():.4f}')
ax4.axvline(all_se.mean(), color='r', linestyle='--', label='Mean')
ax4.legend()

plt.tight_layout()
plt.savefig(OUTPUT_VIS_PATH, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {OUTPUT_VIS_PATH}")

# ================= 打印示例语义嵌入 =================
print("\n" + "=" * 60)
print("示例语义嵌入（前 5 个 patch，前 10 维）")
print("=" * 60)

for scale_idx in range(min(3, len(semantic_embeddings_list))):
    print(f"\n{scale_info[scale_idx]['name']}:")
    se = semantic_embeddings_list[scale_idx]
    print(f"  Patch 0: {se[0, :10]} ... (共{se.shape[1]}维)")
    print(f"  Patch 1: {se[1, :10]} ...")

# ================= 总结 =================
print("\n" + "=" * 60)
print("阶段 5 完成！")
print("=" * 60)
print(f"""
测试结果：
- 无语义嵌入 MOS: {mos1:.4f}
- 有语义嵌入 MOS: {mos2:.4f}
- 差异：{diff:+.4f} ({diff_percent:+.2f}%)

输出文件：
1. fusion_test_result.png - 融合测试可视化

语义嵌入融合说明：
- 语义嵌入通过 z_i = x_i + e_i^sem + e_i^pos + e_i^scale 融合
- 当前 W_sem 为随机初始化，分数变化大是预期行为
- 下一步：通过训练学习合适的 W_sem 权重

下一步：阶段 6 - 跨尺度语义一致性度量
计算各尺度语义嵌入的全局池化和跨尺度余弦相似度
""")
print("=" * 60)
