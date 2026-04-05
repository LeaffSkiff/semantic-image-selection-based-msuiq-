import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""
阶段 4：语义嵌入矩阵生成

目标：
1. 加载阶段 3 生成的语义向量 s_i ∈ R^K
2. 定义线性变换层 W_sem: R^K → R^384
3. 生成语义嵌入 e_i^sem = W_sem @ s_i + b
4. 应用 LayerNorm 归一化
5. 可视化语义嵌入分布
6. 保存供阶段 5 使用

核心公式：
    e_i^sem = LayerNorm(W_sem @ s_i + b)

其中：
- s_i: patch i 的语义向量 [K 维重叠率]
- W_sem: 可学习权重 [384, K]
- b: 偏置 [384]
- e_i^sem: 语义嵌入 [384 维，与 MUSIQ hidden_size 对齐]

为什么需要 384 维？
- MUSIQ 的 Transformer hidden_size = 384
- 语义嵌入需要与图像特征 x_i 相加：z_i = x_i + e_i^sem + ...
- 维度必须一致才能进行元素级相加

输出：
- semantic_embeddings.pkl - 语义嵌入矩阵 [N_patches, 384]
- semantic_embedding_visualization.png - 可视化
- linear_weight_visualization.png - 线性权重可视化
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pickle

# ================= 配置 =================
SEMANTIC_EMBEDDING_DIM = 384  # 与 MUSIQ hidden_size 对齐
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("阶段 4：语义嵌入矩阵生成")
print("=" * 60)

# ================= 步骤 1: 加载阶段 3 生成的语义向量 =================
print("\n步骤 1: 加载语义向量")

with open(r".\tutorials\03_overlap_computation\semantic_vectors.pkl", "rb") as f:
    stage3_data = pickle.load(f)

semantic_vectors = stage3_data['semantic_vectors']  # List of [N_s, K]
scale_info = stage3_data['scale_info']
mask_scores = stage3_data['mask_scores']
K = stage3_data['K']

print(f"加载了 {len(semantic_vectors)} 个尺度的语义向量")
for i, sv in enumerate(semantic_vectors):
    print(f"  尺度 {i}: shape={sv.shape}, K={K}")

print(f"\n使用设备：{DEVICE}")

# ================= 步骤 2: 定义语义嵌入层 =================
print("\n" + "=" * 60)
print("步骤 2: 定义语义嵌入层")
print("=" * 60)

class SemanticEmbedding(nn.Module):
    """
    语义嵌入层：将 K 维语义向量映射到 384 维语义嵌入空间

    输入：s_i ∈ R^K (patch i 与 K 个 mask 的重叠率)
    输出：e_i^sem ∈ R^384 (语义嵌入)
    """
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, s):
        """
        Args:
            s: 语义向量 [N, K]
        Returns:
            e_sem: 语义嵌入 [N, 384]
        """
        e_sem = self.linear(s)
        e_sem = self.layer_norm(e_sem)
        return e_sem


model = SemanticEmbedding(input_dim=K, embedding_dim=SEMANTIC_EMBEDDING_DIM).to(DEVICE)

print(f"SemanticEmbedding 层参数:")
print(f"  输入维度：{K}")
print(f"  输出维度：{SEMANTIC_EMBEDDING_DIM}")
print(f"  W_sem 形状：{model.linear.weight.shape}")
print(f"  b 形状：{model.linear.bias.shape}")

# ================= 步骤 3: 生成语义嵌入 =================
print("\n" + "=" * 60)
print("步骤 3: 生成语义嵌入")
print("=" * 60)

all_semantic_embeddings = []

for scale_idx, sv in enumerate(semantic_vectors):
    print(f"\n处理尺度 {scale_idx}...")

    s_tensor = torch.from_numpy(sv).float().to(DEVICE)
    print(f"  输入语义向量 shape: {s_tensor.shape}")

    with torch.no_grad():
        e_sem = model(s_tensor)

    print(f"  输出语义嵌入 shape: {e_sem.shape}")

    e_sem_np = e_sem.cpu().numpy()
    all_semantic_embeddings.append(e_sem_np)

    print(f"  语义嵌入统计:")
    print(f"    均值：{e_sem_np.mean():.6f}")
    print(f"    标准差：{e_sem_np.std():.6f}")
    print(f"    最小值：{e_sem_np.min():.6f}")
    print(f"    最大值：{e_sem_np.max():.6f}")

# ================= 步骤 4: 保存语义嵌入 =================
print("\n" + "=" * 60)
print("步骤 4: 保存语义嵌入")
print("=" * 60)

output_data = {
    'semantic_embeddings': all_semantic_embeddings,
    'semantic_vectors': semantic_vectors,
    'scale_info': scale_info,
    'mask_scores': mask_scores,
    'embedding_dim': SEMANTIC_EMBEDDING_DIM,
    'K': K,
}

with open(r".\tutorials\04_semantic_embedding\semantic_embeddings.pkl", "wb") as f:
    pickle.dump(output_data, f)

print(f"已保存到 semantic_embeddings.pkl")
print(f"  尺度数量：{len(all_semantic_embeddings)}")
for i, se in enumerate(all_semantic_embeddings):
    print(f"  尺度 {i}: shape={se.shape}")

# ================= 步骤 5: 可视化语义嵌入分布 =================
print("\n" + "=" * 60)
print("步骤 5: 可视化语义嵌入分布")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 原始语义向量热力图（前 20 个 patch）
ax1 = axes[0, 0]
sv_sample = semantic_vectors[0][:20, :]
im1 = ax1.imshow(sv_sample, cmap='YlOrRd', aspect='auto')
ax1.set_xlabel('Mask Index (K)')
ax1.set_ylabel('Patch Index')
ax1.set_title(f'原始语义向量 (s_i)\nShape: {sv_sample.shape}')
ax1.set_xticks(range(K))
ax1.set_xticklabels([f'M{k}' for k in range(K)])
plt.colorbar(im1, ax=ax1)

# 2. 语义嵌入热力图（前 20 个 patch，前 50 维）
ax2 = axes[0, 1]
se_sample = all_semantic_embeddings[0][:20, :50]
im2 = ax2.imshow(se_sample, cmap='viridis', aspect='auto')
ax2.set_xlabel('Embedding Dimension')
ax2.set_ylabel('Patch Index')
ax2.set_title(f'语义嵌入 (e_i^sem)\nShape: {se_sample.shape} (前 50 维)')
plt.colorbar(im2, ax=ax2)

# 3. 语义嵌入分布直方图
ax3 = axes[0, 2]
all_embeddings = np.vstack(all_semantic_embeddings)
ax3.hist(all_embeddings.flatten(), bins=50, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Embedding Value')
ax3.set_ylabel('Frequency')
ax3.set_title(f'语义嵌入分布\nMean={all_embeddings.mean():.4f}, Std={all_embeddings.std():.4f}')
ax3.axvline(all_embeddings.mean(), color='r', linestyle='--', label='Mean')
ax3.legend()

# 4. 嵌入维度散点图（随机选两个维度观察分布）
ax4 = axes[1, 0]
# 随机选两个维度做散点图
dim1, dim2 = 10, 20
ax4.scatter(all_embeddings[:500, dim1], all_embeddings[:500, dim2], alpha=0.5, s=10)
ax4.set_xlabel(f'Dimension {dim1}')
ax4.set_ylabel(f'Dimension {dim2}')
ax4.set_title(f'嵌入维度散点图 (Dim {dim1} vs {dim2})')
ax4.grid(True, alpha=0.3)

# 5. 嵌入维度相关性热力图（前 20 维）
ax5 = axes[1, 1]
corr_matrix = np.corrcoef(all_embeddings[:, :20].T)
im5 = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax5.set_xlabel('Embedding Dimension')
ax5.set_ylabel('Embedding Dimension')
ax5.set_title('相关性矩阵 (前 20 维)')
plt.colorbar(im5, ax=ax5)

# 6. 每个尺度的嵌入范数分布
ax6 = axes[1, 2]
for scale_idx, se in enumerate(all_semantic_embeddings):
    norms = np.linalg.norm(se, axis=1)
    ax6.hist(norms, alpha=0.5, bins=30, label=f'Scale {scale_idx}')

ax6.set_xlabel('L2 Norm')
ax6.set_ylabel('Frequency')
ax6.set_title('各尺度嵌入范数分布')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r".\tutorials\outputImg\semantic_embedding_visualization.png", dpi=150, bbox_inches='tight')
print(r"可视化已保存到：.\tutorials\outputImg\semantic_embedding_visualization.png")

# ================= 步骤 6: 分析线性变换权重 =================
print("\n" + "=" * 60)
print("步骤 6: 分析线性变换权重 W_sem")
print("=" * 60)

W_sem = model.linear.weight.cpu().detach().numpy()
b = model.linear.bias.cpu().detach().numpy()

print(f"W_sem 形状：{W_sem.shape}")
print(f"  每行对应一个输出维度，每列对应一个输入 mask")
print(f"  W_sem[:, k] 表示 mask k 对各嵌入维度的影响")

print(f"\nb 形状：{b.shape}")
print(f"  均值：{b.mean():.6f}")
print(f"  标准差：{b.std():.6f}")

# 可视化 W_sem
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(W_sem[:100, :], cmap='coolwarm', aspect='auto')
ax.set_xlabel('Mask Index (Input K)')
ax.set_ylabel('Embedding Dimension (前 100 维)')
ax.set_title(f'线性变换权重 W_sem\nShape: {W_sem.shape} (显示前 100 维)')
ax.set_xticks(range(K))
ax.set_xticklabels([f'M{k}\n(s={mask_scores[k]:.2f})' for k in range(K)])
plt.colorbar(im, ax=ax, label='Weight Value')
plt.tight_layout()
plt.savefig(r".\tutorials\outputImg\linear_weight_visualization.png", dpi=150, bbox_inches='tight')
print(r"权重可视化已保存到：.\tutorials\outputImg\linear_weight_visualization.png")

# ================= 总结 =================
print("\n" + "=" * 60)
print("阶段 4 完成！")
print("=" * 60)
print("""
输出文件：
1. semantic_embeddings.pkl - 语义嵌入矩阵 [N_patches, 384]
2. semantic_embedding_visualization.png - 嵌入分布可视化
3. linear_weight_visualization.png - 线性权重可视化

语义嵌入说明：
- 每个 patch 有一个 384 维向量 e_i^sem
- 通过 e_i^sem = W_sem @ s_i + b 生成
- 经 LayerNorm 归一化，均值≈0，标准差≈1
- 与 MUSIQ hidden_size 对齐，可直接相加融合

下一步：阶段 5 - 语义特征融合到 MUSIQ
修改 musiq_arch.py，实现：z_i = x_i + e_i^sem + e_i^pos + e_i^scale
""")
print("=" * 60)

# 调试信息
print("\n=== 调试信息 ===")
print(f"W_sem 均值：{W_sem.mean():.6f}")
print(f"W_sem 标准差：{W_sem.std():.6f}")
print(f"W_sem 最小值：{W_sem.min():.6f}")
print(f"W_sem 最大值：{W_sem.max():.6f}")

print(f"\n输入语义向量范围：{semantic_vectors[0].min():.4f} ~ {semantic_vectors[0].max():.4f}")

# 检查线性变换输出（LayerNorm 之前）
with torch.no_grad():
    raw_output = model.linear(torch.from_numpy(semantic_vectors[0]).float().to(DEVICE))
print(f"\n线性变换输出（LayerNorm 前）:")
print(f"  均值：{raw_output.mean():.6f}")
print(f"  标准差：{raw_output.std():.6f}")
print(f"  L2 范数均值：{torch.norm(raw_output, dim=1).mean():.6f}")

# 调试：检查 LayerNorm 后的范数
print("\nLayerNorm 后的 L2 范数:")
for scale_idx, se in enumerate(all_semantic_embeddings):
    norms = np.linalg.norm(se, axis=1)
    print(f"  尺度 {scale_idx}: 均值={norms.mean():.4f}, 范围={norms.min():.4f}~{norms.max():.4f}")
