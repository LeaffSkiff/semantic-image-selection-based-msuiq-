"""
Sem-MUSIQ 使用示例

展示如何使用 sem-musiq 模块进行图像质量评估。
"""

import torch
from PIL import Image
import sem_musiq


def main():
    # 配置
    image_path = 'semantic_musiq/img.png'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_semantic = True  # 是否使用语义嵌入
    semantic_k = 5  # 语义向量维度（SAM mask 数量）

    print("=" * 60)
    print("Sem-MUSIQ 使用示例")
    print("=" * 60)
    print(f"图像：{image_path}")
    print(f"设备：{device}")
    print(f"语义嵌入：{'启用' if use_semantic else '禁用'}")
    print()

    # 加载图像
    img = Image.open(image_path).convert('RGB')
    print(f"图像尺寸：{img.size[0]} x {img.size[1]}")

    # 转换为 tensor [1, 3, H, W]
    import torchvision.transforms as transforms
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # 创建 MUSIQ 模型
    print("\n加载 MUSIQ 模型...")
    model = sem_musiq.MUSIQ(
        pretrained='koniq10k',  # 使用 KonIQ-10k 预训练权重（输出 0-100）
        use_semantic=use_semantic,
        semantic_input_dim=semantic_k if use_semantic else 5,
    )
    model.to(device)
    model.eval()

    # 准备语义向量（如果启用）
    semantic_vectors = None
    if use_semantic:
        print(f"\n生成语义向量（{semantic_k} 个 SAM mask）...")
        # 这里应该调用 semantic_vector_generator 生成真实的语义向量
        # 为了演示，我们创建随机向量
        from sem_musiq.data.multiscale_trans_util import get_multiscale_patches

        with torch.no_grad():
            # 获取 patch 数量
            patches = get_multiscale_patches(
                img_tensor,
                patch_size=32,
                patch_stride=32,
                hse_grid_size=10,
                longer_side_lengths=[224, 384],
                max_seq_len_from_original_res=-1,
            )
            num_patches = patches.shape[1]
            print(f"  总 patch 数量：{num_patches}")

        # 创建随机语义向量（实际应该用 SAM 计算）
        semantic_vectors = torch.rand(1, num_patches, semantic_k).to(device)
        print(f"  语义向量形状：{semantic_vectors.shape}")

    # 推理
    print("\n进行质量评估...")
    with torch.no_grad():
        score = model(
            img_tensor,
            return_mos=True,
            semantic_vectors=semantic_vectors,
        )

    print(f"\nMUSIQ 分数：{score.item():.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
