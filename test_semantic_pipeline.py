"""
测试双分支语义图像质量评估流程

- SAM 特征提取器：使用 checkpoints 下的预训练权重
- Transformer 编码器：随机初始化
- 一致性打分器：随机初始化
- MUSIQ：使用 koinq10k 预训练权重
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
import sem_musiq

# 配置
IMAGE_PATH = 'images/example.png'
SAM_CHECKPOINT = 'checkpoints/sam_vit_b_01ec64.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"使用设备：{DEVICE}")
print("=" * 60)

# ================= 1. 测试 SAM 特征提取 =================
print("\n[1/4] 测试 SAMFeatureExtractor...")

try:
    sam_extractor = sem_musiq.SAMFeatureExtractor(
        sam_checkpoint=SAM_CHECKPOINT,
        device=DEVICE,
    )

    img = Image.open(IMAGE_PATH).convert('RGB')
    sam_result = sam_extractor(img, return_details=False)

    print(f"  SAM 特征提取成功!")
    print(f"  尺度数量：{sam_result['num_scales']}")
    print(f"  Embedding 维度：{sam_result['embed_dim']}")
    for i, (emb, info) in enumerate(zip(sam_result['sam_embeddings'], sam_result['scale_info'])):
        print(f"  尺度 {i}: {emb.shape} (patch_size={info['patch_size']})")

except Exception as e:
    print(f"  SAM 特征提取失败：{e}")
    sam_result = None

# ================= 2. 测试语义 Transformer 编码 =================
print("\n[2/4] 测试 SemanticTransformerEncoder...")

if sam_result:
    try:
        semantic_encoder = sem_musiq.SemanticTransformerEncoder(
            input_dim=256,  # SAM ViT-B 输出维度
            output_dim=384,  # 与 MUSIQ hidden_size 对齐
            num_layers=6,
            num_scales=3,
        )
        semantic_encoder.to(DEVICE)
        semantic_encoder.eval()

        # 将 numpy 转为 tensor 并移动到 GPU
        sam_embeddings_tensor = [
            torch.from_numpy(emb).float().to(DEVICE)
            for emb in sam_result['sam_embeddings']
        ]

        with torch.no_grad():
            semantic_embeds = semantic_encoder(sam_embeddings_tensor)

        print(f"  语义编码成功!")
        for i, emb in enumerate(semantic_embeds):
            print(f"  尺度 {i}: {emb.shape}")

    except Exception as e:
        print(f"  语义编码失败：{e}")
        semantic_embeds = None
else:
    print("  跳过（SAM 特征提取失败）")
    semantic_embeds = None

# ================= 3. 测试一致性打分 =================
print("\n[3/4] 测试 SemanticConsistencyScorer...")

if semantic_embeds:
    try:
        consistency_scorer = sem_musiq.SemanticConsistencyScorer(num_scales=3)
        consistency_scorer.to(DEVICE)
        consistency_scorer.eval()

        with torch.no_grad():
            sim_score = consistency_scorer(semantic_embeds)

        print(f"  一致性打分成功!")
        print(f"  SIM 分数：{sim_score:.4f}")

    except Exception as e:
        print(f"  一致性打分失败：{e}")
        sim_score = None
else:
    print("  跳过（语义编码失败）")
    sim_score = None

# ================= 4. 测试 MUSIQ 质量预测 =================
print("\n[4/4] 测试 MUSIQ...")

try:
    musiq = sem_musiq.MUSIQ(pretrained='koniq10k')
    musiq.to(DEVICE)
    musiq.eval()

    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        quality_score = musiq(img_tensor)

    print(f"  MUSIQ 预测成功!")
    print(f"  质量分数 Q = {quality_score.item():.4f}")

except Exception as e:
    print(f"  MUSIQ 预测失败：{e}")
    quality_score = None

# ================= 融合分数 =================
print("\n" + "=" * 60)
print("最终结果:")

if quality_score is not None and sim_score is not None:
    lambda_q = 0.7
    lambda_sim = 0.3
    final_score = lambda_q * quality_score.item() + lambda_sim * (sim_score * 100)

    print(f"  质量分数 Q   = {quality_score.item():.4f}")
    print(f"  语义一致性 SIM = {sim_score:.4f}")
    print(f"  融合分数 F   = {final_score:.4f}")
    print(f"  (权重：λ_q={lambda_q}, λ_sim={lambda_sim})")
else:
    print("  部分模块测试失败，无法计算最终分数")

print("\n测试完成!")
