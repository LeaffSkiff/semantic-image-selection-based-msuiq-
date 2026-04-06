import torch
from PIL import Image
import torchvision.transforms as transforms
import sem_musiq
from semantic_vector_generator import SemanticVectorGenerator

# 加载图像
img_path = 'img.png'
img = Image.open(img_path).convert('RGB')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 60)
print("Sem-MUSIQ 完整流程演示")
print("=" * 60)
print(f"图像：{img_path}")
print(f"设备：{device}")
print()

# 创建语义向量生成器（使用 SAM）
sam_checkpoint = 'checkpoints/sam_vit_b_01ec64.pth'
print(f"加载 SAM 模型：{sam_checkpoint}...")
sem_generator = SemanticVectorGenerator(
    sam_checkpoint=sam_checkpoint,
    top_k=5,
    device=device,
)

# 生成语义向量
print("生成语义向量...")
result = sem_generator(img, return_details=False)
semantic_vectors = result['semantic_vectors']
sim_score = result['consistency_score']
print(f"  语义一致性 SIM = {sim_score:.4f}")

# 拼接各尺度的语义向量
import numpy as np
all_semantic_vectors = np.concatenate(semantic_vectors, axis=0)
semantic_tensor = torch.from_numpy(all_semantic_vectors).float().unsqueeze(0).to(device)
print(f"  语义向量形状：{semantic_tensor.shape}")

# 创建 MUSIQ 模型（启用语义嵌入）
print("加载 MUSIQ 模型...")
model = sem_musiq.MUSIQ(
    pretrained='koniq10k',
    use_semantic=True,
    semantic_input_dim=5,
)
model.to(device)
model.eval()

# 转换为 tensor
img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

# 推理（传入语义向量）
print("进行质量评估...")
with torch.no_grad():
    quality_score = model(img_tensor, semantic_vectors=semantic_tensor)

print()
print(f"质量分数 Q = {quality_score.item():.4f}")
print(f"语义一致性 SIM = {sim_score:.4f}")

# 融合分数
final_score = 0.7 * quality_score.item() + 0.3 * (sim_score * 100)
print(f"最终分数 F = {final_score:.4f}")
print("=" * 60)

# 完成了整个图像处理过程。剔除对 IQA-PyTorch 的依赖。新建模块 sem_musiq，包含了修改后的 musiq 以及部分辅助函数