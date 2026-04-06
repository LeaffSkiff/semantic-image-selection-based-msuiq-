"""
使用示例：调用 sem_musiq 进行图像质量评估

前提：
1. 已安装依赖：pip install torch torchvision pillow
2. 可选：安装 SAM（pip install segment-anything）
3. SAM checkpoint 放在 checkpoints/sam_vit_b_01ec64.pth
"""

import torch
import sem_musiq

# ================= 最简单的用法 =================
# 一行代码完成全部流程（需要 SAM checkpoint）

model = sem_musiq.SemMUSIQFusion(
    musiq_pretrained='koniq10k',      # 使用 KonIQ-10k 预训练权重
    sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',  # SAM 模型路径
    top_k=5,                          # 使用 5 个语义 mask
    lambda_q_init=0.7,                # 初始权重：质量 70%，语义 30%
)

# 预测单张图像
result = model.predict('images/example.png')

print(f"质量分数 Q = {result['quality_score']:.4f}")
print(f"语义一致性 SIM = {result['sim_score']:.4f}")
print(f"融合分数 F = {result['final_score']:.4f}")
print(f"当前权重：Q={result['weights'][0]:.2f}, SIM={result['weights'][1]:.2f}")


# # ================= 批量预测 =================
# image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']

# for img_path in image_list:
#     result = model.predict(img_path)
#     print(f"{img_path}: Q={result['quality_score']:.2f}, F={result['final_score']:.2f}")


# # ================= 高级用法：分离调用 =================
# from PIL import Image
# import torchvision.transforms as transforms
# import numpy as np

# # 1. 单独生成语义向量
# sem_gen = sem_musiq.SemanticVectorGenerator(
#     sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
#     top_k=5,
# )

# img = Image.open('your_image.jpg').convert('RGB')
# sem_result = sem_gen(img, return_details=False)

# semantic_vectors = sem_result['semantic_vectors']  # List[np.ndarray]
# sim_score = sem_result['consistency_score']        # float (0-1)

# # 2. 单独调用 MUSIQ
# musiq = sem_musiq.MUSIQ(
#     pretrained='koniq10k',
#     use_semantic=True,
#     semantic_input_dim=5,
# )

# # 准备输入
# img_tensor = transforms.ToTensor()(img).unsqueeze(0)
# all_semantic_vectors = np.concatenate(semantic_vectors, axis=0)
# semantic_tensor = torch.from_numpy(all_semantic_vectors).float().unsqueeze(0)

# # 推理
# with torch.no_grad():
#     quality_score = musiq(img_tensor, semantic_vectors=semantic_tensor)

# print(f"质量分数 Q = {quality_score.item():.4f}")
# print(f"语义一致性 SIM = {sim_score:.4f}")

# # 3. 单独使用融合模块
# fusion = sem_musiq.QualityScoreFusion(
#     lambda_q_init=0.7,
#     lambda_sim_init=0.3,
# )

# sim_tensor = torch.tensor(sim_score).float()
# final_score = fusion(quality_score, sim_tensor)
# print(f"融合分数 F = {final_score.item():.4f}")


# # ================= 训练融合权重 =================
# # 假设你有带标注的数据集（图像路径 + ground truth MOS 分数）

# def train_example():
#     from torch.utils.data import Dataset, DataLoader

#     # 自定义数据集
#     class MyDataset(Dataset):
#         def __init__(self, image_paths, mos_scores):
#             self.paths = image_paths
#             self.mos = mos_scores

#         def __getitem__(self, idx):
#             return {
#                 'image': self.paths[idx],
#                 'mos': torch.tensor([self.mos[idx]]).float()
#             }

#         def __len__(self):
#             return len(self.paths)

#     # 准备数据
#     dataset = MyDataset(
#         image_paths=['img1.jpg', 'img2.jpg', 'img3.jpg'],
#         mos_scores=[75.0, 82.0, 68.0]  # ground truth MOS
#     )
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#     # 初始化模型
#     model = sem_musiq.SemMUSIQFusion(
#         musiq_pretrained='koniq10k',
#         sam_checkpoint='checkpoints/sam_vit_b_01ec64.pth',
#         top_k=5,
#     )

#     # 训练融合权重（冻结 MUSIQ 和 SAM）
#     model.train_fusion_only(
#         dataloader=dataloader,
#         criterion=torch.nn.MSELoss(),
#         optimizer=torch.optim.Adam(model.fusion.parameters(), lr=0.01),
#         num_epochs=10,
#     )

#     # 查看训练后的权重
#     lambda_q, lambda_sim = model.get_current_weights()
#     print(f"训练后权重：Q={lambda_q:.4f}, SIM={lambda_sim:.4f}")
