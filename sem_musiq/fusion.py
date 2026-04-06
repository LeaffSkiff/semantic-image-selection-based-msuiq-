"""
端到端的图像质量评估与语义融合模块

提供 SemMUSIQFusion 类，整合 SAM 语义生成、MUSIQ 推理和可学习权重融合。
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Union, Dict, List
from pathlib import Path

from .archs.musiq_arch import MUSIQ
from .semantic.vector_generator import SemanticVectorGenerator


class QualityScoreFusion(nn.Module):
    """
    可学习的质量分数融合模块

    将 MUSIQ 质量分数 (Q) 和语义一致性分数 (SIM) 融合为最终分数：
        F = lambda_q * Q + lambda_sim * (SIM * 100)

    其中 lambda_q + lambda_sim = 1（通过 softmax 约束）
    """

    def __init__(self, lambda_q_init: float = 0.7, lambda_sim_init: float = 0.3):
        """
        初始化融合模块

        Args:
            lambda_q_init: 质量分数初始权重
            lambda_sim_init: 语义分数初始权重
        """
        super().__init__()
        # 可学习参数（原始值，通过 softmax 归一化）
        self.lambda_q_raw = nn.Parameter(torch.tensor(lambda_q_init))
        self.lambda_sim_raw = nn.Parameter(torch.tensor(lambda_sim_init))

    def forward(self, quality_score: torch.Tensor, sim_score: torch.Tensor) -> torch.Tensor:
        """
        融合分数

        Args:
            quality_score: [B] 或 scalar - MUSIQ 质量分数 (0-100)
            sim_score: [B] 或 scalar - 语义一致性分数 (0-1)

        Returns:
            final_score: 融合后的分数
        """
        # Softmax 归一化权重，确保和为 1
        weights = torch.softmax(
            torch.stack([self.lambda_q_raw, self.lambda_sim_raw]),
            dim=0
        )
        lambda_q = weights[0]
        lambda_sim = weights[1]

        # SIM 归一化到 0-100
        sim_normalized = sim_score * 100

        # 融合
        final_score = lambda_q * quality_score + lambda_sim * sim_normalized
        return final_score

    def get_weights(self) -> tuple:
        """返回当前权重值"""
        weights = torch.softmax(
            torch.stack([self.lambda_q_raw, self.lambda_sim_raw]),
            dim=0
        )
        return weights[0].item(), weights[1].item()


class SemMUSIQFusion(nn.Module):
    """
    端到端的语义感知图像质量评估模型

    整合了：
    1. SAM 语义向量生成
    2. MUSIQ 质量分数预测
    3. 可学习权重融合

    输入：图像（PIL 或路径或 tensor）
    输出：质量分数 Q、语义一致性 SIM、融合分数 F
    """

    def __init__(
        self,
        musiq_pretrained: str = 'koniq10k',
        sam_checkpoint: Optional[str] = None,
        top_k: int = 5,
        lambda_q_init: float = 0.7,
        device: Optional[str] = None,
    ):
        """
        初始化 SemMUSIQFusion

        Args:
            musiq_pretrained: MUSIQ 预训练模型名，可选 'koniq10k', 'ava', 'spaq', 'paq2piq'
            sam_checkpoint: SAM 模型权重路径。如果为 None，使用语义功能时需手动指定
            top_k: SAM 生成的 mask 数量（语义维度 K）
            lambda_q_init: 质量分数初始权重
            device: 运行设备，默认自动选择
        """
        super().__init__()

        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # 初始化 MUSIQ（启用语义嵌入）
        self.musiq = MUSIQ(
            pretrained=musiq_pretrained,
            use_semantic=True,
            semantic_input_dim=top_k,
        )
        self.musiq.to(self.device)

        # 初始化 SAM 语义向量生成器
        self.sam_generator = SemanticVectorGenerator(
            sam_checkpoint=sam_checkpoint,
            top_k=top_k,
            device=self.device,
        )

        # 可学习融合模块
        self.fusion = QualityScoreFusion(
            lambda_q_init=lambda_q_init,
            lambda_sim_init=1.0 - lambda_q_init,
        )

        # 标记是否已加载 SAM
        self.sam_ready = self.sam_generator.predictor is not None

    def forward(
        self,
        img_tensor: torch.Tensor,
        semantic_vectors: torch.Tensor,
        sim_score: torch.Tensor,
        return_all: bool = True,
    ) -> Union[torch.Tensor, Dict]:
        """
        前向传播（需要预先生成语义向量）

        Args:
            img_tensor: [B, 3, H, W] - 输入图像 tensor
            semantic_vectors: [B, N, K] - 语义向量
            sim_score: [B] 或 scalar - 语义一致性分数
            return_all: 是否返回所有分数（Q, SIM, F），默认 True

        Returns:
            如果 return_all=True: dict，包含 quality_score, sim_score, final_score
            如果 return_all=False: final_score
        """
        # MUSIQ 质量分数
        quality_score = self.musiq(
            img_tensor,
            return_mos=True,
            return_dist=False,
            semantic_vectors=semantic_vectors,
        )

        # 融合分数
        final_score = self.fusion(quality_score, sim_score)

        if return_all:
            return {
                'quality_score': quality_score,
                'sim_score': sim_score,
                'final_score': final_score,
            }
        else:
            return final_score

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_details: bool = False,
    ) -> Dict:
        """
        端到端预测（自动处理语义生成和 MUSIQ 推理）

        Args:
            image: 输入图像（路径、PIL 或 numpy）
            return_details: 是否返回详细信息（mask、相似度矩阵等）

        Returns:
            dict，包含：
                - quality_score: MUSIQ 质量分数
                - sim_score: 语义一致性分数
                - final_score: 融合分数
                - [可选] masks, similarity_matrix 等
        """
        self.musiq.eval()

        # 1. 加载图像
        if isinstance(image, (str, Path)):
            img_pil = Image.open(str(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            img_pil = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            raise TypeError(f"不支持的图像类型：{type(image)}")

        # 2. 生成语义向量
        import torchvision.transforms as transforms
        sem_result = self.sam_generator(img_pil, return_details=return_details)
        semantic_vectors_list = sem_result['semantic_vectors']
        sim_score = sem_result['consistency_score']

        # 3. 拼接各尺度语义向量
        all_semantic_vectors = np.concatenate(semantic_vectors_list, axis=0)
        semantic_tensor = torch.from_numpy(all_semantic_vectors).float()
        semantic_tensor = semantic_tensor.unsqueeze(0).to(self.device)  # [1, N, K]

        # 4. 转换为 tensor
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device)

        # 5. MUSIQ 推理
        quality_score = self.musiq(
            img_tensor,
            return_mos=True,
            return_dist=False,
            semantic_vectors=semantic_tensor,
        )

        # 6. 融合分数
        sim_tensor = torch.tensor(sim_score).float().to(self.device)
        final_score = self.fusion(quality_score, sim_tensor)

        # 7. 组装结果
        result = {
            'quality_score': quality_score.item(),
            'sim_score': sim_score,
            'final_score': final_score.item(),
            'weights': self.fusion.get_weights(),
        }

        if return_details:
            result['masks'] = sem_result.get('masks')
            result['similarity_matrix'] = sem_result.get('similarity_matrix')

        return result

    def train_fusion_only(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 10,
    ) -> List[float]:
        """
        仅训练融合权重（冻结 MUSIQ 和 SAM）

        Args:
            dataloader: 数据加载器，每个 batch 应包含：
                - image: 图像
                - mos: ground truth MOS 分数
            criterion: 损失函数
            optimizer: 优化器，默认 Adam(lr=0.01)
            num_epochs: 训练轮数

        Returns:
            每轮的平均损失列表
        """
        # 冻结 MUSIQ
        self.musiq.eval()
        for param in self.musiq.parameters():
            param.requires_grad = False

        # 仅优化融合权重
        if optimizer is None:
            optimizer = torch.optim.Adam(self.fusion.parameters(), lr=0.01)

        losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                image = batch['image']
                mos_gt = batch['mos']  # ground truth

                # 前向传播
                result = self.predict(image, return_details=False)

                # 计算损失
                loss = criterion(
                    torch.tensor(result['final_score']).to(self.device),
                    mos_gt.to(self.device)
                )

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Weights: {self.fusion.get_weights()}")

        return losses

    def get_current_weights(self) -> tuple:
        """获取当前融合权重"""
        return self.fusion.get_weights()

    def set_requires_grad(self, requires_grad: bool):
        """设置所有参数是否需要梯度"""
        for param in self.parameters():
            param.requires_grad = requires_grad
