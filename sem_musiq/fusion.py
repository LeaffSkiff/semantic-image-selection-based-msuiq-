"""
双分支融合模块

整合语义分支和 MUSIQ 主干，实现端到端的图像质量评估。

架构：
    输入图像
        ├→ 语义分支 → W₂ (语义一致性) + 语义嵌入
        └→ MUSIQ 分支 → W₁ (空间质量)
        → 融合模块 → F = λ_q × W₁ + λ_sim × W₂ × 100

注意：
    - 语义分支的 patch 数量与 MUSIQ 的 patch 数量不同
    - 需要将语义嵌入投影并匹配到 MUSIQ 的 patch 网格
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Union, Dict, List
from pathlib import Path

from .archs.musiq_arch import MUSIQ
from .semantic.sam_feature_extractor import SAMFeatureExtractor
from .transformer.semantic_transformer import (
    SemanticTransformerEncoder,
    build_semantic_embeddings,
)
from .semantic.consistency_scorer import SemanticConsistencyScorer


class QualityScoreFusion(nn.Module):
    """
    可学习的质量分数融合模块

    将 MUSIQ 质量分数 (W₁) 和语义一致性分数 (W₂) 融合为最终分数：
        F = λ_q × W₁ + λ_sim × W₂ × 100

    其中 λ_q + λ_sim = 1（通过 softmax 约束）
    """

    def __init__(self, lambda_q_init: float = 0.7, lambda_sim_init: float = 0.3):
        super().__init__()
        # 可学习参数（原始值，通过 softmax 归一化）
        self.lambda_q_raw = nn.Parameter(torch.tensor(lambda_q_init))
        self.lambda_sim_raw = nn.Parameter(torch.tensor(lambda_sim_init))

    def forward(
        self,
        quality_score: torch.Tensor,  # W₁ (0-100)
        sim_score: torch.Tensor,      # W₂ (0-1)
    ) -> torch.Tensor:
        # Softmax 归一化权重
        weights = torch.softmax(
            torch.stack([self.lambda_q_raw, self.lambda_sim_raw]),
            dim=0
        )
        lambda_q = weights[0]
        lambda_sim = weights[1]

        # W₂ 归一化到 0-100
        sim_normalized = sim_score * 100

        # 融合
        final_score = lambda_q * quality_score + lambda_sim * sim_normalized
        return final_score

    def get_weights(self) -> tuple:
        weights = torch.softmax(
            torch.stack([self.lambda_q_raw, self.lambda_sim_raw]),
            dim=0
        )
        return weights[0].item(), weights[1].item()


class SemMUSIQFusion(nn.Module):
    """
    端到端的语义感知图像质量评估模型（双分支架构）

    分支 1（语义分支）：
        - SAMFeatureExtractor：提取多尺度 SAM 特征
        - SemanticTransformerEncoder：编码语义特征
        - SemanticConsistencyScorer：计算 W₂

    分支 2（MUSIQ 主干）：
        - MUSIQ：预测空间质量 W₁

    融合：
        - F = λ_q × W₁ + λ_sim × W₂ × 100

    注意：
        - 当前设计：两条分支独立工作，然后融合
        - 语义嵌入 cat 拼接功能待后续实现
    """

    def __init__(
        self,
        musiq_pretrained: str = 'koniq10k',
        sam_checkpoint: Optional[str] = None,
        # 语义分支配置
        semantic_transformer_layers: int = 6,
        semantic_output_dim: int = 384,
        # 融合配置
        lambda_q_init: float = 0.7,
        # 设备配置
        device: Optional[str] = None,
    ):
        """
        初始化 SemMUSIQFusion

        Args:
            musiq_pretrained: MUSIQ 预训练模型名
            sam_checkpoint: SAM 模型权重路径
            semantic_transformer_layers: 语义 Transformer 层数（建议 6-14）
            semantic_output_dim: 语义嵌入输出维度（默认 384，与 MUSIQ 对齐）
            lambda_q_init: 质量分数初始权重
            device: 运行设备，默认自动选择
        """
        super().__init__()

        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # ================= 分支 1：语义分支 =================
        self.sam_extractor = SAMFeatureExtractor(
            sam_checkpoint=sam_checkpoint,
            device=self.device,
        )

        self.semantic_transformer = SemanticTransformerEncoder(
            input_dim=256,  # SAM ViT-B 输出维度
            output_dim=semantic_output_dim,
            num_layers=semantic_transformer_layers,
            num_scales=3,  # 16×16, 32×32, 64×64
            share_weights=False,  # 每个尺度独立权重
        )

        self.consistency_scorer = SemanticConsistencyScorer(num_scales=3)

        # ================= 分支 2：MUSIQ 主干 =================
        # 注意：暂时不启用语义嵌入，让 MUSIQ 独立工作
        self.musiq = MUSIQ(
            pretrained=musiq_pretrained,
            use_semantic=False,  # 暂时不启用
        )
        self.musiq.to(self.device)

        # ================= 融合模块 =================
        self.fusion = QualityScoreFusion(
            lambda_q_init=lambda_q_init,
            lambda_sim_init=1.0 - lambda_q_init,
        )

        # 标记 SAM 是否就绪
        self.sam_ready = self.sam_extractor.predictor is not None

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_details: bool = False,
    ) -> Dict:
        """
        端到端预测（自动处理双分支）

        Args:
            image: 输入图像
            return_details: 是否返回详细信息

        Returns:
            dict，包含 W₁, W₂, F 和权重
        """
        self.eval()

        # 1. 加载图像
        if isinstance(image, (str, Path)):
            img_pil = Image.open(str(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            img_pil = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image)
        else:
            raise TypeError(f"不支持的图像类型：{type(image)}")

        import torchvision.transforms as transforms

        # ================= 分支 1：语义分支 =================
        # 2. SAM 特征提取
        sam_result = self.sam_extractor(img_pil, return_details=False)
        sam_embeddings = sam_result['sam_embeddings']  # List of [N_s, 256]

        # 3. 转换为 tensor
        sam_embeddings_tensor = [
            torch.from_numpy(emb).float().to(self.device)
            for emb in sam_embeddings
        ]

        # 4. 构建 Transformer 输入
        transformer_inputs = build_semantic_embeddings(sam_embeddings)

        # 5. 语义 Transformer 编码
        semantic_outputs = self.semantic_transformer(
            transformer_inputs['sam_embeddings_tensor'],
            transformer_inputs['spatial_positions_list'],
            transformer_inputs['masks_list'],
        )  # List of [N_s, 384]

        # 6. 计算语义一致性分数 W₂
        w2_score = self.consistency_scorer(semantic_outputs)

        # 7. 准备 MUSIQ 输入
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(self.device)

        # ================= 分支 2：MUSIQ 主干 =================
        # 8. MUSIQ 预测 W₁（不使用语义嵌入）
        w1_score = self.musiq(
            img_tensor,
            return_mos=True,
            return_dist=False,
        )

        # ================= 融合 =================
        w2_tensor = torch.tensor(w2_score).float().to(self.device)
        final_score = self.fusion(w1_score, w2_tensor)

        # ================= 组装结果 =================
        result = {
            'quality_score': w1_score.item(),  # W₁ (0-100)
            'sim_score': w2_score,             # W₂ (0-1)
            'final_score': final_score.item(), # F
            'weights': self.fusion.get_weights(),
        }

        if return_details:
            result['semantic_embeddings'] = semantic_outputs
            result['scale_info'] = sam_result['scale_info']

        return result

    def train_fusion_only(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module = nn.MSELoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        num_epochs: int = 10,
    ) -> List[float]:
        """
        仅训练融合权重（冻结其他部分）

        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            optimizer: 优化器
            num_epochs: 训练轮数

        Returns:
            每轮的平均损失列表
        """
        # 冻结所有部分
        self.sam_extractor.eval()
        for param in self.sam_extractor.parameters():
            param.requires_grad = False

        self.semantic_transformer.eval()
        for param in self.semantic_transformer.parameters():
            param.requires_grad = False

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
                mos_gt = batch['mos']

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