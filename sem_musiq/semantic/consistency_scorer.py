"""
语义一致性打分器

计算同尺度内 patch 之间的余弦相似度，输出语义一致性分数 W₂。

输入：List[[N_s, D_sem]] - 各尺度的语义嵌入
输出：W₂ (0-1) - 语义一致性分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SemanticConsistencyScorer(nn.Module):
    """
    语义一致性打分器

    计算同尺度内 patch 之间的余弦相似度，作为语义一致性分数。

    设计原理：
    - 同一尺度内，语义相似的 patch 应该有较高的相似度
    - 尺度内 patch 相似度越高，说明该尺度的语义一致性越好
    - 各尺度的一致性分数取平均，得到最终的 W₂

    输入：List[[N_s, D_sem]] - 各尺度的语义嵌入
    输出：W₂ (0-1) - 语义一致性分数
    """

    def __init__(
        self,
        num_scales: int = 3,
        similarity_type: str = 'mean',  # 'mean' 或 'min'
        temperature: float = 1.0,
    ):
        """
        初始化一致性打分器

        Args:
            num_scales: 尺度数量（默认 3：16×16, 32×32, 64×64）
            similarity_type: 相似度聚合方式
                - 'mean': 所有 patch 对的平均相似度
                - 'min': 最不相似 patch 对的相似度（保守策略）
            temperature: 温度参数，用于调节相似度分布
        """
        super().__init__()
        self.num_scales = num_scales
        self.similarity_type = similarity_type
        self.temperature = temperature

    def _compute_pairwise_cosine_similarity(
        self,
        embeddings: torch.Tensor,  # [N, D]
    ) -> torch.Tensor:
        """
        计算两两之间的余弦相似度

        Args:
            embeddings: [N, D] - N 个 patch 的嵌入

        Returns:
            similarity_matrix: [N, N] - 两两相似度矩阵
        """
        # L2 归一化
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [N, D]

        # 计算余弦相似度矩阵
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())  # [N, N]

        return similarity_matrix

    def _compute_scale_consistency(
        self,
        embeddings: torch.Tensor,  # [N, D]
    ) -> torch.Tensor:
        """
        计算单个尺度的语义一致性分数

        Args:
            embeddings: [N, D] - 该尺度的语义嵌入

        Returns:
            consistency_score: scalar - 该尺度的一致性分数
        """
        N = embeddings.shape[0]

        if N == 1:
            # 只有一个 patch，无法计算一致性，返回 1.0
            return torch.tensor(1.0, device=embeddings.device)

        # 计算相似度矩阵
        similarity_matrix = self._compute_pairwise_cosine_similarity(embeddings)

        if self.similarity_type == 'mean':
            # 非对角线元素的平均值（排除自相似）
            # 创建一个 mask 排除对角线
            mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
            off_diag_similarities = similarity_matrix[mask]
            consistency_score = off_diag_similarities.mean()

        elif self.similarity_type == 'min':
            # 非对角线元素的最小值（保守策略）
            mask = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
            off_diag_similarities = similarity_matrix[mask]
            consistency_score = off_diag_similarities.min()

        else:
            raise ValueError(f"未知的 similarity_type: {self.similarity_type}")

        return consistency_score

    def forward(
        self,
        semantic_embeddings: List[torch.Tensor],  # List of [N_s, D_sem]
        return_per_scale: bool = False,
    ) -> torch.Tensor:
        """
        计算语义一致性分数 W₂

        Args:
            semantic_embeddings: List of [N_s, D_sem] - 各尺度的语义嵌入
            return_per_scale: 是否返回各尺度的分数

        Returns:
            w2_score: scalar - 语义一致性分数 (0-1)
            [可选] per_scale_scores: List[scalar] - 各尺度的一致性分数
        """
        per_scale_scores = []

        for i, emb in enumerate(semantic_embeddings):
            if emb.dim() == 3:
                # 如果有 batch 维度，取第一个
                emb = emb[0]  # [N_s, D_sem]

            score = self._compute_scale_consistency(emb)
            per_scale_scores.append(score)

        # 各尺度的一致性分数取平均
        w2_score = torch.stack(per_scale_scores).mean()

        # 通过 temperature 调节（可选）
        # w2_score = w2_score / self.temperature

        if return_per_scale:
            return w2_score, per_scale_scores
        else:
            return w2_score


class LearnableConsistencyScorer(nn.Module):
    """
    可学习的语义一致性打分器

    在基础一致性分数上添加可学习参数，让模型自适应调整。
    """

    def __init__(
        self,
        num_scales: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 0.0,
    ):
        """
        初始化可学习打分器

        Args:
            num_scales: 尺度数量
            init_alpha: 缩放参数初始值
            init_beta: 偏移参数初始值
        """
        super().__init__()
        self.num_scales = num_scales

        # 可学习参数
        self.alpha = nn.Parameter(torch.tensor(init_alpha))  # 缩放
        self.beta = nn.Parameter(torch.tensor(init_beta))    # 偏移

        # 基础打分器
        self.base_scorer = SemanticConsistencyScorer(num_scales=num_scales)

    def forward(
        self,
        semantic_embeddings: List[torch.Tensor],
        return_per_scale: bool = False,
    ) -> torch.Tensor:
        """
        计算可学习的语义一致性分数

        公式：W₂ = sigmoid(alpha * base_score + beta)

        Args:
            semantic_embeddings: List of [N_s, D_sem] - 各尺度的语义嵌入
            return_per_scale: 是否返回各尺度的分数

        Returns:
            w2_score: scalar - 语义一致性分数 (0-1)
        """
        # 基础一致性分数
        base_score = self.base_scorer(semantic_embeddings)

        # 可学习变换 + sigmoid 确保在 (0, 1) 范围内
        w2_score = torch.sigmoid(self.alpha * base_score + self.beta)

        return w2_score


# ================= 便捷函数 =================

def compute_semantic_consistency(
    semantic_embeddings: List[torch.Tensor],
    similarity_type: str = 'mean',
) -> float:
    """
    便捷函数：计算语义一致性分数

    Args:
        semantic_embeddings: List of [N_s, D_sem] - 各尺度的语义嵌入
        similarity_type: 相似度聚合方式

    Returns:
        w2_score: float - 语义一致性分数 (0-1)
    """
    scorer = SemanticConsistencyScorer(similarity_type=similarity_type)
    w2_score = scorer(semantic_embeddings)
    return w2_score.item()