"""
语义 Transformer Encoder

处理真正多尺度 patch 的语义 embedding，输出语义特征矩阵。

核心设计：
    - 每个尺度独立处理：缩放图像 → 切 patch → 独立 Transformer 编码
    - 小 patch(16×16) 提取局部细节，大 patch(64×64) 提取全局语义
    - 每个尺度输出独立的特征表示，最后拼接

输入：List[[N_s, D]] - 各尺度的 SAM embedding
输出：List[[N_s, D_sem]] - 各尺度的语义嵌入矩阵
"""

import torch
import torch.nn as nn
from typing import Optional, List


class AddHashSpatialPositionEmbs(nn.Module):
    """
    Hash-based spatial positional embeddings.

    与 MUSIQ 的设计一致，但独立权重。
    """

    def __init__(self, spatial_pos_grid_size: int, dim: int):
        super().__init__()
        self.position_emb = nn.Parameter(
            torch.randn(1, spatial_pos_grid_size * spatial_pos_grid_size, dim)
        )
        nn.init.normal_(self.position_emb, std=0.02)

    def forward(self, inputs: torch.Tensor, inputs_positions: torch.Tensor) -> torch.Tensor:
        return inputs + self.position_emb.squeeze(0)[inputs_positions.long()]


class SemanticTransformerBlock(nn.Module):
    """
    Transformer block for semantic encoding.

    与 MUSIQ 的 TransformerBlock 类似，但独立权重。
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_heads: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attention = MultiHeadAttention(
            dim, num_heads, bias=True, attn_drop=attn_drop
        )
        self.drop_path = nn.Identity() if drop_path <= 0.0 else DropPath(drop_path)
        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, inputs_masks: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.attention(y, inputs_masks)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    """MLP layer for Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        bias: bool = False,
        attn_drop: float = 0.0,
        out_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_h = mask.reshape(B, 1, N, 1)
            mask_w = mask.reshape(B, 1, 1, N)
            mask2d = mask_h * mask_w
            attn = attn.masked_fill(mask2d == 0, -1e3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.out_drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ScaleSemanticTransformer(nn.Module):
    """
    单尺度的语义 Transformer

    每个尺度有独立的 Transformer，处理该尺度的 patch embedding。

    输入：[N_s, D] - 该尺度的 SAM embedding
    输出：[N_s, D_sem] - 该尺度的语义嵌入
    """

    def __init__(
        self,
        input_dim: int = 256,          # SAM ViT-B 的输出维度
        output_dim: int = 384,         # 输出维度（与 MUSIQ 对齐）
        mlp_dim: int = 1152,           # MLP 维度
        num_heads: int = 6,            # 注意力头数
        num_layers: int = 6,           # Transformer 层数
        spatial_pos_grid_size: int = 16,  # 空间位置网格大小 (支持最多 16×16=256 个 patch)
        drop: float = 0.0,             # Dropout 率
        attn_drop: float = 0.0,        # 注意力 Dropout 率
        drop_path: float = 0.0,        # DropPath 率
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 输入投影
        self.input_proj = nn.Linear(input_dim, output_dim)

        # 位置嵌入
        self.posembed_input = AddHashSpatialPositionEmbs(
            spatial_pos_grid_size, output_dim
        )

        # Dropout
        self.dropout = nn.Dropout(drop)

        # 层归一化
        self.encoder_norm = nn.LayerNorm(output_dim, eps=1e-6)

        # Transformer 层
        self.transformer = nn.ModuleList()
        for i in range(num_layers):
            self.transformer.append(
                SemanticTransformerBlock(
                    dim=output_dim,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                )
            )

    def forward(
        self,
        embeddings: torch.Tensor,           # [N_s, D]
        spatial_positions: torch.Tensor,    # [N_s]
        mask: torch.Tensor,                 # [N_s] 布尔
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            embeddings: [N_s, D] - 该尺度的 SAM embedding
            spatial_positions: [N_s] - 空间位置索引
            mask: [N_s] 布尔 - 有效 patch 的 mask

        Returns:
            output: [N_s, output_dim] - 语义嵌入
        """
        # 添加 batch 维度
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # [1, N_s, D]
            spatial_positions = spatial_positions.unsqueeze(0)  # [1, N_s]
            mask = mask.unsqueeze(0)  # [1, N_s]

        # 1. 输入投影
        x = self.input_proj(embeddings)  # [B, N_s, output_dim]

        # 2. 添加位置嵌入
        x = self.posembed_input(x, spatial_positions)

        # 3. Dropout
        x = self.dropout(x)

        # 4. Transformer 层
        for i, layer in enumerate(self.transformer):
            x = layer(x, mask)

        # 5. 层归一化
        x = self.encoder_norm(x)

        return x  # [B, N_s, output_dim]


class SemanticTransformerEncoder(nn.Module):
    """
    语义 Transformer Encoder（多尺度）

    每个尺度有独立的 Transformer，处理该尺度的 patch embedding：
    - 尺度 1 (16×16): 局部细节
    - 尺度 2 (32×32): 中等结构
    - 尺度 3 (64×64): 全局语义

    输入：List[[N_s, D]] - 各尺度的 SAM embedding
    输出：List[[N_s, D_sem]] - 各尺度的语义嵌入矩阵
    """

    def __init__(
        self,
        input_dim: int = 256,          # SAM ViT-B 的输出维度
        output_dim: int = 384,         # 输出维度（与 MUSIQ 对齐）
        mlp_dim: int = 1152,           # MLP 维度
        num_heads: int = 6,            # 注意力头数
        num_layers: int = 6,           # Transformer 层数（建议 6-14）
        num_scales: int = 3,           # 尺度数量（16, 32, 64）
        spatial_pos_grid_size: int = 16,  # 空间位置网格大小 (支持最多 16×16=256 个 patch)
        drop: float = 0.0,             # Dropout 率
        attn_drop: float = 0.0,        # 注意力 Dropout 率
        drop_path: float = 0.0,        # DropPath 率
        share_weights: bool = False,   # 是否共享权重（默认不共享）
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.share_weights = share_weights

        if share_weights:
            # 共享权重：所有尺度用同一个 Transformer
            self.shared_transformer = ScaleSemanticTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                spatial_pos_grid_size=spatial_pos_grid_size,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
            )
        else:
            # 独立权重：每个尺度有自己的 Transformer
            self.scale_transformers = nn.ModuleList()
            for _ in range(num_scales):
                self.scale_transformers.append(
                    ScaleSemanticTransformer(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        mlp_dim=mlp_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        spatial_pos_grid_size=spatial_pos_grid_size,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path,
                    )
                )

    def forward(
        self,
        sam_embeddings: List[torch.Tensor],  # List of [N_s, D]
        spatial_positions_list: List[torch.Tensor] = None,  # List of [N_s]
        masks_list: List[torch.Tensor] = None,              # List of [N_s] 布尔
    ) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            sam_embeddings: List of [N_s, D] - 各尺度的 SAM embedding
            spatial_positions_list: List of [N_s] - 各尺度的空间位置索引（可选）
            masks_list: List of [N_s] 布尔 - 各尺度的有效 patch mask（可选）

        Returns:
            semantic_embeddings: List of [N_s, output_dim] - 各尺度的语义嵌入矩阵
        """
        outputs = []

        if self.share_weights:
            # 共享权重模式
            for i, emb in enumerate(sam_embeddings):
                if spatial_positions_list is None:
                    # 自动生成位置索引
                    num_patches = emb.shape[0]
                    spatial_pos = torch.arange(num_patches).to(emb.device)
                else:
                    spatial_pos = spatial_positions_list[i]

                if masks_list is None:
                    # 默认所有 patch 有效
                    mask = torch.ones(emb.shape[0], dtype=torch.bool, device=emb.device)
                else:
                    mask = masks_list[i]

                out = self.shared_transformer(emb, spatial_pos, mask)
                outputs.append(out.squeeze(0))  # [N_s, output_dim]
        else:
            # 独立权重模式
            for i, emb in enumerate(sam_embeddings):
                if i >= self.num_scales:
                    break

                if spatial_positions_list is None:
                    # 自动生成位置索引
                    num_patches = emb.shape[0]
                    spatial_pos = torch.arange(num_patches).to(emb.device)
                else:
                    spatial_pos = spatial_positions_list[i]

                if masks_list is None:
                    # 默认所有 patch 有效
                    mask = torch.ones(emb.shape[0], dtype=torch.bool, device=emb.device)
                else:
                    mask = masks_list[i]

                out = self.scale_transformers[i](emb, spatial_pos, mask)
                outputs.append(out.squeeze(0))  # [N_s, output_dim]

        return outputs

    def get_all_embeddings(
        self,
        sam_embeddings: List[torch.Tensor],
        spatial_positions_list: List[torch.Tensor] = None,
        masks_list: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        获取拼接后的所有语义嵌入

        Args:
            sam_embeddings: List of [N_s, D] - 各尺度的 SAM embedding
            spatial_positions_list: List of [N_s] - 各尺度的空间位置索引（可选）
            masks_list: List of [N_s] 布尔 - 各尺度的有效 patch mask（可选）

        Returns:
            all_embeddings: [N_total, output_dim] - 拼接后的语义嵌入
        """
        outputs = self.forward(
            sam_embeddings, spatial_positions_list, masks_list
        )
        return torch.cat(outputs, dim=0)  # [N_total, output_dim]


# ================= 便捷函数 =================

import numpy as np


def build_semantic_embeddings(
    sam_embeddings: List[np.ndarray],  # List of [N_s, D]
    device: Optional[str] = None,
) -> dict:
    """
    便捷函数：构建语义 Transformer 输入

    Args:
        sam_embeddings: List of [N_s, D] numpy 数组 - 各尺度的 SAM embedding
        device: 目标设备，默认 CPU

    Returns:
        inputs: dict，包含：
            - sam_embeddings_tensor: List of [N_s, D]
            - spatial_positions_list: List of [N_s]
            - masks_list: List of [N_s] 布尔
    """
    # 1. 转换为 tensor 并添加 batch 维度
    sam_embeddings_tensor = []
    spatial_positions_list = []
    masks_list = []

    for emb in sam_embeddings:
        emb_tensor = torch.from_numpy(emb).float()  # [N_s, D]
        if device is not None:
            emb_tensor = emb_tensor.to(device)
        num_patches = emb_tensor.shape[0]

        # 空间位置索引
        spatial_pos = torch.arange(num_patches)
        if device is not None:
            spatial_pos = spatial_pos.to(device)

        # Mask（所有 patch 有效）
        mask = torch.ones(num_patches, dtype=torch.bool)
        if device is not None:
            mask = mask.to(device)

        sam_embeddings_tensor.append(emb_tensor)
        spatial_positions_list.append(spatial_pos)
        masks_list.append(mask)

    return {
        'sam_embeddings_tensor': sam_embeddings_tensor,
        'spatial_positions_list': spatial_positions_list,
        'masks_list': masks_list,
    }