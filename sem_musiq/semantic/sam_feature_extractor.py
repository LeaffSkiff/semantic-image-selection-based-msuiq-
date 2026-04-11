"""
SAM 特征提取器
从 SAM ViT 提取语义 embedding

输入：图像路径或 PIL Image
输出：
    - sam_embeddings: List[np.ndarray] - 各尺度的 SAM embedding [N_s, D]
    - scale_info: 各尺度信息

关键设计：
    - 每个尺度独立处理：缩放图像 → 切 patch → SAM 提取特征
    - 小 patch(16×16) 提取局部细节，大 patch(64×64) 提取全局语义
    - 每个尺度输出独立的特征表示
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union

# ================= 导入依赖 =================
try:
    from segment_anything import SamPredictor, sam_model_registry
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("警告：未安装 segment_anything，SAM 功能不可用")


def resize_image_preserve_aspect_ratio(
    image: np.ndarray,
    longer_side_length: int,
) -> np.ndarray:
    """
    保持长宽比缩放图像

    Args:
        image: [H, W, 3] numpy array
        longer_side_length: 缩放后长边的长度

    Returns:
        resized_image: 缩放后的图像
    """
    h, w = image.shape[:2]
    scale = longer_side_length / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # 使用 PIL 进行缩放
    img_pil = Image.fromarray(image)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.array(img_resized)


class SAMFeatureExtractor:
    """
    SAM 特征提取器

    从 SAM ViT 提取语义 embedding，而非 mask 重叠率。

    关键设计：
    - 每个尺度独立处理：缩放 → 切 patch → SAM 提取特征
    - 小 patch(16×16) 提取局部细节，大 patch(64×64) 提取全局语义
    - 每个尺度输出独立的特征表示
    """

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        device: Optional[str] = None,
        # 多尺度配置：每个尺度对应一个图像缩放尺寸
        # patch_size=16 对应 longer_side=224
        # patch_size=32 对应 longer_side=384
        # patch_size=64 对应 longer_side=512
        scale_longer_sides: Optional[List[int]] = None,
    ):
        """
        初始化 SAMFeatureExtractor

        Args:
            sam_checkpoint: SAM 模型权重路径
            sam_model_type: SAM 模型类型，可选 "vit_b", "vit_l", "vit_h"
            device: 运行设备，默认自动选择
            scale_longer_sides: 每个尺度的长边长度，默认 [224, 384, 512]
        """
        # SAM ViT-B 的输出维度是 256
        self.embed_dim = 256

        # Patch 尺寸
        self.patch_sizes = [16, 32, 64]

        # 每个尺度对应的图像缩放尺寸（长边长度）
        if scale_longer_sides is None:
            self.scale_longer_sides = [224, 384, 512]
        else:
            self.scale_longer_sides = scale_longer_sides

        assert len(self.patch_sizes) == len(self.scale_longer_sides), \
            "patch_sizes 和 scale_longer_sides 长度必须相同"

        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 初始化 SAM
        self.predictor = None
        self.sam_model = None
        if HAS_SAM:
            self._init_sam(sam_checkpoint, sam_model_type)

    def _init_sam(self, checkpoint: Optional[str], model_type: str):
        """初始化 SAM 模型"""
        if checkpoint is None:
            # 尝试从默认路径加载
            default_paths = [
                "sam_vit_b_01ec64.pth",
                "checkpoints/sam_vit_b_01ec64.pth",
                Path.home() / ".cache" / "torch" / "hub" / "sam_vit_b_01ec64.pth",
            ]
            for p in default_paths:
                if Path(p).exists():
                    checkpoint = str(p)
                    break

        if checkpoint is None:
            print("警告：未找到 SAM checkpoint，请手动下载放入 checkpoints 文件夹")
            print("下载地址：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            return

        print(f"加载 SAM 模型：{model_type} from {checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        sam.eval()

        self.sam_model = sam
        self.predictor = SamPredictor(sam)

    @torch.no_grad()
    def _extract_patch_embeddings_from_image(
        self,
        image: np.ndarray,
        patch_size: int,
    ) -> np.ndarray:
        """
        从单个尺度的图像中提取 patch embedding

        Args:
            image: numpy array [H, W, 3], RGB 格式（已缩放到目标尺寸）
            patch_size: patch 尺寸（16, 32, 64）

        Returns:
            patch_embeddings: [N, D] N 个 patch 的 embedding
        """
        if self.predictor is None:
            raise RuntimeError("SAM 未初始化，请检查 checkpoint 路径")

        h, w = image.shape[:2]

        # 1. 整图输入 SAM，获取特征图
        self.predictor.set_image(image)
        features = self.predictor.features  # [1, D, H', W']

        # SAM ViT-B 的 stride 是 4（特征图是原图 1/4）
        sam_stride = 4

        features_np = features.cpu().numpy()[0]  # [D, H', W']
        features_np = features_np.transpose(1, 2, 0)  # [H', W', D]

        feat_h, feat_w = features_np.shape[:2]

        # 2. 基于特征图计算 patch 数量（而非原图）
        # 特征图上每个 patch 对应的区域大小
        # patch_size=16 → 特征图 4×4
        # patch_size=32 → 特征图 8×8
        # patch_size=64 → 特征图 16×16
        feat_patch_size = patch_size // sam_stride

        # 基于特征图实际尺寸计算 patch 数量
        num_patches_h = (feat_h + feat_patch_size - 1) // feat_patch_size
        num_patches_w = (feat_w + feat_patch_size - 1) // feat_patch_size

        # 3. 池化每个 patch 的特征
        patch_embeddings = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # 特征图坐标
                y1 = i * feat_patch_size
                y2 = min((i + 1) * feat_patch_size, feat_h)
                x1 = j * feat_patch_size
                x2 = min((j + 1) * feat_patch_size, feat_w)

                # 提取特征并池化
                patch_feat = features_np[y1:y2, x1:x2, :]  # [h, w, D]
                if patch_feat.size > 0:
                    patch_embedding = patch_feat.mean(axis=(0, 1))  # [D]
                    patch_embeddings.append(patch_embedding)

        return np.array(patch_embeddings)  # [N, D]

    def __call__(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_details: bool = False,
    ) -> Dict:
        """
        提取 SAM 语义 embedding（多尺度独立处理）

        Args:
            image: 输入图像
                - str/Path: 图像路径
                - PIL.Image: PIL 图像
                - np.ndarray: [H, W, 3] RGB 格式
            return_details: 是否返回详细信息

        Returns:
            result: Dict，包含：
                - sam_embeddings: List[np.ndarray] - 各尺度的 SAM embedding [N_s, D]
                - scale_info: List[Dict] - 各尺度信息
                - [可选] resized_images: 各尺度的缩放图像
        """
        # 1. 加载图像
        if isinstance(image, (str, Path)):
            img_pil = Image.open(str(image)).convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image, Image.Image):
            img_pil = image.convert("RGB")
            img_np = np.array(img_pil)
        elif isinstance(image, np.ndarray):
            img_np = image
        else:
            raise TypeError(f"不支持的图像类型：{type(image)}")

        # 2. 对每个尺度独立处理
        all_embeddings = []
        scale_info = []
        resized_images = []

        for scale_id, (patch_size, longer_side) in enumerate(
            zip(self.patch_sizes, self.scale_longer_sides)
        ):
            # 缩放图像到目标尺寸
            resized_img = resize_image_preserve_aspect_ratio(
                img_np, longer_side
            )
            resized_images.append(resized_img)

            # 提取该尺度的 patch embedding
            embeddings = self._extract_patch_embeddings_from_image(
                resized_img, patch_size
            )

            all_embeddings.append(embeddings)

            # 记录尺度信息
            h, w = resized_img.shape[:2]
            num_patches = len(embeddings)
            scale_info.append({
                'scale_id': scale_id,
                'patch_size': patch_size,
                'longer_side': longer_side,
                'resized_size': (h, w),
                'num_embeddings': num_patches,
            })

        # 3. 组装结果
        result = {
            'sam_embeddings': all_embeddings,  # List of [N_s, D]
            'scale_info': scale_info,
            'embed_dim': self.embed_dim,
            'num_scales': len(self.patch_sizes),
        }

        if return_details:
            result['resized_images'] = resized_images

        return result


# ================= 便捷函数 =================

def extract_sam_features(
    image: Union[str, Path, Image.Image],
    sam_checkpoint: Optional[str] = None,
    return_details: bool = False,
) -> Dict:
    """
    便捷函数：一行提取 SAM 特征

    Args:
        image: 图像路径或 PIL 图像
        sam_checkpoint: SAM 模型路径
        return_details: 是否返回详细信息

    Returns:
        result: Dict（同 SAMFeatureExtractor.__call__）

    示例：
        result = extract_sam_features("test.jpg")
        embeddings = result['sam_embeddings']  # List of [N_s, 256]
    """
    extractor = SAMFeatureExtractor(sam_checkpoint=sam_checkpoint)
    return extractor(image, return_details=return_details)


# ================= CLI 入口 =================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提取 SAM 语义 embedding（多尺度）")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--sam", type=str, default=None, help="SAM checkpoint 路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 提取特征
    result = extract_sam_features(
        args.image,
        sam_checkpoint=args.sam,
        return_details=args.verbose,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("SAM 特征提取结果（多尺度独立处理）")
    print("=" * 60)
    print(f"图像：{args.image}")
    print(f"尺度数量：{result['num_scales']}")
    print(f"Embedding 维度：{result['embed_dim']}")
    print()
    for i, (emb, info) in enumerate(zip(result['sam_embeddings'], result['scale_info'])):
        print(f"尺度 {i}:")
        print(f"  - patch_size: {info['patch_size']}")
        print(f"  - 图像缩放长边：{info['longer_side']}")
        print(f"  - 缩放后尺寸：{info['resized_size']}")
        print(f"  - embedding 数量：{len(emb)}")
        print(f"  - embedding 形状：{emb.shape}")
    print()
    if args.verbose:
        print(f"缩放图像数量：{len(result['resized_images'])}")