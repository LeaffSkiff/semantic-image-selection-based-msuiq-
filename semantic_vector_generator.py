"""
语义向量生成器

功能：给定图像，自动生成语义向量（patch-mask 重叠率）和跨尺度一致性分数。

使用方式：
    # 方式 1: 使用类
    generator = SemanticVectorGenerator(sam_checkpoint="sam_vit_b.pth")

    # 直接返回结果（numpy 格式，在内存中）
    result = generator(image_path)
    s_i = result['semantic_vectors']  # List[np.ndarray] - 各尺度语义向量
    sim = result['consistency_score']  # float - 一致性分数

    # 保存到 pkl 文件
    result = generator(image_path, save_path="output/result.pkl")

    # 从 pkl 文件加载
    loaded = generator.load_from_pickle("output/result.pkl")

    # 方式 2: 使用便捷函数
    from semantic_musiq.semantic_vector_generator import generate_semantic_vectors
    result = generate_semantic_vectors("image.jpg", sam_checkpoint="sam_vit_b.pth", save_path="output/result.pkl")
"""

import torch
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

# ================= 导入依赖 =================
try:
    from segment_anything import SamPredictor, sam_model_registry
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("警告：未安装 segment_anything，SAM 功能不可用")

# 从 sem_musiq 导入多尺度处理函数
from sem_musiq.data.multiscale_trans_util import get_multiscale_patches, resize_preserve_aspect_ratio


class SemanticVectorGenerator:
    """
    语义向量生成器

    输入：图像（路径或 PIL Image）
    输出：字典，包含：
        - semantic_vectors: List[np.ndarray] - 各尺度的语义向量 [N_s, K]
        - consistency_score: float - 跨尺度语义一致性分数 (0-1)
        - similarity_matrix: np.ndarray - 尺度间相似度矩阵 [S, S]
        - masks: np.ndarray - SAM 生成的 mask [K, H, W]
        - mask_scores: np.ndarray - mask 置信度 [K]
    """

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        top_k: int = 5,
        device: Optional[str] = None,
    ):
        """
        初始化生成器

        Args:
            sam_checkpoint: SAM 模型权重路径。如果为 None，需要手动下载
            sam_model_type: SAM 模型类型，可选 "vit_b", "vit_l", "vit_h"
            top_k: 保留的 mask 数量（语义维度 K）
            device: 运行设备，默认自动选择
        """
        self.top_k = top_k
        self.patch_size = 32
        self.longer_side_lengths = [224, 384]  # 2 个缩放尺度 + 原图

        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 初始化 SAM（如果可用）
        self.predictor = None
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
            print("警告：未找到 SAM checkpoint，请手动下载放入目录下checkpoints文件夹下或指定路径")
            print("下载地址：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            return

        print(f"加载 SAM 模型：{model_type} from {checkpoint}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def _generate_sam_masks(self, image: np.ndarray) -> Dict:
        """
        使用 SAM 生成 mask

        Args:
            image: numpy array [H, W, 3], RGB 格式

        Returns:
            masks: [K, H, W]
            scores: [K]
        """
        if self.predictor is None:
            raise RuntimeError("SAM 未初始化，请检查 checkpoint 路径")

        # 设置图像
        self.predictor.set_image(image)

        # 预测 mask
        masks, scores, logits = self.predictor.predict(
            multimask_output=True,
        )

        # 筛选 Top-K
        sorted_idx = scores.argsort()[::-1][:self.top_k]
        masks = masks[sorted_idx]
        scores = scores[sorted_idx]

        return {
            'masks': masks.astype(np.float32),  # [K, H, W]
            'scores': scores.astype(np.float32),  # [K]
        }

    def _pad_semantic_vectors(
        self,
        semantic_vectors: List[np.ndarray],
        target_dim: int,
    ) -> List[np.ndarray]:
        """
        对语义向量进行 padding，确保维度不少于 target_dim。

        SAM 可能返回少于 top_k 个 mask，需要将语义向量 padding 到固定维度。

        Args:
            semantic_vectors: List[np.ndarray] - 各尺度的语义向量 [N_s, K_actual]
            target_dim: int - 目标维度（如 5）

        Returns:
            List[np.ndarray] - padding 后的语义向量 [N_s, target_dim]
        """
        padded_vectors = []
        for sv in semantic_vectors:
            if sv.shape[1] < target_dim:
                pad_width = target_dim - sv.shape[1]
                sv = np.pad(
                    sv,
                    ((0, 0), (0, pad_width)),
                    mode='constant',
                    constant_values=0
                )
            padded_vectors.append(sv)
        return padded_vectors

    def _compute_patch_mask_overlap(
        self,
        masks: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> Dict:
        """
        计算每个 patch 与每个 mask 的重叠率（与 MUSIQ 的 get_multiscale_patches 对齐）

        Args:
            masks: [K, H, W] 二值 mask
            orig_w, orig_h: 原始图像尺寸

        Returns:
            semantic_vectors: List[np.ndarray] - 各尺度的语义向量（已 padding 到方形）
            scale_info: List[Dict] - 各尺度信息
        """
        K = len(masks)
        scale_info = []
        all_semantic_vectors = []

        # 对每个尺度计算（与 MUSIQ 的 longer_side_lengths 一致）
        for i, longer_size in enumerate(self.longer_side_lengths):
            # 计算缩放后尺寸（与 MUSIQ 一致：round）
            scale = longer_size / max(orig_w, orig_h)
            rw, rh = int(round(orig_w * scale)), int(round(orig_h * scale))

            # 计算 patch 数量（与 MUSIQ 一致：向上取整）
            num_patches_h = (rh + self.patch_size - 1) // self.patch_size
            num_patches_w = (rw + self.patch_size - 1) // self.patch_size
            actual_patches = num_patches_h * num_patches_w

            # MUSIQ 风格：填充到方形 (max_side/patch_size)^2
            max_side = max(rw, rh)
            grid_size = max_side // self.patch_size
            padded_count = grid_size * grid_size

            scale_info.append({
                'name': f'Scale {i} ({longer_size})',
                'size': (rw, rh),
                'num_patches_h': num_patches_h,
                'num_patches_w': num_patches_w,
                'actual_patches': actual_patches,
                'padded_count': padded_count,
                'scale_factor': scale,
            })

            # 计算当前尺度的语义向量（实际 patch）
            semantic_vectors = []
            for patch_i in range(num_patches_h):
                for patch_j in range(num_patches_w):
                    # patch 在当前尺度的坐标
                    patch_x_scale = patch_j * self.patch_size
                    patch_y_scale = patch_i * self.patch_size

                    # 映射回原图
                    patch_x_orig = patch_x_scale / scale
                    patch_y_orig = patch_y_scale / scale
                    patch_w_orig = self.patch_size / scale
                    patch_h_orig = self.patch_size / scale

                    # 计算与每个 mask 的重叠率
                    overlap_ratios = []
                    for k in range(K):
                        # patch 在原图的坐标（整数）
                        y1 = max(0, int(round(patch_y_orig)))
                        y2 = min(orig_h, int(round(patch_y_orig + patch_h_orig)))
                        x1 = max(0, int(round(patch_x_orig)))
                        x2 = min(orig_w, int(round(patch_x_orig + patch_w_orig)))

                        # 提取 mask 区域
                        mask_region = masks[k, y1:y2, x1:x2]

                        # 重叠率
                        overlap_area = mask_region.sum()
                        patch_area = (y2 - y1) * (x2 - x1)

                        if patch_area > 0:
                            ratio = float(overlap_area) / float(patch_area)
                        else:
                            ratio = 0.0

                        overlap_ratios.append(ratio)

                    semantic_vectors.append(overlap_ratios)

            semantic_vectors = np.array(semantic_vectors, dtype=np.float32)  # [actual, K]

            # Padding 到方形（与 MUSIQ 一致）
            if actual_patches < padded_count:
                pad_width = padded_count - actual_patches
                semantic_vectors = np.pad(
                    semantic_vectors,
                    ((0, pad_width), (0, 0)),
                    mode='constant',
                    constant_values=0
                )

            all_semantic_vectors.append(semantic_vectors)

        # 添加原图尺度（与 MUSIQ 的 max_seq_len_from_original_res=-1 一致）
        # 当 max_seq_len < 0 时，MUSIQ 不进行 padding，只返回实际 patch 数量
        num_patches_h = (orig_h + self.patch_size - 1) // self.patch_size
        num_patches_w = (orig_w + self.patch_size - 1) // self.patch_size
        actual_patches = num_patches_h * num_patches_w

        scale_info.append({
            'name': 'Scale Original',
            'size': (orig_w, orig_h),
            'num_patches_h': num_patches_h,
            'num_patches_w': num_patches_w,
            'actual_patches': actual_patches,
            'padded_count': actual_patches,  # 原图尺度不 padding
            'scale_factor': 1.0,
        })

        # 计算原图尺度的语义向量（不 padding）
        semantic_vectors = []
        for patch_i in range(num_patches_h):
            for patch_j in range(num_patches_w):
                patch_x_orig = patch_j * self.patch_size
                patch_y_orig = patch_i * self.patch_size

                overlap_ratios = []
                for k in range(K):
                    y1 = max(0, patch_y_orig)
                    y2 = min(orig_h, patch_y_orig + self.patch_size)
                    x1 = max(0, patch_x_orig)
                    x2 = min(orig_w, patch_x_orig + self.patch_size)

                    mask_region = masks[k, y1:y2, x1:x2]
                    overlap_area = mask_region.sum()
                    patch_area = (y2 - y1) * (x2 - x1)

                    ratio = float(overlap_area) / float(patch_area) if patch_area > 0 else 0.0
                    overlap_ratios.append(ratio)

                semantic_vectors.append(overlap_ratios)

        semantic_vectors = np.array(semantic_vectors, dtype=np.float32)  # [actual, K]
        # 原图尺度不 padding
        all_semantic_vectors.append(semantic_vectors)

        return {
            'semantic_vectors': all_semantic_vectors,
            'scale_info': scale_info,
        }

    def _compute_consistency_score(
        self,
        semantic_vectors: List[np.ndarray],
    ) -> Dict:
        """
        计算跨尺度语义一致性分数

        Args:
            semantic_vectors: List of [N_s, K]

        Returns:
            consistency_score: float (0-1)
            similarity_matrix: [S, S]
        """
        # 1. 全局池化：对每个尺度的所有 patch 求平均
        scale_embeddings = []
        for sv in semantic_vectors:
            pooled = sv.mean(axis=0)  # [K]
            scale_embeddings.append(pooled)

        scale_embeddings = np.array(scale_embeddings)  # [S, K]
        num_scales = len(scale_embeddings)

        # 2. 计算余弦相似度矩阵
        def cosine_similarity(a, b):
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        similarity_matrix = np.zeros((num_scales, num_scales))
        for i in range(num_scales):
            for j in range(num_scales):
                similarity_matrix[i, j] = cosine_similarity(scale_embeddings[i], scale_embeddings[j])

        # 3. 计算一致性分数（非对角线的平均值）
        off_diag_sims = []
        for i in range(num_scales):
            for j in range(i + 1, num_scales):
                off_diag_sims.append(similarity_matrix[i, j])

        consistency_score = float(np.mean(off_diag_sims)) if off_diag_sims else 0.0

        return {
            'consistency_score': consistency_score,
            'similarity_matrix': similarity_matrix,
            'scale_embeddings': scale_embeddings,
        }

    def __call__(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        return_details: bool = False,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        生成语义向量和一致性分数

        Args:
            image: 输入图像
                - str/Path: 图像路径
                - PIL.Image: PIL 图像
                - np.ndarray: [H, W, 3] RGB 格式
            return_details: 是否返回详细信息（mask、相似度矩阵等）
            save_path: 可选，保存结果到 pkl 文件的路径。如果为 None，则不保存

        Returns:
            result: Dict，包含：
                - semantic_vectors: List[np.ndarray] - 各尺度语义向量
                - consistency_score: float - 一致性分数
                - [可选] masks: [K, H, W]
                - [可选] mask_scores: [K]
                - [可选] similarity_matrix: [S, S]
                - [可选] scale_info: List[Dict]
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
            img_pil = Image.fromarray(image)
        else:
            raise TypeError(f"不支持的图像类型：{type(image)}")

        orig_w, orig_h = img_pil.size
        print(f"图像尺寸：{orig_w} × {orig_h}")

        # 2. 生成 SAM mask
        print("生成 SAM mask...")
        sam_result = self._generate_sam_masks(img_np)
        masks = sam_result['masks']  # [K, H, W]
        mask_scores = sam_result['scores']  # [K]
        print(f"  生成 {len(masks)} 个 mask")

        # 3. 计算 patch-mask 重叠率
        print("计算 patch-mask 重叠率...")
        overlap_result = self._compute_patch_mask_overlap(masks, orig_w, orig_h)
        semantic_vectors = overlap_result['semantic_vectors']
        scale_info = overlap_result['scale_info']

        # 4. 检查并 padding 语义向量到 target_dim
        print(f"检查语义向量维度（target={self.top_k}）...")
        actual_k = semantic_vectors[0].shape[1]
        if actual_k < self.top_k:
            print(f"  Padding 语义向量：{actual_k} -> {self.top_k}")
            semantic_vectors = self._pad_semantic_vectors(semantic_vectors, self.top_k)
        else:
            print(f"  语义向量维度正确：K={actual_k}")

        total_patches = sum(sv.shape[0] for sv in semantic_vectors)
        print(f"  总计 {total_patches} 个 patch，语义向量维度 K={self.top_k}")

        # 4. 计算一致性分数
        print("计算跨尺度一致性分数...")
        consistency_result = self._compute_consistency_score(semantic_vectors)
        consistency_score = consistency_result['consistency_score']
        similarity_matrix = consistency_result['similarity_matrix']
        print(f"  一致性分数 SIM = {consistency_score:.4f}")

        # 5. 组装结果
        result = {
            'semantic_vectors': semantic_vectors,  # List of [N_s, K]
            'consistency_score': consistency_score,
            'scale_info': scale_info,
            'K': self.top_k,
            'num_scales': len(semantic_vectors),
        }

        if return_details:
            result['masks'] = masks
            result['mask_scores'] = mask_scores
            result['similarity_matrix'] = similarity_matrix
            result['scale_embeddings'] = consistency_result['scale_embeddings']

        # 6. 保存到文件（如果指定了路径）
        if save_path is not None:
            self.save_to_pickle(result, save_path)

        return result

    def save_to_pickle(
        self,
        result: Dict,
        output_path: Union[str, Path],
    ):
        """保存结果到 pkl 文件"""
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"结果已保存到：{output_path}")

    def load_from_pickle(
        self,
        path: Union[str, Path],
    ) -> Dict:
        """从 pkl 文件加载结果"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


# ================= 便捷函数 =================

def generate_semantic_vectors(
    image: Union[str, Path, Image.Image],
    sam_checkpoint: Optional[str] = None,
    top_k: int = 5,
    return_details: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    便捷函数：一行生成语义向量

    Args:
        image: 图像路径或 PIL 图像
        sam_checkpoint: SAM 模型路径
        top_k: mask 数量
        return_details: 是否返回详细信息
        save_path: 可选，保存结果到 pkl 文件的路径。如果为 None，则不保存

    Returns:
        result: Dict（同 SemanticVectorGenerator.__call__）

    示例：
        result = generate_semantic_vectors("test.jpg")
        s_i = result['semantic_vectors']
        sim = result['consistency_score']

        # 保存到文件
        result = generate_semantic_vectors("test.jpg", save_path="output/result.pkl")
    """
    generator = SemanticVectorGenerator(sam_checkpoint=sam_checkpoint, top_k=top_k)
    return generator(image, return_details=return_details, save_path=save_path)


# ================= CLI 入口 =================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成语义向量和跨尺度一致性分数")
    parser.add_argument("image", type=str, help="输入图像路径")
    parser.add_argument("--sam", type=str, default=None, help="SAM checkpoint 路径")
    parser.add_argument("--top-k", type=int, default=5, help="mask 数量")
    parser.add_argument("--output", type=str, default=None, help="输出 pkl 路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")

    args = parser.parse_args()

    # 生成
    result = generate_semantic_vectors(
        args.image,
        sam_checkpoint=args.sam,
        top_k=args.top_k,
        return_details=args.verbose,
        save_path=args.output,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    print(f"图像：{args.image}")
    print(f"尺度数量：{result['num_scales']}")
    print(f"语义向量维度 K: {result['K']}")
    print(f"总 patch 数：{sum(sv.shape[0] for sv in result['semantic_vectors'])}")
    print(f"跨尺度一致性分数 SIM: {result['consistency_score']:.4f}")
    if args.output:
        print(f"结果已保存到：{args.output}")
