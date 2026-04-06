"""
训练语义投影层和融合系数

针对 KonIQ-10k 数据集，训练以下两部分：
1. 语义投影层 (AddSemanticEmbs.semantic_proj): 将 K 维语义向量映射到 384 维
2. 融合系数 (QualityScoreFusion.lambda_q, lambda_sim): 质量分数和语义分数的融合权重

冻结部分：
- MUSIQ 骨干网络（除 semantic_proj 外）
- SAM（无参数）
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sem_musiq
from PIL import Image
import torchvision.transforms as transforms


class KonIQ10kDataset(Dataset):
    """
    KonIQ-10k 数据集加载器

    假设数据集结构：
    data/
        1024x768/          # 图像文件夹
            123456.jpg
            ...
        labels.csv         # 包含 image_name,mos 两列
    """

    def __init__(self, image_root: str, labels_csv: str, transform=None):
        """
        Args:
            image_root: 图像根目录（包含 1024x768 等子目录）
            labels_csv: labels.csv 文件路径
            transform: 图像变换（可选）
        """
        self.image_root = Path(image_root)
        self.transform = transform or transforms.ToTensor()

        # 读取 labels.csv
        self.samples = []
        with open(labels_csv, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # 跳过表头
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 6:  # KonIQ-10k 格式：image_name,mos,zscore,...
                    image_name = parts[0]
                    mos = float(parts[1])
                    self.samples.append((image_name, mos))

        print(f"加载了 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, mos = self.samples[idx]

        # KonIQ-10k 图像在 1024x768 子目录下
        img_path = self.image_root / "1024x768" / image_name
        img = Image.open(img_path).convert('RGB')

        return {
            'image': img,
            'mos': torch.tensor([mos]).float(),
            'image_name': image_name,
        }


def prepare_model(sam_checkpoint: str, top_k: int = 5, device: str = None):
    """
    准备模型，冻结不需要的参数

    Args:
        sam_checkpoint: SAM 模型权重路径
        top_k: SAM mask 数量
        device: 运行设备

    Returns:
        model, optimizer
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化融合模型
    model = sem_musiq.SemMUSIQFusion(
        musiq_pretrained='koniq10k',
        sam_checkpoint=sam_checkpoint,
        top_k=top_k,
        lambda_q_init=0.7,
        device=device,
    )

    # 冻结 MUSIQ 骨干（保留 semantic_proj）
    for name, param in model.musiq.named_parameters():
        if 'semantic_proj' not in name:
            param.requires_grad = False
        else:
            print(f"[可训练] {name} - 形状：{param.shape}")

    # 确认融合权重可训练
    for name, param in model.fusion.named_parameters():
        print(f"[可训练] fusion.{name} - 形状：{param.shape}")

    # 创建优化器：同时优化 semantic_proj 和 fusion
    optimizer = torch.optim.Adam([
        {
            'params': model.musiq.transformer_encoder.semanticembed_input.parameters(),
            'lr': 1e-3,  # 语义投影层学习率
        },
        {
            'params': model.fusion.parameters(),
            'lr': 1e-2,  # 融合系数学习率
        },
    ])

    return model, optimizer, device


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.musiq.train()  # 设为训练模式
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        img_pil = batch['image']
        mos_gt = batch['mos'].to(device)

        # 前向传播（使用模型的 predict 方法）
        result = model.predict(img_pil, return_details=False)

        # 计算损失
        pred_score = torch.tensor([result['final_score']]).to(device)
        loss = criterion(pred_score, mos_gt)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            lambda_q, lambda_sim = model.get_current_weights()
            print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, "
                  f"权重 Q={lambda_q:.3f}/SIM={lambda_sim:.3f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证"""
    model.musiq.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            img_pil = batch['image']
            mos_gt = batch['mos'].to(device)

            result = model.predict(img_pil, return_details=False)
            pred_score = torch.tensor([result['final_score']]).to(device)

            loss = criterion(pred_score, mos_gt)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    # ================= 配置区域 =================

    # 数据路径
    IMAGE_ROOT = "data/Koniq-10k"  # KonIQ-10k 图像根目录
    LABELS_CSV = "data/Koniq-10k/labels.csv"  # labels.csv 路径

    # SAM 模型路径
    SAM_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"

    # 训练配置
    TOP_K = 5              # SAM mask 数量
    BATCH_SIZE = 4         # 批量大小
    NUM_EPOCHS = 20        # 训练轮数
    VAL_SPLIT = 0.1        # 验证集比例

    # 保存路径
    SAVE_DIR = "checkpoints/semantic_fusion"
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # ===========================================

    print("=" * 50)
    print("训练语义投影层和融合系数")
    print("=" * 50)

    # 检查 SAM 权重
    if not Path(SAM_CHECKPOINT).exists():
        print(f"[警告] SAM 权重不存在：{SAM_CHECKPOINT}")
        print("请先下载 SAM 权重：wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return

    # 检查数据
    if not Path(LABELS_CSV).exists():
        print(f"[错误] labels.csv 不存在：{LABELS_CSV}")
        return

    # 准备数据集
    full_dataset = KonIQ10kDataset(IMAGE_ROOT, LABELS_CSV)

    # 划分训练集和验证集
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"训练集：{len(train_dataset)} 样本")
    print(f"验证集：{len(val_dataset)} 样本")

    # 准备模型
    model, optimizer, device = prepare_model(SAM_CHECKPOINT, TOP_K)
    criterion = nn.MSELoss()

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*50}")

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        print(f"训练损失：{train_loss:.4f}")

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        print(f"验证损失：{val_loss:.4f}")

        # 显示当前权重
        lambda_q, lambda_sim = model.get_current_weights()
        print(f"当前融合权重：Q={lambda_q:.4f}, SIM={lambda_sim:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # 保存状态字典
            checkpoint = {
                'epoch': epoch + 1,
                'semantic_proj': model.musiq.transformer_encoder.semanticembed_input.state_dict(),
                'fusion': model.fusion.state_dict(),
                'lambda_q': lambda_q,
                'lambda_sim': lambda_sim,
                'val_loss': val_loss,
            }

            save_path = f"{SAVE_DIR}/best_semantic_fusion.pth"
            torch.save(checkpoint, save_path)
            print(f"[保存] 最佳模型 -> {save_path}")

    print(f"\n训练完成！最佳验证损失：{best_val_loss:.4f}")
    print(f"最终融合权重：Q={lambda_q:.4f}, SIM={lambda_sim:.4f}")


if __name__ == '__main__':
    main()
