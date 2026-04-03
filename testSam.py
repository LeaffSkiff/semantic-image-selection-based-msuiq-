import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from PIL import Image

# 1. 加载模型
print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint="checkpoints\sam_vit_b_01ec64.pth")
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
sam.eval()
print("✓ Model loaded")

# 2. 加载图像
image = np.array(Image.open("selected_images/img.png").convert("RGB"))
print(f"Image shape: {image.shape}")

# 3. 预测
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, logits = predictor.predict()
print(f"Generated {len(masks)} masks")