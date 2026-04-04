import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pyiqa
from PIL import Image
import matplotlib.pyplot as plt

# ====== 配置 ======
img_path = r"..\..\images\img.png"   # 改成你的图片路径
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 加载 MUSIQ 模型 ======
print("Loading MUSIQ...")
model = pyiqa.create_metric("musiq", device=device)

# ====== 评分 ======
score = model(img_path)

# ====== 显示图片 ======
img = Image.open(img_path).convert("RGB")
plt.imshow(img)
plt.axis("off")
plt.title(f"MUSIQ Score: {score.item():.4f}")
plt.show()

# ====== 输出分数 ======
print(f"MUSIQ Score: {score.item():.4f}")