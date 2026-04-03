# SAM 安装说明

## 方案 A：原版 SAM（推荐）

```bash
pip install segment-anything
```

需要从 GitHub 下载预训练权重：
https://github.com/facebookresearch/segment-anything

权重文件：
- `sam_vit_h_4b8939.pth` (最大，效果最好，~2.5GB)
- `sam_vit_l_0b3195.pth` (中等)
- `sam_vit_b_01ec64.pth` (最小，速度快)

下载后放在项目根目录。

---

## 方案 B：MobileSAM（显存有限时用）

```bash
pip install mobile-sam
```

权重文件：`mobile_sam.pt` (~40MB)

速度更快，显存占用更少，适合本地运行。

---

## 验证安装

```python
import segment_anything
print(segment_anything.__version__)
```

或者：

```python
from segment_anything import SamPredictor, sam_model_registry
print("SAM 安装成功!")
```
