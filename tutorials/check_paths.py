"""
路径检查脚本
运行此脚本验证所有文件路径是否正确配置。
"""

import os
from pathlib import Path

print("=" * 60)
print("路径检查")
print("=" * 60)

# 获取当前目录
current_dir = Path(__file__).parent
print(f"当前目录：{current_dir}")
print()

# 检查目录结构
directories = [
    "01_patch_visualization",
    "02_sam_generation",
    "03_overlap_computation",
    "04_semantic_embedding",
    "images",
]

print("检查目录结构:")
for d in directories:
    path = current_dir / d
    exists = "[OK]" if path.exists() else "[MISSING]"
    print(f"  {exists} {d}/")

print()

# 检查文件
files = [
    ("01_patch_visualization/visualize_patches.py", "阶段 1 脚本"),
    ("01_patch_visualization/README.md", "阶段 1 说明"),
    ("02_sam_generation/run_sam.py", "阶段 2 脚本"),
    ("02_sam_generation/README.md", "阶段 2 说明"),
    ("02_sam_generation/sam_masks.pkl", "SAM mask 数据（运行后生成）"),
    ("03_overlap_computation/compute_overlap.py", "阶段 3 脚本"),
    ("03_overlap_computation/README.md", "阶段 3 说明"),
    ("images/img.png", "测试图像"),
]

print("检查文件:")
for file, desc in files:
    path = current_dir / file
    exists = "[OK]" if path.exists() else "[MISSING]"
    print(f"  {exists} {file} - {desc}")

print()

# 检查 Python 路径配置
print("检查 Python 路径配置:")
test_paths = [
    "./images/img.png",
    "./02_sam_generation/sam_masks.pkl",
]

for p in test_paths:
    path = Path(p)
    exists = "[OK]" if path.exists() else "[MISSING]"
    print(f"  {exists} {p}")

print()
print("=" * 60)
print("检查完成！")
print("=" * 60)

# 修复建议
missing = []
for file, _ in files:
    if not (current_dir / file).exists():
        missing.append(file)

if missing:
    print("\n以下文件缺失:")
    for f in missing:
        print(f"  - {f}")
    print("\n请运行相应脚本生成或手动创建这些文件。")
else:
    print("\n所有文件都存在！可以开始运行教程了。")
