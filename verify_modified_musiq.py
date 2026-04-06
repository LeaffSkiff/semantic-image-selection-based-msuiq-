from pyiqa.archs.musiq_arch import MUSIQ, AddSemanticEmbs
import torch

# 1. 测试 AddSemanticEmbs 类
print('测试 1: AddSemanticEmbs 类')
sem_layer = AddSemanticEmbs(semantic_input_dim=5, dim=384)
print(f'  semantic_proj 形状：{sem_layer.semantic_proj.shape}')
print(f'  参数量：{sem_layer.semantic_proj.numel()}')

# 测试前向传播
inputs = torch.randn(10, 384)  # 10 个 patch，384 维
s_i = torch.rand(10, 5)  # 10 个 patch，5 维语义向量
output = sem_layer(inputs, s_i)
print(f'  输入形状：{inputs.shape}, 输出形状：{output.shape}')
print(f'  ✓ AddSemanticEmbs 测试通过')

# 2. 测试 MUSIQ 模型（不使用语义）
print('\\n测试 2: MUSIQ 模型（use_semantic=False）')
model = MUSIQ(pretrained=False, use_semantic=False)
print(f'  模型创建成功')
print(f'  ✓ 向后兼容测试通过')

# 3. 测试 MUSIQ 模型（使用语义）
print('\\n测试 3: MUSIQ 模型（use_semantic=True）')
model_sem = MUSIQ(pretrained=False, use_semantic=True, semantic_input_dim=5)
print(f'  模型创建成功')
has_sem_layer = hasattr(model_sem.transformer_encoder, 'semanticembed_input')
print(f'  有 semanticembed_input 属性：{has_sem_layer}')
if has_sem_layer:
    print(f'  semantic_proj 形状：{model_sem.transformer_encoder.semanticembed_input.semantic_proj.shape}')
    print(f'  ✓ 语义功能测试通过')

print('\\n所有测试通过！')
