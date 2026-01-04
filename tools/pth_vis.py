import torch

# 1. 加载你微调后的 70MB 模型
checkpoint_path = "path/to/your/finetuned_model.pth"
state_dict = torch.load(checkpoint_path, map_location='cpu')

# 2. 查看里面有什么键 (Keys)
if isinstance(state_dict, dict):
    print("文件包含的 Keys:", state_dict.keys())
    # 通常会看到: ['model', 'optimizer', 'epoch', 'args'] 等
    
    # 检查是否有优化器状态
    if 'optimizer' in state_dict or 'optimizer_state_dict' in state_dict:
        print("确认：文件中包含了优化器状态，这是导致体积变大的主要原因。")

# 3. 提取纯净的模型权重并重新保存
# 注意：根据你之前的代码，EfficientSAM 加载时可能需要 'model' 或 'model_state_dict' 键，或者直接是权重字典
save_dict = {}

if 'model_state_dict' in state_dict:
    save_dict = state_dict['model_state_dict']
elif 'model' in state_dict:
    save_dict = state_dict['model']
else:
    save_dict = state_dict # 本身就是权重字典

# 4. 保存瘦身后的模型
torch.save(save_dict, "finetuned_model_deploy.pth")
print("已保存瘦身后的模型：finetuned_model_deploy.pth")