#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
测试脚本 - 验证微调环境是否正确配置
"""

import sys
import torch
import torchvision
from PIL import Image
import numpy as np


def test_imports():
    """测试依赖导入"""
    print("🔍 测试依赖导入...")

    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False

    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision导入失败: {e}")
        return False

    try:
        from pycocotools.coco import COCO
        print("✅ pycocotools: 导入成功")
    except ImportError as e:
        print(f"❌ pycocotools导入失败: {e}")
        print("💡 请安装: pip install pycocotools")
        return False

    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard: 导入成功")
    except ImportError as e:
        print(f"❌ TensorBoard导入失败: {e}")
        print("💡 请安装: pip install tensorboard")
        return False

    return True


def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")

    try:
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
        print("✅ EfficientSAM模型模块: 导入成功")
    except ImportError as e:
        print(f"❌ EfficientSAM模块导入失败: {e}")
        return False

    try:
        # 测试模型构建
        print("🏗️  构建EfficientSAM-Ti模型...")
        model_vitt = build_efficient_sam_vitt()
        print(f"✅ EfficientSAM-Ti: 构建成功 (参数量: {sum(p.numel() for p in model_vitt.parameters()):,})")

        print("🏗️  构建EfficientSAM-S模型...")
        model_vits = build_efficient_sam_vits()
        print(f"✅ EfficientSAM-S: 构建成功 (参数量: {sum(p.numel() for p in model_vits.parameters()):,})")

    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        return False

    return True


def test_data_loading():
    """测试数据加载器"""
    print("\n🔍 测试数据加载器...")

    try:
        from coco_dataset import COCODataset, get_coco_transforms
        print("✅ 数据集模块: 导入成功")
    except ImportError as e:
        print(f"❌ 数据集模块导入失败: {e}")
        return False

    try:
        # 测试数据变换
        transform = get_coco_transforms(1024)
        test_image = Image.new('RGB', (512, 512), color='red')
        transformed = transform(test_image)
        print(f"✅ 数据变换: 成功 (输出形状: {transformed.shape})")

    except Exception as e:
        print(f"❌ 数据变换失败: {e}")
        return False

    return True


def test_loss_functions():
    """测试损失函数"""
    print("\n🔍 测试损失函数...")

    try:
        from losses import SegmentationLosses, KnowledgeDistillation
        print("✅ 损失函数模块: 导入成功")
    except ImportError as e:
        print(f"❌ 损失函数模块导入失败: {e}")
        return False

    try:
        # 测试分割损失
        seg_losses = SegmentationLosses()
        pred_logits = torch.randn(1, 1024, 1024)
        target_masks = torch.randint(0, 2, (1, 1024, 1024)).float()

        losses = seg_losses.compute_loss(
            pred_logits.unsqueeze(0).unsqueeze(0),  # [1, 1, 1, 1024, 1024]
            target_masks.unsqueeze(0),              # [1, 1, 1024, 1024]
            torch.tensor([1])
        )

        print(f"✅ 分割损失: 计算成功 (总损失: {losses['total_loss']:.4f})")

        # 测试知识蒸馏
        kd = KnowledgeDistillation()
        print("✅ 知识蒸馏: 初始化成功")

    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False

    return True


def test_gpu_availability():
    """测试GPU可用性"""
    print("\n🔍 测试GPU可用性...")

    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA不可用，将使用CPU训练")
        print("💡 如需GPU加速，请安装CUDA版本的PyTorch")
        return False


def test_model_forward():
    """测试模型前向传播"""
    print("\n🔍 测试模型前向传播...")

    try:
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

        # 构建模型
        model = build_efficient_sam_vitt()
        model.eval()

        # 创建测试数据
        batch_size = 1
        images = torch.randn(batch_size, 3, 1024, 1024)
        point_prompts = torch.tensor([[[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]]], dtype=torch.float32)
        point_labels = torch.tensor([[[2, 2, 3, 3]]], dtype=torch.float32)

        # 前向传播
        with torch.no_grad():
            masks, ious = model(images, point_prompts, point_labels)

        print(f"✅ 前向传播: 成功")
        print(f"   - 输出masks形状: {masks.shape}")
        print(f"   - 输出IoUs形状: {ious.shape}")

    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False

    return True


def main():
    """主测试函数"""
    print("🚀 EfficientSAM 微调环境测试")
    print("=" * 50)

    tests = [
        ("依赖导入", test_imports),
        ("模型加载", test_model_loading),
        ("数据加载", test_data_loading),
        ("损失函数", test_loss_functions),
        ("GPU可用性", test_gpu_availability),
        ("模型推理", test_model_forward),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            results.append(False)

    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果总结:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ 通过" if results[i] else "❌ 失败"
        print(f"   {test_name}: {status}")

    passed = sum(results)
    total = len(results)
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 恭喜！环境配置完美，可以开始微调训练！")
        return 0
    else:
        print("⚠️  存在配置问题，请根据上述信息进行修复")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)