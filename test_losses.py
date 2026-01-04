#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
测试损失函数
"""

import torch
import numpy as np
from losses import SegmentationLosses


def test_loss_functions():
    """测试损失函数"""
    print("🔍 测试损失函数...")

    try:
        # 初始化损失函数
        seg_losses = SegmentationLosses()

        # 创建测试数据
        batch_size = 2
        height = 64
        width = 64
        max_objects = 3

        # 模拟预测logits [B, max_queries, num_masks, H, W]
        pred_logits = torch.randn(batch_size, 1, max_objects, height, width)

        # 模拟目标masks [B, max_objects, H, W]
        target_masks = torch.randint(0, 2, (batch_size, max_objects, height, width)).float()

        # 模拟物体数量
        num_objects = torch.tensor([2, 1])  # 第一张图片2个物体，第二张1个物体

        print(f"测试数据形状:")
        print(f"  pred_logits: {pred_logits.shape}")
        print(f"  target_masks: {target_masks.shape}")
        print(f"  num_objects: {num_objects}")

        # 测试损失计算
        losses = seg_losses.compute_loss(pred_logits, target_masks, num_objects)

        print(f"计算结果:")
        for loss_name, loss_value in losses.items():
            print(f"  {loss_name}: {loss_value.item():.4f}")

        # 检查损失值是否合理
        for loss_name, loss_value in losses.items():
            if not torch.is_tensor(loss_value):
                print(f"❌ {loss_name} 不是张量: {type(loss_value)}")
                return False
            if torch.isnan(loss_value):
                print(f"❌ {loss_name} 是NaN")
                return False
            if loss_value < 0:
                print(f"❌ {loss_name} 是负值: {loss_value.item()}")
                return False

        print("✅ 损失函数测试通过")
        return True

    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_losses():
    """测试单个损失函数"""
    print("\n🔍 测试单个损失函数...")

    try:
        seg_losses = SegmentationLosses()

        # 创建简单的测试数据
        pred = torch.randn(64, 64)  # [H, W]
        target = torch.randint(0, 2, (64, 64)).float()  # [H, W]

        print(f"测试数据形状: pred {pred.shape}, target {target.shape}")

        # 测试各个损失函数
        focal_loss = seg_losses.focal_loss(pred, target)
        dice_loss = seg_losses.dice_loss(pred, target)
        iou_loss = seg_losses.iou_loss(pred, target)
        boundary_loss = seg_losses.boundary_loss(pred, target)

        print(f"单个损失结果:")
        print(f"  focal_loss: {focal_loss.item():.4f}")
        print(f"  dice_loss: {dice_loss.item():.4f}")
        print(f"  iou_loss: {iou_loss.item():.4f}")
        print(f"  boundary_loss: {boundary_loss.item():.4f}")

        # 检查损失值
        all_losses = [focal_loss, dice_loss, iou_loss, boundary_loss]
        for i, loss in enumerate(all_losses):
            loss_names = ['focal_loss', 'dice_loss', 'iou_loss', 'boundary_loss']
            if not torch.is_tensor(loss):
                print(f"❌ {loss_names[i]} 不是张量")
                return False
            if torch.isnan(loss):
                print(f"❌ {loss_names[i]} 是NaN")
                return False
            if loss < 0:
                print(f"❌ {loss_names[i]} 是负值")
                return False

        print("✅ 单个损失函数测试通过")
        return True

    except Exception as e:
        print(f"❌ 单个损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🚀 损失函数测试")
    print("=" * 50)

    tests = [
        ("单个损失函数", test_individual_losses),
        ("完整损失计算", test_loss_functions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 运行测试: {test_name}")
            result = test_func()
            results.append(result)
            print(f"✅ {test_name}: {'通过' if result else '失败'}")
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            import traceback
            traceback.print_exc()
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
        print("🎉 损失函数修复成功！")
        return 0
    else:
        print("⚠️  存在问题，请检查错误信息")
        return 1


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)