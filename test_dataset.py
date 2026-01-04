#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
测试修复后的COCO数据集加载器
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from coco_dataset import COCODataset, get_coco_transforms, collate_fn


def create_test_annotation():
    """创建测试用的COCO标注文件"""
    # 创建测试标注
    test_annotation = {
        "images": [
            {
                "id": 1,
                "width": 512,
                "height": 512,
                "file_name": "test_image.jpg"
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],  # 简单正方形
                "area": 10000,
                "bbox": [100, 100, 100, 100],  # x, y, w, h
                "iscrowd": 0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[300, 300, 400, 300, 400, 400, 300, 400]],  # 另一个正方形
                "area": 10000,
                "bbox": [300, 300, 100, 100],  # x, y, w, h
                "iscrowd": 0
            }
        ]
    }

    # 保存测试标注文件
    with open('test_annotation.json', 'w') as f:
        json.dump(test_annotation, f, indent=2)

    print("✅ 创建测试标注文件: test_annotation.json")


def create_test_image():
    """创建测试用的图像文件"""
    from PIL import Image
    import numpy as np

    # 创建一个简单的测试图像
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save('test_image.jpg')

    print("✅ 创建测试图像: test_image.jpg")


def test_dataset_loading():
    """测试数据集加载"""
    print("\n🔍 测试数据集加载...")

    try:
        # 创建测试数据
        create_test_annotation()
        create_test_image()

        # 初始化数据集
        dataset = COCODataset(
            root_dir=".",  # 当前目录
            annotation_file="test_annotation.json",
            transform=get_coco_transforms(1024),
            target_size=1024,
            max_objects=10,
            iou_threshold=0.5,
            random_box_augmentation=False,
            box_noise_scale=0.0
        )

        print(f"✅ 数据集初始化成功，包含 {len(dataset)} 个样本")

        # 测试获取单个样本
        print("\n🔍 测试获取单个样本...")
        sample = dataset[0]

        # 检查样本内容
        required_keys = ['image', 'original_size', 'boxes', 'point_prompts', 'point_labels', 'masks', 'num_objects']

        for key in required_keys:
            if key not in sample:
                print(f"❌ 缺少关键键: {key}")
                return False
            else:
                print(f"✅ 找到键: {key}, 形状: {sample[key].shape if hasattr(sample[key], 'shape') else type(sample[key])}")

        # 检查数据类型和形状
        image = sample['image']
        if not isinstance(image, torch.Tensor) or image.shape != (3, 1024, 1024):
            print(f"❌ 图像格式错误: {type(image)}, 形状: {image.shape}")
            return False

        boxes = sample['boxes']
        if not isinstance(boxes, torch.Tensor):
            print(f"❌ boxes格式错误: {type(boxes)}")
            return False

        point_prompts = sample['point_prompts']
        if not isinstance(point_prompts, torch.Tensor):
            print(f"❌ point_prompts格式错误: {type(point_prompts)}")
            return False

        masks = sample['masks']
        if not isinstance(masks, torch.Tensor):
            print(f"❌ masks格式错误: {type(masks)}")
            return False

        num_objects = sample['num_objects']
        if num_objects > 0:
            print(f"✅ 检测到 {num_objects} 个有效物体")
        else:
            print("⚠️  没有检测到有效物体")

        return True

    except Exception as e:
        print(f"❌ 数据集加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader():
    """测试数据加载器"""
    print("\n🔍 测试数据加载器...")

    try:
        # 初始化数据集
        dataset = COCODataset(
            root_dir=".",
            annotation_file="test_annotation.json",
            transform=get_coco_transforms(1024),
            target_size=1024,
            max_objects=10,
            iou_threshold=0.5,
            random_box_augmentation=False,
            box_noise_scale=0.0
        )

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 使用主进程，避免多进程问题
            collate_fn=collate_fn
        )

        print("✅ 数据加载器创建成功")

        # 测试批量加载
        print("\n🔍 测试批量加载...")
        batch = next(iter(dataloader))

        # 检查batch内容
        required_keys = ['images', 'original_sizes', 'point_prompts', 'point_labels', 'masks', 'num_objects']

        for key in required_keys:
            if key not in batch:
                print(f"❌ batch缺少关键键: {key}")
                return False
            else:
                print(f"✅ batch找到键: {key}, 形状: {batch[key].shape}")

        # 检查batch数据形状
        batch_size = len(dataset)
        if batch['images'].shape[0] != batch_size:
            print(f"❌ batch size不匹配: {batch['images'].shape[0]} != {batch_size}")
            return False

        print(f"✅ 批量加载测试成功，batch size: {batch_size}")

        return True

    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """清理测试文件"""
    test_files = ['test_annotation.json', 'test_image.jpg']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ 删除测试文件: {file}")


def main():
    """主测试函数"""
    print("🚀 COCO数据集加载器测试")
    print("=" * 50)

    tests = [
        ("数据集加载", test_dataset_loading),
        ("数据加载器", test_dataloader),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            results.append(False)

    # 清理测试文件
    cleanup_test_files()

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
        print("🎉 恭喜！数据集加载器修复成功，可以开始训练！")
        return 0
    else:
        print("⚠️  存在问题，请检查错误信息")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)