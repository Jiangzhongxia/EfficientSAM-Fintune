#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
EfficientSAM 微调使用示例
演示如何使用训练好的模型进行推理
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
import json


class EfficientSAMPredictor:
    """
    EfficientSAM 推理器
    """

    def __init__(self, model_path: str, device: torch.device = torch.device('cuda')):
        """
        初始化预测器

        Args:
            model_path: 训练好的模型路径
            device: 推理设备
        """
        self.device = device
        self.model = self._load_model(model_path, device)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_path: str, device: torch.device):
        """加载训练好的模型"""
        # 构建模型
        model = build_efficient_sam_vitt()

        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        return model

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        if not image.mode == 'RGB':
            image = image.convert('RGB')

        return self.transform(image).unsqueeze(0)  # [1, 3, 1024, 1024]

    def box_to_points(self, box: tuple, original_size: tuple) -> tuple:
        """
        将box转换为point prompt格式

        Args:
            box: (x1, y1, x2, y2) 归一化坐标
            original_size: (h, w) 原始图像尺寸

        Returns:
            point_prompts, point_labels
        """
        x1, y1, x2, y2 = box

        # 归一化到[0,1]
        h_orig, w_orig = original_size

        # 生成box的四个角点
        points = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]  # 左上, 右上, 右下, 左下
        labels = [[2, 2, 3, 3]]  # 2=左上角, 3=右下角

        return torch.tensor(points, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def predict(self, image: Image.Image, boxes: list, original_size: tuple = None) -> dict:
        """
        预测masks

        Args:
            image: 输入图像
            boxes: box列表 [(x1, y1, x2, y2), ...] 归一化坐标
            original_size: 原始图像尺寸 (h, w)

        Returns:
            预测结果字典
        """
        if original_size is None:
            original_size = image.size[::-1]  # (h, w)

        # 预处理图像
        image_tensor = self.preprocess_image(image).to(self.device)

        # 转换boxes为point prompts
        point_prompts_list = []
        point_labels_list = []

        for box in boxes:
            points, labels = self.box_to_points(box, original_size)
            point_prompts_list.append(points)
            point_labels_list.append(labels)

        # 填充到固定长度
        max_objects = 10
        batch_size = 1

        point_prompts = torch.full((batch_size, max_objects, 4, 2), -1.0, dtype=torch.float32).to(self.device)
        point_labels = torch.full((batch_size, max_objects, 4), -1.0, dtype=torch.float32).to(self.device)

        for i, (points, labels) in enumerate(zip(point_prompts_list, point_labels_list)):
            if i < max_objects:
                point_prompts[0, i] = points[0]
                point_labels[0, i] = labels[0]

        # 预测
        with torch.no_grad():
            image_embeddings = self.model.get_image_embeddings(image_tensor)

            predicted_masks, predicted_iou = self.model.predict_masks(
                image_embeddings=image_embeddings,
                batched_points=point_prompts,
                batched_point_labels=point_labels,
                multimask_output=True,
                input_h=original_size[0],
                input_w=original_size[1],
                output_h=original_size[0],
                output_w=original_size[1]
            )

        # 处理输出
        results = []
        num_valid_objects = min(len(boxes), max_objects)

        for i in range(num_valid_objects):
            masks = predicted_masks[0, i]  # [num_masks, h, w]
            ious = predicted_iou[0, i]     # [num_masks]

            # 选择最佳mask
            best_idx = torch.argmax(ious).item()
            best_mask = masks[best_idx]
            best_iou = ious[best_idx].item()

            # 二值化
            binary_mask = (torch.sigmoid(best_mask) > 0.5).cpu().numpy()

            results.append({
                'mask': binary_mask,
                'iou': best_iou,
                'box': boxes[i]
            })

        return {
            'masks': results,
            'original_size': original_size
        }


def demo_usage():
    """使用示例"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化预测器
    model_path = "path/to/your/best_model.pth"
    predictor = EfficientSAMPredictor(model_path, device)

    # 加载测试图像
    image_path = "path/to/your/test_image.jpg"
    image = Image.open(image_path)

    # 定义测试boxes (归一化坐标)
    boxes = [
        (0.1, 0.1, 0.5, 0.8),   # 第一个物体
        (0.6, 0.2, 0.9, 0.7),   # 第二个物体
    ]

    # 预测
    results = predictor.predict(image, boxes)

    # 可视化结果
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")

    # 绘制boxes
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        h, w = results['original_size']
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)

        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.title("Bounding Boxes")

    # 绘制预测masks
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    for i, result in enumerate(results['masks']):
        mask = result['mask']
        colored_mask = np.zeros((*mask.shape, 4))

        # 为每个mask分配不同颜色
        colors = [(1, 0, 0, 0.3), (0, 1, 0, 0.3), (0, 0, 1, 0.3), (1, 1, 0, 0.3)]
        color = colors[i % len(colors)]

        colored_mask[mask == 1] = color
        plt.imshow(colored_mask)

        # 显示IoU
        plt.text(10, 10 + i*30, f"Mask {i+1}: IoU={result['iou']:.3f}",
                color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.title("Predicted Masks")
    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()

    print("预测完成！结果保存为 prediction_result.png")


def batch_inference_example():
    """批量推理示例"""
    import glob
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = EfficientSAMPredictor("path/to/your/best_model.pth", device)

    # 读取测试数据
    test_data = "path/to/test_data.json"  # 包含图像路径和boxes

    with open(test_data, 'r') as f:
        test_samples = json.load(f)

    results = {}

    for sample in test_samples:
        image_path = sample['image_path']
        boxes = sample['boxes']

        # 加载图像
        image = Image.open(image_path)

        # 预测
        pred_results = predictor.predict(image, boxes)
        results[image_path] = pred_results

        print(f"处理完成: {image_path}, 预测了 {len(pred_results['masks'])} 个mask")

    # 保存结果
    with open("batch_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("批量推理完成！结果保存为 batch_results.json")


if __name__ == '__main__':
    print("EfficientSAM 微调模型使用示例")
    print("1. 单张图像推理示例")
    print("2. 批量推理示例")
    print("请选择要运行的示例 (1 或 2): ")

    choice = input().strip()

    if choice == '1':
        demo_usage()
    elif choice == '2':
        batch_inference_example()
    else:
        print("无效选择！")