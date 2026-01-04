# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import random


class COCODataset(Dataset):
    """
    COCO格式数据集加载器，支持PCB分割任务
    专为EfficientSAM微调设计，使用box prompt
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[transforms.Compose] = None,
        target_size: int = 1024,
        max_objects: int = 10,
        iou_threshold: float = 0.5,
        random_box_augmentation: bool = True,
        box_noise_scale: float = 0.1,
    ):
        """
        初始化COCO数据集

        Args:
            root_dir: 图像根目录
            annotation_file: COCO标注文件路径
            transform: 图像变换
            target_size: 目标图像尺寸
            max_objects: 每张图像最大物体数量
            iou_threshold: IoU阈值用于过滤低质量标注
            random_box_augmentation: 是否对box进行随机增强
            box_noise_scale: box噪声缩放因子
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.target_size = target_size
        self.max_objects = max_objects
        self.iou_threshold = iou_threshold
        self.random_box_augmentation = random_box_augmentation
        self.box_noise_scale = box_noise_scale

        # 获取所有图像ID
        self.image_ids = list(self.coco.imgs.keys())

        # 过滤有标注的图像
        self.image_ids = [
            img_id for img_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]

        print(f"加载了 {len(self.image_ids)} 张有效图像")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本

        Returns:
            dict: 包含图像、box prompt、mask等信息的字典
        """
        img_id = self.image_ids[idx]

        # 获取图像信息
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        original_size = image.size[::-1]  # (H, W)

        # 获取标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # 过滤低质量标注
        annotations = [
            ann for ann in annotations
            if ann.get('area', 0) > 100 and ann.get('iscrowd', 0) == 0
        ]

        # 限制物体数量
        if len(annotations) > self.max_objects:
            annotations = random.sample(annotations, self.max_objects)

        # 生成box prompt和target masks
        boxes, masks = self._process_annotations(annotations, original_size)

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        # 对box进行随机增强
        if self.random_box_augmentation and len(boxes) > 0:
            boxes = self._augment_boxes(boxes, original_size)

        # 将box转换为point prompt格式
        # EfficientSAM期望的格式：[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]]
        # 其中 [x1,y1] = 左上角, [x2,y2] = 右上角, [x3,y3] = 右下角, [x4,y4] = 左下角
        point_prompts, point_labels = self._boxes_to_points(boxes)

        # 填充到固定长度
        point_prompts, point_labels = self._pad_prompts(point_prompts, point_labels)

        # 调整masks尺寸
        masks = self._resize_masks(masks, original_size)

        # 调试信息（可选）
        # if len(masks) > 0:
        #     print(f"Debug: Processed {len(masks)} masks, shapes: {[m.shape for m in masks]}")

        return {
            'image': image,
            'original_size': torch.tensor(original_size, dtype=torch.float32),
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4)),
            'point_prompts': torch.tensor(np.array(point_prompts), dtype=torch.float32),
            'point_labels': torch.tensor(np.array(point_labels), dtype=torch.float32),
            'masks': torch.tensor(np.array(masks), dtype=torch.float32),
            'num_objects': len(boxes)
        }

    def _process_annotations(self, annotations: List[Dict], original_size: Tuple[int, int]) -> Tuple[List, List]:
        """
        处理标注信息，生成box和mask

        Args:
            annotations: 标注列表
            original_size: 原始图像尺寸 (H, W)

        Returns:
            boxes: 归一化的box列表 [[x1,y1,x2,y2], ...]
            masks: 二值mask列表
        """
        boxes = []
        masks = []
        h_orig, w_orig = original_size

        for ann in annotations:
            try:
                # 跳过无效标注
                if 'bbox' not in ann or 'segmentation' not in ann:
                    continue

                # 跳过crowd标注
                if ann.get('iscrowd', 0) == 1:
                    continue

                # 获取box (x1, y1, w, h) -> (x1, y1, x2, y2)
                bbox = ann['bbox']
                if len(bbox) != 4:
                    continue

                x1, y1, w, h = bbox

                # 验证box有效性
                if w <= 0 or h <= 0:
                    continue

                x2, y2 = x1 + w, y1 + h

                # 确保box在图像范围内
                x1, x2 = max(0, x1), min(w_orig - 1, x2)
                y1, y2 = max(0, y1), min(h_orig - 1, y2)

                if x1 >= x2 or y1 >= y2:
                    continue

                # 归一化box坐标
                box = [
                    x1 / w_orig,
                    y1 / h_orig,
                    x2 / w_orig,
                    y2 / h_orig
                ]

                # 获取mask
                segmentation = ann['segmentation']
                if not segmentation:
                    continue

                try:
                    if isinstance(segmentation, list):
                        # 多边形格式
                        if len(segmentation) == 0:
                            continue
                        rle = mask_utils.frPyObjects(segmentation, h_orig, w_orig)
                    else:
                        # RLE格式
                        rle = segmentation

                    mask = mask_utils.decode(rle)

                    # 验证mask
                    if mask is None or mask.size == 0:
                        continue

                    # 确保mask是二值的
                    if mask.dtype != bool:
                        mask = mask > 0.5

                    # 检查mask质量：确保mask有效且不是空的
                    mask_area = mask.sum()
                    if mask_area < 50:  # 最小面积阈值，过滤太小的mask
                        continue

                    # 计算bbox面积
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area <= 0:
                        continue

                    # 计算mask覆盖bbox的比例
                    coverage_ratio = mask_area / bbox_area
                    if coverage_ratio < 0.2:  # 最小覆盖比例
                        continue

                    # 检查mask是否在bbox内
                    mask_bbox = mask_utils.toBbox(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
                    if len(mask_bbox) >= 4:
                        mask_x1, mask_y1, mask_w, mask_h = mask_bbox[:4]
                        # 计算与标注bbox的重叠度
                        overlap_x1 = max(x1, mask_x1)
                        overlap_y1 = max(y1, mask_y1)
                        overlap_x2 = min(x2, mask_x1 + mask_w)
                        overlap_y2 = min(y2, mask_y1 + mask_h)

                        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                            overlap_ratio = overlap_area / bbox_area
                            if overlap_ratio < 0.1:  # 重叠度过低
                                continue

                    # 所有检查通过，添加到结果
                    boxes.append(box)
                    masks.append(mask.astype(np.float32))

                except Exception as e:
                    # 记录错误但继续处理其他标注
                    print(f"Warning: Failed to process annotation: {e}")
                    continue

            except Exception as e:
                # 记录错误但继续处理其他标注
                print(f"Warning: Failed to process annotation: {e}")
                continue

        return boxes, masks

    def _augment_boxes(self, boxes: List, original_size: Tuple[int, int]) -> List:
        """
        对box进行随机增强

        Args:
            boxes: box列表
            original_size: 原始图像尺寸

        Returns:
            增强后的box列表
        """
        augmented_boxes = []
        h_orig, w_orig = original_size

        for box in boxes:
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = x2 + h

            # 添加随机噪声
            noise_x = (torch.rand(1).item() - 0.5) * 2 * self.box_noise_scale
            noise_y = (torch.rand(1).item() - 0.5) * 2 * self.box_noise_scale

            # 计算box宽高
            w_box = x2 - x1
            h_box = y2 - y1

            # 应用噪声
            x1 = max(0, min(1, x1 + noise_x * w_box))
            y1 = max(0, min(1, y1 + noise_y * h_box))
            x2 = max(0, min(1, x2 + noise_x * w_box))
            y2 = max(0, min(1, y2 + noise_y * h_box))

            # 确保box有效
            if x2 > x1 and y2 > y1:
                augmented_boxes.append([x1, y1, x2, y2])

        return augmented_boxes

    def _boxes_to_points(self, boxes: List) -> Tuple[List, List]:
        """
        将box转换为point prompt格式

        Args:
            boxes: box列表 [[x1,y1,x2,y2], ...]

        Returns:
            point_prompts: point坐标列表 [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...]
            point_labels: point标签列表 [[2,3,2,3], ...] (2=左上角, 3=右下角)
        """
        point_prompts = []
        point_labels = []

        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = x1 * self.target_size
            y1 = y1 * self.target_size
            x2 = x2 * self.target_size
            y2 = y2 * self.target_size

            # 生成box的四个角点
            # 标签：2=左上角，3=右下角
            points = [
                [x1, y1],  # 左上角
                [x2, y2],  # 右下角
            ]

            labels = [1, 1]  # 对应四个角点的标签

            point_prompts.append(points)
            point_labels.append(labels)

        return point_prompts, point_labels

    def _pad_prompts(self, point_prompts: List, point_labels: List) -> Tuple[List, List]:
        """
        填充prompts到固定长度

        Args:
            point_prompts: point坐标列表
            point_labels: point标签列表

        Returns:
            填充后的prompts和labels
        """
        if len(point_prompts) == 0:
            # 没有物体时返回空prompts
            return [[[-1, -1], [-1, -1], [-1, -1], [-1, -1]]], [[-1, -1, -1, -1]]

        # 填充到max_objects长度
        while len(point_prompts) < self.max_objects:
            point_prompts.append([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
            point_labels.append([-1, -1, -1, -1])

        return point_prompts[:self.max_objects], point_labels[:self.max_objects]

    def _resize_masks(self, masks: List, original_size: Tuple[int, int]) -> List:
        """
        调整masks尺寸到target_size

        Args:
            masks: mask列表
            original_size: 原始图像尺寸

        Returns:
            调整后的masks
        """
        resized_masks = []
        h_orig, w_orig = original_size

        for mask in masks:
            # 确保mask是2D的
            if mask.ndim == 3:
                if mask.shape[2] == 1:
                    # [H, W, 1] -> [H, W]
                    mask = mask.squeeze(-1)
                elif mask.shape[0] == 1:
                    # [1, H, W] -> [H, W]
                    mask = mask.squeeze(0)
                else:
                    # [H, W, C] 其中C>1，取第一个通道
                    mask = mask[:, :, 0]
                    print(f"Warning: Mask has multiple channels, using first channel: {mask.shape}")

            # 检查mask形状
            if mask.ndim != 2:
                print(f"Warning: Mask has unexpected shape: {mask.shape}, skipping")
                continue

            # 调整mask尺寸
            if h_orig != self.target_size or w_orig != self.target_size:
                try:
                    # 转换为张量并添加batch和channel维度: (H, W) -> (1, 1, H, W)
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

                    # 调整尺寸
                    mask_tensor = torch.nn.functional.interpolate(
                        mask_tensor,
                        size=(self.target_size, self.target_size),
                        mode='bilinear',
                        align_corners=False
                    )

                    # 移除batch和channel维度: (1, 1, H, W) -> (H, W)
                    mask = mask_tensor.squeeze().numpy()

                except Exception as e:
                    print(f"Warning: Failed to resize mask: {e}")
                    # 如果调整失败，使用原始mask
                    pass

            # 二值化
            mask = (mask > 0.5).astype(np.float32)
            resized_masks.append(mask)

        # 填充到固定数量
        while len(resized_masks) < self.max_objects:
            resized_masks.append(np.zeros((self.target_size, self.target_size), dtype=np.float32))

        return resized_masks[:self.max_objects]


def get_coco_transforms(target_size: int = 1024) -> transforms.Compose:
    """
    获取COCO数据集的图像变换

    Args:
        target_size: 目标尺寸

    Returns:
        变换组合
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def collate_fn(batch: List[Dict]) -> Dict:
    """
    批处理函数

    Args:
        batch: 批次数据

    Returns:
        批处理后的数据
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    batch_size = len(batch)
    target_size = 1024  # 固定目标尺寸

    # 堆叠图像和原始尺寸
    try:
        images = torch.stack([item['image'] for item in batch])
        original_sizes = torch.stack([item['original_size'] for item in batch])
    except Exception as e:
        raise ValueError(f"Error stacking images or original_sizes: {e}")

    # 获取最大物体数量
    max_objects = max(item['num_objects'] for item in batch)
    max_objects = max(max_objects, 1)  # 至少为1

    # 初始化batch数据
    point_prompts_batch = torch.zeros(batch_size, max_objects, 4, 2)
    point_labels_batch = torch.full((batch_size, max_objects, 4), -1, dtype=torch.float32)
    masks_batch = torch.zeros(batch_size, max_objects, target_size, target_size)
    num_objects_batch = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        num_obj = item['num_objects']

        # 处理point prompts
        if num_obj > 0:
            try:
                # 确保数据是numpy数组或列表
                point_prompts = np.array(item['point_prompts'][:num_obj])
                point_labels = np.array(item['point_labels'][:num_obj])
                masks = np.array(item['masks'][:num_obj])

                # 检查形状并处理
                try:
                    if point_prompts.ndim == 3 and point_prompts.shape[1] == 4 and point_prompts.shape[2] == 2:
                        point_prompts_batch[i, :num_obj] = torch.from_numpy(point_prompts).float()
                    else:
                        print(f"Warning: Unexpected point_prompts shape: {point_prompts.shape}")

                    if point_labels.ndim == 2 and point_labels.shape[1] == 4:
                        point_labels_batch[i, :num_obj] = torch.from_numpy(point_labels).float()
                    else:
                        print(f"Warning: Unexpected point_labels shape: {point_labels.shape}")

                    if masks.ndim == 3 and masks.shape[1] == target_size and masks.shape[2] == target_size:
                        masks_batch[i, :num_obj] = torch.from_numpy(masks).float()
                    else:
                        print(f"Warning: Unexpected masks shape: {masks.shape}, expected ({target_size}, {target_size})")
                        # 尝试调整形状
                        if masks.ndim == 2 and masks.shape[0] == target_size and masks.shape[1] == target_size:
                            masks_batch[i, :num_obj] = torch.from_numpy(masks).float()
                        else:
                            print(f"Warning: Cannot process masks with shape: {masks.shape}")

                except Exception as e:
                    print(f"Warning: Error processing item {i} in collate_fn: {e}")
                    # 使用默认值（已经初始化为0或-1）
                    pass

                num_objects_batch[i] = num_obj

            except Exception as e:
                print(f"Warning: Error processing item {i} in collate_fn: {e}")
                # 使用默认值（已经初始化为0或-1）
                num_objects_batch[i] = 0

    return {
        'images': images,
        'original_sizes': original_sizes,
        'point_prompts': point_prompts_batch,
        'point_labels': point_labels_batch,
        'masks': masks_batch,
        'num_objects': num_objects_batch
    }