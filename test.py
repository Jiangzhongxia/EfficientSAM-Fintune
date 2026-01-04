#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
EfficientSAM COCO数据集评估与可视化脚本
"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import cv2
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import colorsys
class EfficientSAMPredictor:
    """
    EfficientSAM 推理器 (修正版)
    """

    def __init__(self, model_path: str, device: torch.device = torch.device('cuda'), target_size=1024):
        self.device = device
        self.target_size = target_size
        self.model = self._load_model(model_path, device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_path: str, device: torch.device):
        print(f"Loading model from {model_path}...")
        model = build_efficient_sam_vitt()
        checkpoint = torch.load(model_path, map_location=device)
        
        # 兼容不同格式的 checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 移除 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        model.eval()
        return model

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        return self.transform(image).unsqueeze(0)

    def box_to_points(self, box_norm: list) -> tuple:
        """
        将归一化box转换为point prompt格式 (修正版)
        依据: image_bdb99f.png
        """
        x1, y1, x2, y2 = box_norm

        # 1. 坐标反归一化，映射到 target_size (1024)
        x1 = x1 * self.target_size
        y1 = y1 * self.target_size
        x2 = x2 * self.target_size
        y2 = y2 * self.target_size

        # 2. 只使用两个点 (左上角, 右下角)
        # 注意：这里增加了一个维度 [[]] 以匹配后续 tensor 堆叠的需求
        points = [[x1, y1], [x2, y2]] 
        
        # 3. 对应标签: 2=左上角, 3=右下角
        labels = [2, 3]

        return torch.tensor([points], dtype=torch.float32), torch.tensor([labels], dtype=torch.float32)

    def predict_batch(self, image: Image.Image, boxes_norm: list) -> list:
        """
        对一张图片中的多个box进行批量预测
        """
        original_size = image.size[::-1] # (H, W)
        orig_h, orig_w = original_size
        
        if len(boxes_norm) == 0:
            return []

        # 预处理图像
        image_tensor = self.preprocess_image(image).to(self.device)
        num_boxes = len(boxes_norm)

        # 准备 prompts
        # 修改点：现在每个box只有2个点，而不是4个
        point_prompts = torch.zeros((1, num_boxes, 2, 2), dtype=torch.float32).to(self.device)
        point_labels = torch.full((1, num_boxes, 2), -1.0, dtype=torch.float32).to(self.device)

        for i, box in enumerate(boxes_norm):
            p, l = self.box_to_points(box)
            point_prompts[0, i] = p
            point_labels[0, i] = l

        # 推理
        with torch.no_grad():
            image_embeddings = self.model.get_image_embeddings(image_tensor)
            
            predicted_masks_logits, predicted_iou = self.model.predict_masks(
                image_embeddings=image_embeddings,
                batched_points=point_prompts,
                batched_point_labels=point_labels,
                multimask_output=True,
                input_h=orig_h, input_w=orig_w,
                output_h=self.target_size, output_w=self.target_size
            )

        final_masks = []
        for i in range(num_boxes):
            masks_logits = predicted_masks_logits[0, i]
            ious = predicted_iou[0, i]

            best_idx = torch.argmax(ious).item()
            best_mask_logits = masks_logits[best_idx]

            # 二值化
            mask_1024 = (torch.sigmoid(best_mask_logits) > 0.5).cpu().numpy().astype(np.uint8)

            # 还原回原始图像尺寸
            mask_orig = cv2.resize(mask_1024, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            final_masks.append(mask_orig)

        return final_masks

def calculate_iou(mask1, mask2):
    """计算两个二值mask的IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def generate_category_colors(cat_ids):
    """
    为每个类别ID生成一个固定的颜色 (BGR格式，用于OpenCV)
    """
    colors = {}
    # 使用 HSV 空间生成颜色，保证亮度一致，色相均匀分布
    for i, cat_id in enumerate(cat_ids):
        # 将色相(Hue)均匀分布在 0-1 之间
        h = i / len(cat_ids)
        # 饱和度(Saturation)和亮度(Value)设为较高值，保证颜色鲜艳
        s = 0.9
        v = 0.9
        
        # hsv -> rgb
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # rgb -> bgr (OpenCV使用BGR), 并扩展到 0-255
        color_bgr = (int(b * 255), int(g * 255), int(r * 255))
        colors[cat_id] = color_bgr
        
    return colors

def visualize_comparison(image_path, save_path, gt_masks, pred_masks, boxes_abs, ious, cat_ids, cat_names, color_map):
    """
    可视化 Ground Truth 和 预测结果的对比 (支持多类别彩色)
    """
    # 读取图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return
    
    H, W = img_bgr.shape[:2]
    
    # 创建画布
    # 左边画 GT，右边画 Prediction
    img_gt = img_bgr.copy()
    img_pred = img_bgr.copy()
    
    alpha = 0.5 # 透明度

    for i in range(len(gt_masks)):
        cat_id = cat_ids[i]
        cat_name = cat_names[i]
        
        # 获取该类别对应的颜色
        color = color_map.get(cat_id, (0, 0, 255)) # 默认红色
        
        # --- 绘制 GT (左图) ---
        if gt_masks[i] is not None:
            # 找到mask区域
            mask_bool = gt_masks[i] == 1
            
            # 创建彩色遮罩
            overlay_gt = img_gt.copy()
            overlay_gt[mask_bool] = color # 填充颜色
            
            # 融合
            cv2.addWeighted(overlay_gt, alpha, img_gt, 1 - alpha, 0, img_gt)
            
            # 画框 (实线)
            x, y, w, h = map(int, boxes_abs[i])
            cv2.rectangle(img_gt, (x, y), (x+w, y+h), color, 2)
            
            # 写类别名
            cv2.putText(img_gt, cat_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 绘制 Prediction (右图) ---
        # 找到mask区域
        mask_bool = pred_masks[i] == 1
        
        # 创建彩色遮罩
        overlay_pred = img_pred.copy()
        overlay_pred[mask_bool] = color
        
        # 融合
        cv2.addWeighted(overlay_pred, alpha, img_pred, 1 - alpha, 0, img_pred)

        # 画框 (实线)
        x, y, w, h = map(int, boxes_abs[i])
        cv2.rectangle(img_pred, (x, y), (x+w, y+h), color, 2)
        
        # 写类别名和IoU
        iou_val = ious[i]
        text = f"{cat_name} {iou_val:.2f}"
        cv2.putText(img_pred, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 拼接图像
    combined_img = np.hstack((img_gt, img_pred))
    
    # 添加标题栏
    header = np.zeros((50, combined_img.shape[1], 3), dtype=np.uint8)
    # 标题文字颜色用白色
    cv2.putText(header, "Ground Truth (Color per Class)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(header, "Prediction & IoU", (W + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    combined_img = np.vstack((header, combined_img))
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined_img)


def evaluate_on_coco(model_path, coco_json_path, image_root, output_dir, device_str='cuda'):
    """
    在COCO数据集上进行评估的核心函数
    """
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    # 1. 初始化模型和COCO API
    predictor = EfficientSAMPredictor(model_path, device)
    coco = COCO(coco_json_path)
    img_ids = coco.getImgIds()
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
    # === 新增：生成颜色表 ===
    all_cat_ids = coco.getCatIds()
    color_map = generate_category_colors(all_cat_ids)
    print(f"Generated {len(color_map)} unique colors for classes.")
    # ========================
    # 用于存储指标
    category_ious = defaultdict(list)
    all_ious = []
    
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"Starting evaluation on {len(img_ids)} images...")
    
    # 2. 遍历数据集
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(image_root, img_info['file_name'])
        
        # 加载图像
        try:
            image_pil = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue
            
        W_orig, H_orig = image_pil.size
        
        # 加载该图像的标注
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # 过滤有效标注 (有segmentation且不是人群)
        valid_anns = [ann for ann in anns if ann.get('segmentation') and ann.get('iscrowd') == 0]
        
        if not valid_anns:
            continue
            
        # 准备数据
        gt_masks = []
        boxes_norm = [] # 用于模型输入 [x1_n, y1_n, x2_n, y2_n]
        boxes_abs = []  # 用于可视化和记录 [x, y, w, h]
        cat_names = []
        current_img_cat_ids = [] # === 新增：记录当前图片的类别ID ===

        for ann in valid_anns:
            # 解析 GT Mask
            mask = coco.annToMask(ann)
            gt_masks.append(mask)
            
            # 解析 GT Box (COCO格式: [x, y, w, h])
            x, y, w, h = ann['bbox']
            boxes_abs.append([x, y, w, h])
            
            # 转换为归一化坐标 [x1, y1, x2, y2]
            x1_n = x / W_orig
            y1_n = y / H_orig
            x2_n = (x + w) / W_orig
            y2_n = (y + h) / H_orig
            boxes_norm.append([x1_n, y1_n, x2_n, y2_n])
            
            cat_names.append(cat_id_to_name.get(ann['category_id'], 'unknown'))
            # 记录类别ID
            cat_id = ann['category_id']
            current_img_cat_ids.append(cat_id) # === 新增 ===
            cat_names.append(cat_id_to_name.get(cat_id, 'unknown'))

        # 3. 模型推理
        # 输入归一化的box，返回原始尺寸的二值化mask列表
        pred_masks = predictor.predict_batch(image_pil, boxes_norm)
        
        # 4. 计算指标并收集数据
        current_img_ious = []
        for i in range(len(valid_anns)):
            iou = calculate_iou(pred_masks[i], gt_masks[i])
            cat_id = valid_anns[i]['category_id']
            
            category_ious[cat_id].append(iou)
            all_ious.append(iou)
            current_img_ious.append(iou)
            
        # 5. 可视化 (可选：为了速度可以只保存一部分，这里默认全部保存)
        # 保存路径: output_dir/visualizations/file_name.png
        save_path = os.path.join(vis_dir, os.path.splitext(img_info['file_name'])[0] + "_vis.png")
        # === 修改：传入 cat_ids 和 color_map ===
        visualize_comparison(
            image_path, 
            save_path, 
            gt_masks, 
            pred_masks, 
            boxes_abs, 
            current_img_ious, 
            current_img_cat_ids, # 传入ID列表
            cat_names, 
            color_map            # 传入颜色表
        )
        #visualize_comparison(image_path, save_path, gt_masks, pred_masks, boxes_abs, current_img_ious, cat_names)

    # 6. 统计和输出结果
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    
    results_summary = {}
    print(f"{'Category Name':<30} | {'mIoU':<10} | {'Instances':<10}")
    print("-" * 56)
    
    mIoU_per_cat = []
    for cat_id, ious in category_ious.items():
        mean_iou = np.mean(ious)
        cat_name = cat_id_to_name.get(cat_id, f"Cat {cat_id}")
        num_instances = len(ious)
        print(f"{cat_name:<30} | {mean_iou:.4f}     | {num_instances:<10}")
        
        results_summary[cat_name] = {
            'mIoU': float(mean_iou),
            'num_instances': num_instances
        }
        mIoU_per_cat.append(mean_iou)
        
    overall_mIoU = np.mean(all_ious) if all_ious else 0.0
    mean_category_mIoU = np.mean(mIoU_per_cat) if mIoU_per_cat else 0.0
    
    print("="*50)
    print(f"Overall mean IoU (across all instances): {overall_mIoU:.4f}")
    print(f"Mean Category IoU (mIoU): {mean_category_mIoU:.4f}")
    print(f"Total instances evaluated: {len(all_ious)}")
    print("="*50)
    
    # 保存汇总结果到 JSON
    final_metrics = {
        'overall_mIoU': float(overall_mIoU),
        'mean_category_mIoU': float(mean_category_mIoU),
        'total_instances': len(all_ious),
        'category_details': results_summary
    }
    with open(os.path.join(output_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(final_metrics, f, indent=2)
        
    print(f"Results and visualizations saved to: {output_dir}")


if __name__ == '__main__':
    # ==========================================
    # 配置区域 - 请根据你的实际路径修改这里
    # ==========================================
    
    # 1. 训练好的模型权重路径
    MODEL_PATH = "path/to/your/trained_efficient_sam.pth"
    # 例如: "checkpoints/efficient_sam_vitt_finetuned.pth"
    
    # 2. COCO格式的验证集标注文件路径 (.json)
    COCO_JSON_PATH = "path/to/coco/annotations/instances_val2017.json" 
    
    # 3. COCO验证集图片所在的根目录
    IMAGE_ROOT = "path/to/coco/val2017"
    
    # 4. 输出目录 (用于保存可视化图片和指标JSON)
    OUTPUT_DIR = "evaluation_results_coco"
    
    # 5. 推理设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ==========================================
    
    # 检查路径是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path not found: {MODEL_PATH}")
        exit(1)
    if not os.path.exists(COCO_JSON_PATH):
        print(f"Error: COCO JSON path not found: {COCO_JSON_PATH}")
        exit(1)
    if not os.path.exists(IMAGE_ROOT):
        print(f"Error: Image root dir not found: {IMAGE_ROOT}")
        exit(1)

    # 开始评估
    evaluate_on_coco(
        model_path=MODEL_PATH,
        coco_json_path=COCO_JSON_PATH,
        image_root=IMAGE_ROOT,
        output_dir=OUTPUT_DIR,
        device_str=DEVICE
    )