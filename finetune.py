#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
EfficientSAM 微调脚本
支持COCO格式PCB分割数据集，使用box prompt进行mask_decoder微调
集成多种微调和知识蒸馏方法
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from coco_dataset import COCODataset, get_coco_transforms, collate_fn
from losses import SegmentationLosses, KnowledgeDistillation, ProgressiveTraining, AdaptiveLossWeights
import numpy as np

def setup_logger(save_dir: str) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建日志目录
    log_dir = Path(save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 文件处理器
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class EfficientSAMFinetuner:
    """
    EfficientSAM 微调器
    """

    def __init__(
        self,
        config: Dict,
        device: torch.device,
        logger: logging.Logger,
        save_dir: str
    ):
        """
        初始化微调器

        Args:
            config: 配置字典
            device: 训练设备
            logger: 日志记录器
            save_dir: 保存目录
        """
        self.config = config
        self.device = device
        self.logger = logger
        self.save_dir = save_dir

        # 创建保存目录
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 初始化模型
        self._init_models()

        # 初始化数据集
        self._init_datasets()

        # 初始化损失函数
        self._init_losses()

        # 初始化优化器和学习率调度器
        self._init_optimizer()

        # 初始化渐进式训练
        self.progressive_training = ProgressiveTraining(
            total_epochs=config['training']['epochs'],
            warmup_epochs=config['training']['warmup_epochs'],
            max_temperature=config['distillation']['max_temperature'],
            min_temperature=config['distillation']['min_temperature']
        )

        # 初始化自适应权重
        self.adaptive_weights = AdaptiveLossWeights(
            update_freq=config['training']['adaptive_update_freq'],
            smoothing_factor=config['training']['smoothing_factor']
        )

        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=f"{save_dir}/tensorboard")

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def _init_models(self):
        """初始化模型"""
        self.logger.info("初始化模型...")

        # 加载学生模型
        if self.config['model']['student_variant'] == 'vitt':
            self.student_model = build_efficient_sam_vitt()
        else:
            self.student_model = build_efficient_sam_vits()

        # 冻结image_encoder和prompt_encoder
        for param in self.student_model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.student_model.prompt_encoder.parameters():
            param.requires_grad = False

        # 只训练mask_decoder
        for param in self.student_model.mask_decoder.parameters():
            param.requires_grad = True

        # 加载教师模型（如果启用蒸馏）
        self.teacher_model = None
        if self.config['distillation']['enabled']:
            if self.config['model']['teacher_variant'] == 'vitt':
                self.teacher_model = build_efficient_sam_vitt()
            else:
                self.teacher_model = build_efficient_sam_vits()

            # 教师模型不需要梯度
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # 移动到设备
        self.student_model = self.student_model.to(self.device)
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.device)

        # 打印模型参数数量
        total_params = sum(p.numel() for p in self.student_model.parameters())
        trainable_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        self.logger.info(f"总参数数量: {total_params:,}")
        self.logger.info(f"可训练参数数量: {trainable_params:,}")
        self.logger.info(f"训练比例: {trainable_params/total_params*100:.2f}%")

    def _init_datasets(self):
        """初始化数据集"""
        self.logger.info("初始化数据集...")

        # 数据集配置
        dataset_config = self.config['dataset']

        # 训练集
        self.train_dataset = COCODataset(
            root_dir=dataset_config['train_root'],
            annotation_file=dataset_config['train_annotation'],
            transform=get_coco_transforms(dataset_config['target_size']),
            target_size=dataset_config['target_size'],
            max_objects=dataset_config['max_objects'],
            iou_threshold=dataset_config['iou_threshold'],
            random_box_augmentation=dataset_config.get('random_box_augmentation', True),
            box_noise_scale=dataset_config.get('box_noise_scale', 0.1)
        )

        # 验证集
        self.val_dataset = COCODataset(
            root_dir=dataset_config['val_root'],
            annotation_file=dataset_config['val_annotation'],
            transform=get_coco_transforms(dataset_config['target_size']),
            target_size=dataset_config['target_size'],
            max_objects=dataset_config['max_objects'],
            iou_threshold=dataset_config['iou_threshold'],
            random_box_augmentation=False,  # 验证集不进行数据增强
            box_noise_scale=0.0
        )

        # 数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )

        self.logger.info(f"训练集大小: {len(self.train_dataset)}")
        self.logger.info(f"验证集大小: {len(self.val_dataset)}")

    def _init_losses(self):
        """初始化损失函数"""
        self.logger.info("初始化损失函数...")

        loss_config = self.config['losses']

        # 分割损失
        self.segmentation_losses = SegmentationLosses(
            focal_alpha=loss_config['focal_alpha'],
            focal_gamma=loss_config['focal_gamma'],
            dice_weight=loss_config['dice_weight'],
            focal_weight=loss_config['focal_weight'],
            iou_weight=loss_config['iou_weight'],
            boundary_weight=loss_config['boundary_weight']
        )

        # 知识蒸馏损失
        self.distillation = KnowledgeDistillation(
            temperature=self.config['distillation']['temperature'],
            alpha=self.config['distillation']['alpha'],
            beta=self.config['distillation']['beta'],
            feature_distillation=self.config['distillation'].get('feature_distillation', True),
            attention_distillation=self.config['distillation'].get('attention_distillation', True)
        )

    def _init_optimizer(self):
        """初始化优化器"""
        self.logger.info("初始化优化器...")

        # 只优化mask_decoder参数
        optimizer_params = [
            {'params': self.student_model.mask_decoder.parameters(),
             'lr': self.config['training']['learning_rate']}
        ]

        # 优化器选择
        optimizer_name = self.config['training']['optimizer'].lower()
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                optimizer_params,
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                optimizer_params,
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                optimizer_params,
                momentum=self.config['training']['momentum'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        # 学习率调度器
        scheduler_name = self.config['training']['scheduler'].lower()
        if scheduler_name == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['training']['min_lr']
            )
        elif scheduler_name == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config['training']['learning_rate'],
                epochs=self.config['training']['epochs'],
                steps_per_epoch=len(self.train_loader)
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['lr_decay_step'],
                gamma=self.config['training']['lr_decay_gamma']
            )
        else:
            self.scheduler = None

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.student_model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'focal_loss': 0.0,
            'dice_loss': 0.0,
            'iou_loss': 0.0,
            'boundary_loss': 0.0,
            'kd_loss': 0.0,
            'feature_loss': 0.0,
            'attention_loss': 0.0
        }

        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            # 数据移动到设备
            images = batch['images'].to(self.device)
            point_prompts = batch['point_prompts'].to(self.device)
            point_labels = batch['point_labels'].to(self.device)
            target_masks = batch['masks'].to(self.device)
            num_objects = batch['num_objects'].to(self.device)
            original_sizes = batch['original_sizes'].to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 学生模型前向传播
            with torch.set_grad_enabled(True):
                # 获取图像特征
                image_embeddings = self.student_model.get_image_embeddings(images)

                # 预测masks
                student_masks, student_iou = self.student_model.predict_masks(
                    image_embeddings=image_embeddings,
                    batched_points=point_prompts,
                    batched_point_labels=point_labels,
                    multimask_output=True,
                    input_h=original_sizes[:, 0].long(),
                    input_w=original_sizes[:, 1].long(),
                    output_h=1024,
                    output_w=1024
                )
                # # 在 loss 计算之前插入
                # if batch_idx == 0:  # 每个 epoch 只看第一张
                #     # 取出第一张图和对应的 mask
                #     debug_img = images[0].cpu().numpy().transpose(1, 2, 0) # (C,H,W) -> (H,W,C)
                #     debug_mask = target_masks[0].cpu().numpy() # (H,W)
                    
                #     # 反归一化图片以便显示 (假设是 standard imagenet norm)
                #     mean = np.array([0.485, 0.456, 0.406])
                #     std = np.array([0.229, 0.224, 0.225])
                #     debug_img = std * debug_img + mean
                #     debug_img = np.clip(debug_img, 0, 1)
                    
                #     # 叠加密集 Mask 看看对不对
                #     import matplotlib.pyplot as plt
                #     plt.figure(figsize=(10,5))
                #     plt.subplot(1,2,1); plt.imshow(debug_img); plt.title("Training Image")
                #     plt.subplot(1,2,2); plt.imshow(debug_mask); plt.title("Target Mask (Resized)")
                #     plt.savefig(f"debug_alignment_epoch_{self.current_epoch}.png")
                #     plt.close()

                # 计算分割损失
                seg_losses = self.segmentation_losses.compute_loss(
                    student_masks,
                    target_masks,
                    num_objects
                )

                # 计算蒸馏损失
                distillation_losses = {}
                if self.teacher_model is not None:
                    with torch.no_grad():
                        # 教师模型前向传播
                        teacher_masks, teacher_iou = self.teacher_model.predict_masks(
                            image_embeddings=image_embeddings,
                            batched_points=point_prompts,
                            batched_point_labels=point_labels,
                            multimask_output=True,
                            input_h=original_sizes[:, 0].long(),
                            input_w=original_sizes[:, 1].long(),
                            output_h=1024,
                            output_w=1024
                        )

                    # 计算蒸馏损失
                    teacher_outputs = {'masks': teacher_masks}
                    student_outputs = {'masks': student_masks}
                    distillation_losses = self.distillation.compute_distillation_loss(
                        student_outputs,
                        teacher_outputs,
                        target_masks,
                        num_objects
                    )

                # 更新温度和权重
                current_temp = self.progressive_training.get_current_temperature(self.current_epoch)
                alpha, beta = self.progressive_training.get_alpha_beta(self.current_epoch)

                # 更新自适应权重
                current_weights = self.adaptive_weights.update_weights(
                    {k: v.item() for k, v in seg_losses.items()},
                    self.global_step
                )

                # 计算总损失
                total_loss = 0

                # 分割损失
                for loss_name, loss_value in seg_losses.items():
                    weight = current_weights.get(loss_name.replace('_loss', ''), 1.0)
                    total_loss += weight * loss_value
                    epoch_losses[loss_name] += loss_value.item()

                # 蒸馏损失
                if self.config['distillation']['enabled']:
                    for loss_name, loss_value in distillation_losses.items():
                        if loss_name == 'kd_loss':
                            total_loss += beta * loss_value
                        else:
                            total_loss += 0.1 * loss_value  # 特征和注意力损失的权重
                        epoch_losses[loss_name] += loss_value.item()

                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.mask_decoder.parameters(),
                    self.config['training']['grad_clip_norm']
                )
                self.optimizer.step()

                # 更新学习率
                if self.scheduler is not None:
                    if isinstance(self.scheduler, OneCycleLR):
                        self.scheduler.step()

                # 记录损失
                epoch_losses['total_loss'] += total_loss.item()
                self.global_step += 1

                # 打印进度
                if batch_idx % self.config['training']['print_freq'] == 0:
                    self.logger.info(
                        f"Epoch [{self.current_epoch}/{self.config['training']['epochs']}], "
                        f"Batch [{batch_idx}/{num_batches}], "
                        f"Loss: {total_loss.item():.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()

        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.student_model.eval()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        val_losses = {
            'total_loss': 0.0,
            'focal_loss': 0.0,
            'dice_loss': 0.0,
            'iou_loss': 0.0,
            'boundary_loss': 0.0
        }

        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch in self.val_loader:
                # 数据移动到设备
                images = batch['images'].to(self.device)
                point_prompts = batch['point_prompts'].to(self.device)
                point_labels = batch['point_labels'].to(self.device)
                target_masks = batch['masks'].to(self.device)
                num_objects = batch['num_objects'].to(self.device)
                original_sizes = batch['original_sizes'].to(self.device)

                # 获取图像特征
                image_embeddings = self.student_model.get_image_embeddings(images)

                # 预测masks
                student_masks, student_iou = self.student_model.predict_masks(
                    image_embeddings=image_embeddings,
                    batched_points=point_prompts,
                    batched_point_labels=point_labels,
                    multimask_output=True,
                    input_h=original_sizes[:, 0].long(),
                    input_w=original_sizes[:, 1].long(),
                    output_h=1024,
                    output_w=1024
                )

                # 计算分割损失
                seg_losses = self.segmentation_losses.compute_loss(
                    student_masks,
                    target_masks,
                    num_objects
                )

                # 累计损失
                for loss_name, loss_value in seg_losses.items():
                    val_losses[loss_name] += loss_value.item()

                # 计算总损失
                total_loss = sum(seg_losses.values())
                val_losses['total_loss'] += total_loss.item()

        # 平均损失
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config
        }

        # 保存当前检查点
        checkpoint_path = f"{self.save_dir}/checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳检查点
        if is_best:
            best_path = f"{self.save_dir}/best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到 {best_path}")

        self.logger.info(f"保存检查点到 {checkpoint_path}")

    def train(self):
        """开始训练"""
        self.logger.info("开始训练...")

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # 训练
            train_losses = self.train_epoch()

            # 验证
            val_losses = self.validate()

            # 记录到TensorBoard
            self.writer.add_scalar('Loss/Train_Total', train_losses['total_loss'], epoch)
            self.writer.add_scalar('Loss/Val_Total', val_losses['total_loss'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # 记录详细损失
            for key in train_losses:
                if key != 'total_loss':
                    self.writer.add_scalar(f'Loss/Train_{key}', train_losses[key], epoch)
                    self.writer.add_scalar(f'Loss/Val_{key}', val_losses.get(key, 0), epoch)

            # 打印epoch信息
            self.logger.info(
                f"Epoch [{epoch}/{self.config['training']['epochs']}], "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Loss: {val_losses['total_loss']:.4f}, "
                f"Best Loss: {self.best_loss:.4f}"
            )

            # 保存最佳模型
            if val_losses['total_loss'] < self.best_loss:
                self.best_loss = val_losses['total_loss']
                self.save_checkpoint(is_best=True)
            else:
                self.save_checkpoint(is_best=False)

        self.logger.info("训练完成！")
        self.writer.close()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EfficientSAM 微调')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--save_dir', type=str, required=True, help='保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')

    args = parser.parse_args()

    # 检查设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    config = load_config(args.config)

    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger(args.save_dir)

    # 创建微调器
    trainer = EfficientSAMFinetuner(
        config=config,
        device=device,
        logger=logger,
        save_dir=args.save_dir
    )

    # 恢复训练（如果指定）
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.student_model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.global_step = checkpoint['global_step']
        trainer.best_loss = checkpoint['best_loss']
        logger.info(f"从检查点恢复训练: {args.resume}")

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()