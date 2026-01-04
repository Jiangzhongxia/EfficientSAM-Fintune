# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class SegmentationLosses:
    """
    分割任务的多重损失函数
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        iou_weight: float = 1.0,
        boundary_weight: float = 0.5,
    ):
        """
        初始化分割损失

        Args:
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            dice_weight: Dice Loss权重
            focal_weight: Focal Loss权重
            iou_weight: IoU Loss权重
            boundary_weight: 边界损失权重
        """
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss，用于处理类别不平衡

        Args:
            pred: 预测logits [B, H, W]
            target: 目标mask [B, H, W]

        Returns:
            focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = torch.exp(-bce_loss)
        alpha = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_loss = alpha * (1 - p_t) ** self.focal_gamma * bce_loss

        return focal_loss.mean()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Dice Loss，衡量分割重叠度

        Args:
            pred: 预测概率 [B, H, W] 或 [H, W]
            target: 目标mask [B, H, W] 或 [H, W]

        Returns:
            dice loss
        """
        pred = torch.sigmoid(pred)
        target = target.float()

        # 处理不同维度的输入
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # [H, W] -> [1, H, W]
        if target.dim() == 2:
            target = target.unsqueeze(0)  # [H, W] -> [1, H, W]

        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice_score = (2.0 * intersection + 1e-7) / (union + 1e-7)

        return 1.0 - dice_score.mean()

    def iou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        IoU Loss，直接优化IoU指标

        Args:
            pred: 预测概率 [B, H, W] 或 [H, W]
            target: 目标mask [B, H, W] 或 [H, W]

        Returns:
            iou loss
        """
        pred = torch.sigmoid(pred)
        target = target.float()

        # 处理不同维度的输入
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # [H, W] -> [1, H, W]
        if target.dim() == 2:
            target = target.unsqueeze(0)  # [H, W] -> [1, H, W]

        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
        iou_score = (intersection + 1e-7) / (union + 1e-7)

        return 1.0 - iou_score.mean()

    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        计算单个mask的IoU

        Args:
            pred: 预测概率 [H, W]
            target: 目标mask [H, W]

        Returns:
            IoU分数
        """
        # 确保输入是2D的
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        if target.dim() == 3:
            target = target.squeeze(0)

        # 计算交集和并集
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        # 计算IoU
        if union == 0:
            return 0.0
        iou = intersection / union
        return iou.item()

    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        边界损失，提高分割边缘精度

        Args:
            pred: 预测logits [B, H, W] 或 [H, W]
            target: 目标mask [B, H, W] 或 [H, W]

        Returns:
            boundary loss
        """
        # 处理不同维度的输入
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)  # [H, W] -> [1, H, W]
        if target.dim() == 2:
            target = target.unsqueeze(0)  # [H, W] -> [1, H, W]

        # 确保数值在有效范围内
        pred = torch.clamp(pred, -10, 10)  # 限制logits范围
        target = torch.clamp(target, 0, 1)  # 确保target是二值的

        # 数值稳定的sigmoid计算
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-7, 1 - 1e-7)

        # 计算边缘
        target_edge = self._get_edge(target)
        pred_edge = self._get_edge(pred_sigmoid)

        # 确保边缘值在有效范围内
        target_edge = torch.clamp(target_edge, 1e-7, 1 - 1e-7)
        pred_edge = torch.clamp(pred_edge, 1e-7, 1 - 1e-7)

        # 检查边缘值是否有效
        if torch.isnan(target_edge).any() or torch.isnan(pred_edge).any():
            print(f"Warning: NaN values in edge computation, returning 0")
            return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        if torch.isinf(target_edge).any() or torch.isinf(pred_edge).any():
            print(f"Warning: Inf values in edge computation, returning 0")
            return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        # 使用数值稳定的交叉熵
        try:
            loss = F.binary_cross_entropy(pred_edge, target_edge, reduction='mean')
        except Exception as e:
            print(f"Warning: Error in binary_cross_entropy: {e}, returning 0")
            return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        # 检查损失值是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid boundary loss: {loss.item()}, returning 0")
            return torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        return loss

    def _get_edge(self, mask: torch.Tensor) -> torch.Tensor:
        """
        计算mask的边缘

        Args:
            mask: 输入mask [B, H, W]

        Returns:
            边缘mask
        """
        # 确保mask是4D的 [B, C, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        # 检查输入是否有效
        if torch.isnan(mask).any() or torch.isinf(mask).any():
            print(f"Warning: Invalid values in mask input to _get_edge, returning zeros")
            return torch.zeros_like(mask.squeeze(1))

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)

        # 卷积计算梯度
        try:
            grad_x = F.conv2d(mask, sobel_x, padding=1)
            grad_y = F.conv2d(mask, sobel_y, padding=1)

            # 检查梯度值是否有效
            if torch.isnan(grad_x).any() or torch.isnan(grad_y).any():
                print(f"Warning: NaN values in gradient computation, returning zeros")
                return torch.zeros_like(mask.squeeze(1))

            if torch.isinf(grad_x).any() or torch.isinf(grad_y).any():
                print(f"Warning: Inf values in gradient computation, returning zeros")
                return torch.zeros_like(mask.squeeze(1))
        except Exception as e:
            print(f"Warning: Error in gradient computation: {e}, returning zeros")
            return torch.zeros_like(mask.squeeze(1))

        # 梯度幅值
        try:
            edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)

            # 检查边缘值是否有效
            if torch.isnan(edge).any() or torch.isinf(edge).any():
                print(f"Warning: Invalid values in edge computation, returning zeros")
                return torch.zeros_like(mask.squeeze(1))
        except Exception as e:
            print(f"Warning: Error in edge computation: {e}, returning zeros")
            return torch.zeros_like(mask.squeeze(1))

        # 移除channel维度并归一化到[0,1]
        edge = edge.squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        # 确保边缘值在有效范围内
        edge = torch.clamp(edge, 0, 100)  # 限制边缘值范围
        edge = torch.sigmoid(edge)

        return edge

    def compute_loss(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        num_objects: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失

        Args:
            pred_logits: 预测logits [B, max_queries, num_masks, H, W]
            target_masks: 目标masks [B, max_objects, H, W]
            num_objects: 每张图像的物体数量 [B]

        Returns:
            损失字典
        """
        # 获取张量形状
        batch_size = pred_logits.shape[0]
        max_queries = pred_logits.shape[1]
        num_masks = pred_logits.shape[2]

        # 初始化总损失
        total_focal_loss = 0
        total_dice_loss = 0
        total_iou_loss = 0
        total_boundary_loss = 0
        total_loss = 0
        valid_count = 0

        for b in range(batch_size):
            n_obj = num_objects[b].item()
            if n_obj == 0:
                continue

            for obj_idx in range(n_obj):
                # 确保索引有效
                if obj_idx < num_masks:  # 第二维度是num_masks
                    # 使用第一个查询的第一个mask
                    pred = pred_logits[b, 0, obj_idx]  # [H, W]
                    target = target_masks[b, obj_idx]   # [H, W]

                    # 检查预测和目标是否有效
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print(f"Warning: Invalid pred values at batch {b}, obj {obj_idx}, skipping")
                        continue

                    if torch.isnan(target).any() or torch.isinf(target).any():
                        print(f"Warning: Invalid target values at batch {b}, obj {obj_idx}, skipping")
                        continue

                    # 计算各种损失
                    try:
                        focal_loss = self.focal_loss(pred, target)
                        dice_loss = self.dice_loss(pred, target)
                        iou_loss = self.iou_loss(pred, target)
                        boundary_loss = self.boundary_loss(pred, target)

                        # 检查损失值是否有效
                        if (torch.isnan(focal_loss) or torch.isnan(dice_loss) or
                            torch.isnan(iou_loss) or torch.isnan(boundary_loss) or
                            torch.isinf(focal_loss) or torch.isinf(dice_loss) or
                            torch.isinf(iou_loss) or torch.isinf(boundary_loss)):
                            print(f"Warning: Invalid loss values at batch {b}, obj {obj_idx}, skipping")
                            continue

                        # 加权求和
                        obj_loss = (
                            self.focal_weight * focal_loss +
                            self.dice_weight * dice_loss +
                            self.iou_weight * iou_loss +
                            self.boundary_weight * boundary_loss
                        )

                        # 检查总损失是否有效
                        if torch.isnan(obj_loss) or torch.isinf(obj_loss):
                            print(f"Warning: Invalid total loss at batch {b}, obj {obj_idx}, skipping")
                            continue

                        total_focal_loss += focal_loss
                        total_dice_loss += dice_loss
                        total_iou_loss += iou_loss
                        total_boundary_loss += boundary_loss
                        total_loss += obj_loss
                        valid_count += 1
                    except Exception as e:
                        print(f"Warning: Error computing loss at batch {b}, obj {obj_idx}: {e}, skipping")
                        continue
                else:
                    print(f"Warning: obj_idx {obj_idx} >= num_masks {num_masks}, skipping")

        if valid_count > 0:
            total_focal_loss /= valid_count
            total_dice_loss /= valid_count
            total_iou_loss /= valid_count
            total_boundary_loss /= valid_count
            total_loss /= valid_count
        else:
            total_focal_loss = torch.tensor(0.0, device=pred_logits.device)
            total_dice_loss = torch.tensor(0.0, device=pred_logits.device)
            total_iou_loss = torch.tensor(0.0, device=pred_logits.device)
            total_boundary_loss = torch.tensor(0.0, device=pred_logits.device)
            total_loss = torch.tensor(0.0, device=pred_logits.device)

        return {
            'total_loss': total_loss,
            'focal_loss': total_focal_loss,
            'dice_loss': total_dice_loss,
            'iou_loss': total_iou_loss,
            'boundary_loss': total_boundary_loss
        }


class KnowledgeDistillation:
    """
    知识蒸馏模块
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        feature_distillation: bool = True,
        attention_distillation: bool = True,
    ):
        """
        初始化知识蒸馏

        Args:
            temperature: 蒸馏温度
            alpha: 学生损失权重
            beta: 知识蒸馏损失权重
            feature_distillation: 是否使用特征蒸馏
            attention_distillation: 是否使用注意力蒸馏
        """
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.feature_distillation = feature_distillation
        self.attention_distillation = attention_distillation

    def knowledge_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        标准知识蒸馏损失

        Args:
            student_logits: 学生模型输出 [B, K, N, H, W]
            teacher_logits: 教师模型输出 [B, K, N, H, W]
            target_masks: 目标masks [B, N, H, W]

        Returns:
            蒸馏损失
        """
        # 温度缩放
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL散度损失
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return kd_loss

    def feature_distillation_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        特征蒸馏损失

        Args:
            student_features: 学生特征 [B, C, H, W]
            teacher_features: 教师特征 [B, C, H, W]

        Returns:
            特征蒸馏损失
        """
        # L2距离
        loss = F.mse_loss(student_features, teacher_features)
        return loss

    def attention_distillation_loss(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor
    ) -> torch.Tensor:
        """
        注意力蒸馏损失

        Args:
            student_attention: 学生注意力图 [B, N_head, H, W]
            teacher_attention: 教师注意力图 [B, N_head, H, W]

        Returns:
            注意力蒸馏损失
        """
        # 计算注意力相似度
        student_attn = F.normalize(student_attention.flatten(1), p=2, dim=1)
        teacher_attn = F.normalize(teacher_attention.flatten(1), p=2, dim=1)

        # 余弦相似度损失
        loss = 1 - F.cosine_similarity(student_attn, teacher_attn).mean()
        return loss

    def compute_distillation_loss(
        self,
        student_outputs: Dict,
        teacher_outputs: Optional[Dict] = None,
        target_masks: Optional[torch.Tensor] = None,
        num_objects: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算总蒸馏损失

        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            target_masks: 目标masks
            num_objects: 物体数量

        Returns:
            损失字典
        """
        losses = {}

        if teacher_outputs is not None:
            # 标准知识蒸馏
            kd_loss = self.knowledge_distillation_loss(
                student_outputs['masks'],
                teacher_outputs['masks'],
                target_masks
            )
            losses['kd_loss'] = kd_loss

            # 特征蒸馏
            if (self.feature_distillation and
                'features' in student_outputs and
                'features' in teacher_outputs):
                feat_loss = self.feature_distillation_loss(
                    student_outputs['features'],
                    teacher_outputs['features']
                )
                losses['feature_loss'] = feat_loss

            # 注意力蒸馏
            if (self.attention_distillation and
                'attention' in student_outputs and
                'attention' in teacher_outputs):
                attn_loss = self.attention_distillation_loss(
                    student_outputs['attention'],
                    teacher_outputs['attention']
                )
                losses['attention_loss'] = attn_loss

        return losses


class ProgressiveTraining:
    """
    渐进式训练策略
    """

    def __init__(
        self,
        total_epochs: int,
        warmup_epochs: int = 5,
        max_temperature: float = 4.0,
        min_temperature: float = 1.0,
    ):
        """
        初始化渐进式训练

        Args:
            total_epochs: 总训练轮数
            warmup_epochs: 预热轮数
            max_temperature: 最大蒸馏温度
            min_temperature: 最小蒸馏温度
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature

    def get_current_temperature(self, current_epoch: int) -> float:
        """
        获取当前epoch的蒸馏温度

        Args:
            current_epoch: 当前轮数

        Returns:
            当前温度
        """
        if current_epoch < self.warmup_epochs:
            return self.max_temperature
        else:
            # 线性衰减
            progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.max_temperature - progress * (self.max_temperature - self.min_temperature)

    def get_alpha_beta(self, current_epoch: int) -> Tuple[float, float]:
        """
        获取当前epoch的alpha和beta权重

        Args:
            current_epoch: 当前轮数

        Returns:
            (alpha, beta) 权重
        """
        if current_epoch < self.warmup_epochs:
            # 预热阶段：更关注真实标签
            alpha = 0.9
            beta = 0.1
        else:
            # 逐步增加蒸馏权重
            progress = (current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            alpha = 0.9 - progress * 0.4  # 从0.9到0.5
            beta = 0.1 + progress * 0.4   # 从0.1到0.5

        return alpha, beta


class AdaptiveLossWeights:
    """
    自适应损失权重调整
    """

    def __init__(self, update_freq: int = 100, smoothing_factor: float = 0.95):
        """
        初始化自适应权重

        Args:
            update_freq: 更新频率（每N个batch）
            smoothing_factor: 平滑因子
        """
        self.update_freq = update_freq
        self.smoothing_factor = smoothing_factor
        self.loss_history = {'focal': [], 'dice': [], 'iou': [], 'boundary': []}
        self.weights = {'focal': 1.0, 'dice': 1.0, 'iou': 1.0, 'boundary': 0.5}

    def update_weights(self, losses: Dict[str, float], step: int) -> Dict[str, float]:
        """
        更新损失权重

        Args:
            losses: 当前损失
            step: 当前步数

        Returns:
            更新后的权重
        """
        if step % self.update_freq != 0:
            return self.weights

        # 记录损失
        for key in ['focal', 'dice', 'iou', 'boundary']:
            if f'{key}_loss' in losses:
                self.loss_history[key].append(losses[f'{key}_loss'])
                # 保持固定长度
                if len(self.loss_history[key]) > 100:
                    self.loss_history[key] = self.loss_history[key][-100:]

        # 更新权重
        for key in ['focal', 'dice', 'iou', 'boundary']:
            if len(self.loss_history[key]) > 10:
                # 计算损失的方差，方差大时增加权重
                recent_losses = self.loss_history[key][-10:]
                variance = np.var(recent_losses)

                # 归一化权重
                if key == 'boundary':
                    base_weight = 0.5
                else:
                    base_weight = 1.0

                # 根据方差调整权重
                adjustment = 1.0 + variance * 0.1
                self.weights[key] = base_weight * adjustment

        return self.weights