#!/usr/bin/env python3
"""
检查不同batch_size的显存占用情况
"""

import torch
import gc
import psutil
import os
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt


def get_memory_usage():
    """获取当前显存使用情况"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    else:
        return {'error': 'CUDA not available'}


def test_batch_sizes():
    """测试不同batch_size的显存占用"""
    print("🔍 测试不同batch_size的显存占用...")

    # 初始化模型
    model = build_efficient_sam_vitt()
    model = model.cuda()

    # 测试不同的batch_size
    batch_sizes = [1, 2, 4, 8, 16]
    image_shape = (3, 1024, 1024)  # C, H, W

    results = []

    for batch_size in batch_sizes:
        print(f"\n📊 测试 batch_size = {batch_size}")

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        # 获取初始显存
        initial_memory = get_memory_usage()
        print(f"   初始显存: {initial_memory.get('allocated', 0):.2f} GB")

        try:
            # 创建模拟数据
            images = torch.randn(batch_size, *image_shape).cuda()
            point_prompts = torch.randn(batch_size, 10, 4, 2).cuda()  # max_objects=10
            point_labels = torch.randint(0, 4, (batch_size, 10, 4)).cuda()

            # 前向传播
            with torch.no_grad():
                image_embeddings = model.get_image_embeddings(images)
                masks, ious = model.predict_masks(
                    image_embeddings=image_embeddings,
                    batched_points=point_prompts,
                    batched_point_labels=point_labels,
                    multimask_output=True,
                    input_h=1024,
                    input_w=1024,
                    output_h=1024,
                    output_w=1024
                )

            # 获取最终显存
            final_memory = get_memory_usage()
            allocated = final_memory.get('allocated', 0)
            initial = initial_memory.get('allocated', 0)

            # 计算增量
            memory_increase = allocated - initial

            result = {
                'batch_size': batch_size,
                'initial_memory_gb': initial,
                'final_memory_gb': allocated,
                'increase_gb': memory_increase,
                'increase_per_sample': memory_increase / batch_size if batch_size > 0 else 0
            }

            results.append(result)

            print(f"   最终显存: {allocated:.2f} GB")
            print(f"   显存增加: {memory_increase:.2f} GB")
            print(f"   每样本增加: {memory_increase / batch_size if batch_size > 0 else 0:.3f} GB")

            # 删除张量
            del images, point_prompts, point_labels, image_embeddings, masks, ious

        except torch.cuda.OutOfMemoryError:
            print(f"   ❌ 显存不足！batch_size {batch_size} 太大")
            results.append({
                'batch_size': batch_size,
                'error': 'Out of Memory'
            })
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            results.append({
                'batch_size': batch_size,
                'error': str(e)
            })

    # 打印总结
    print("\n" + "="*60)
    print("📊 显存使用总结:")
    print("="*60)

    print(f"{'Batch Size':<12} {'显存增加(GB)':<15} {'每样本(GB)':<12}")
    print("-" * 40)

    for result in results:
        if 'error' not in result:
            bs = result['batch_size']
            inc = result['increase_gb']
            per_sample = result['increase_per_sample']
            print(f"{bs:<12} {inc:<15.3f} {per_sample:<12.3f}")
        else:
            bs = result['batch_size']
            error = result['error']
            print(f"{bs:<12} {error:<15}")

    return results


def analyze_actual_batch_usage():
    """分析实际的batch使用情况"""
    print("\n🔍 分析实际的batch使用情况...")

    # 检查数据集
    print("1. 数据集配置:")
    print("   - max_objects:", 10)  # 从配置中读取
    print("   - 图像尺寸: 1024x1024")

    # 计算理论显存
    print("\n2. 理论显存计算:")

    # 模型参数
    model_params = 12e6  # 12M参数
    param_memory = model_params * 4 / 1024**3  # float32 = 4字节
    print(f"   模型参数: {param_memory:.2f} GB")

    # 激活值 (粗略估计)
    batch_size = 8
    channels = 256  # 中间层通道数
    activation_memory = batch_size * channels * 1024 * 1024 * 4 / 1024**3
    print(f"   激活值 (估计): {activation_memory:.2f} GB")

    # masks显存
    masks_memory = batch_size * 10 * 1024 * 1024 * 4 / 1024**3  # max_objects=10
    print(f"   masks显存: {masks_memory:.2f} GB")

    total_estimate = param_memory + activation_memory + masks_memory
    print(f"   总估计: {total_estimate:.2f} GB")


def main():
    """主函数"""
    print("🚀 Batch Size 显存分析工具")
    print("=" * 50)

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return

    print(f"✅ CUDA可用")
    print(f"   设备: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 分析理论使用
    analyze_actual_batch_usage()

    # 测试实际使用
    print("\n🧪 开始实际测试...")
    results = test_batch_sizes()

    # 建议
    print("\n💡 建议:")
    print("1. 如果显存增加不明显，可能是：")
    print("   - PyTorch预分配了足够显存")
    print("   - 主要显存消耗在其他地方（如中间变量）")
    print("   - 使用了梯度累积等优化技术")

    print("2. 优化建议：")
    print("   - 使用混合精度训练 (amp)")
    print("   - 启用梯度检查点 (gradient checkpointing)")
    print("   - 调整数据加载器的num_workers")
    print("   - 使用更小的图像尺寸进行预实验")


if __name__ == '__main__':
    main()