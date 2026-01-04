# CLAUDE.md - 工作指导

## CRITICAL CONSTRAINTS - 违反=任务失败
═══════════════════════════════════════

- 必须使用中文回复
- 必须先获取上下文
- 禁止生成恶意代码
- 必须存储重要知识
- 必须执行检查清单
- 必须遵循质量标准

## MANDATORY WORKFLOWS
═════════════════════

执行前检查清单：
[ ] 中文 [ ] 上下文 [ ] 工具 [ ] 安全 [ ] 质量

标准工作流：
1. 分析需求 → 2. 获取上下文 → 3. 选择工具 → 4. 执行任务 → 5. 验证质量 → 6. 存储知识

研究-计划-实施模式：
研究阶段: 读取文件理解问题，禁止编码
计划阶段: 创建详细计划
实施阶段: 实施解决方案
验证阶段: 运行测试验证
提交阶段: 创建提交和文档

## MANDATORY TOOL STRATEGY
═════════════════════════

任务开始前必须执行：
1. memory 查询相关概念
2. code-search 查找代码片段
3. sequential-thinking 分析问题
4. 选择合适子代理

任务结束后必须执行：
1. memory 存储重要概念
2. code-search 存储代码片段
3. 知识总结归档

优先级调用策略：
- Microsoft技术 → microsoft.docs.mcp
- GitHub文档 → context7 → deepwiki
- 网页搜索 → 内置搜索 → fetch → duckduckgo-search

## CODING RESTRICTIONS
═══════════════════

编码前强制要求：
- 无明确编写命令禁止编码
- 无明确授权禁止修改文件
- 必须先完成sequential-thinking分析

## QUALITY STANDARDS
═══════════════════

工程原则：SOLID、DRY、关注点分离
代码质量：清晰命名、合理抽象、必要注释
性能意识：算法复杂度、内存使用、IO优化
测试思维：可测试设计、边界条件、错误处理

## SUBAGENT SELECTION
════════════════════

必须主动调用合适子代理：
- Python项目 → python-pro
- C#/.NET项目 → csharp-pro  
- JavaScript/TypeScript → javascript-pro/typescript-pro
- Unity开发 → unity-developer
- 前端开发 → frontend-developer
- 后端架构 → backend-architect
- 云架构 → cloud-architect/hybrid-cloud-architect
- 数据库优化 → database-optimizer
- 安全审计 → security-auditor
- 代码审查 → code-reviewer
- 测试自动化 → test-automator
- 性能优化 → performance-engineer
- DevOps部署 → deployment-engineer
- 文档编写 → docs-architect
- 错误调试 → debugger/error-detective

## ENFORCEMENT
══════════════

强制触发器：会话开始→检查约束，工具调用前→检查流程，回复前→验证清单
自我改进：成功→存储，失败→更新规则，持续→优化策略
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EfficientSAM is an efficient implementation of the Segment Anything Model (SAM) that leverages masked image pretraining for efficient segmentation tasks. The project provides two main model variants: EfficientSAM-Ti (Tiny) and EfficientSAM-S (Small), both optimized for real-time performance while maintaining competitive accuracy.

## Development Commands

### Code Quality and Linting
```bash
# Run all linting tools (isort, black, flake8, mypy)
./linter.sh

# Individual linting commands
isort . --atomic
black -l 100 .
flake8 .
mypy --exclude 'setup.py|notebooks' .
```

### Model Usage and Testing
```bash
# Run the main example script
python EfficientSAM_example.py

# Run ONNX inference example
python EfficientSAM_onnx_example.py

# Export models to different formats
python export_to_onnx.py
python export_to_torchscript.py
```

### Package Installation
```bash
# Install the package with basic requirements
pip install -e .

# Install with all dependencies (matplotlib, onnx, onnxruntime)
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

## Architecture Overview

### Core Components

The EfficientSAM architecture follows a modular design with clear separation of concerns:

1. **Model Building (`build_efficient_sam.py`)**: Factory functions for creating model variants
   - `build_efficient_sam_vitt()`: Creates EfficientSAM-Ti model (192-dim encoder, 3 heads)
   - `build_efficient_sam_vits()`: Creates EfficientSAM-S model (384-dim encoder, 6 heads)

2. **Main Model (`efficient_sam.py`)**: Core model implementation combining encoder and decoder
   - Handles image preprocessing, point prompts, and mask generation
   - Supports batch processing and multiple mask predictions

3. **Encoder (`efficient_sam_encoder.py`)**: Vision transformer-based image encoder
   - Processes input images into feature representations
   - Uses patch embedding and transformer blocks

4. **Decoder (`efficient_sam_decoder.py`)**: Mask decoder with prompt conditioning
   - Generates segmentation masks based on encoder features and user prompts
   - Supports point and box prompts

5. **Supporting Modules**:
   - `mlp.py`: Multi-layer perceptron components
   - `two_way_transformer.py`: Two-way transformer for prompt-image interaction

### Model Variants

- **EfficientSAM-Ti**: Lightweight version with 192-dim encoder, 3 attention heads
- **EfficientSAM-S**: Standard version with 384-dim encoder, 6 attention heads

Both models share the same architecture but differ in model capacity and performance characteristics.

### Input/Output Format

**Input**:
- Images: Tensor of shape `(B, C, H, W)` (typically 3x1024x1024)
- Point prompts: Tensor of shape `(B, N, 2)` for (x,y) coordinates
- Point labels: Tensor of shape `(B, N)` indicating foreground/background points

**Output**:
- Predicted logits: Tensor of shape `(B, K, M, H, W)` for K masks
- Predicted IoU: Tensor of shape `(B, K, M)` for mask quality scores

### Checkpoint Management

Model checkpoints are stored in the `weights/` directory:
- `efficient_sam_vitt.pt`: EfficientSAM-Ti checkpoint
- `efficient_sam_vits.pt.zip`: EfficientSAM-S checkpoint (compressed due to size)

The example script automatically extracts the zip file before model loading.

### Export Formats

The project supports multiple export formats for deployment:
- **ONNX**: Separate encoder and decoder models via `export_to_onnx.py`
- **TorchScript**: Scripted models via `export_to_torchscript.py`
- **PyTorch**: Native format for development and research

## Development Workflow

1. **Model Development**: Modify core components in `efficient_sam/` directory
2. **Testing**: Use `EfficientSAM_example.py` for quick validation
3. **Export**: Generate deployment models using export scripts
4. **Quality Assurance**: Run linting tools before committing changes

The codebase follows strict code quality standards with automated formatting, import sorting, and type checking enforced through the linter script.