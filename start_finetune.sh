#!/bin/bash

# EfficientSAM 微调启动脚本
# 自动化环境检查和训练启动

set -e

echo "🚀 EfficientSAM 微调启动脚本"
echo "=================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: $PYTHON_VERSION"
}

# 检查CUDA
check_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi 未找到，可能没有GPU或CUDA驱动"
        return 1
    fi

    log_info "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | while read -r line; do
        log_info "   $line"
    done
}

# 创建虚拟环境
create_venv() {
    if [ ! -d "venv" ]; then
        log_info "创建Python虚拟环境..."
        python3 -m venv venv
    fi

    log_info "激活虚拟环境..."
    source venv/bin/activate

    log_info "升级pip..."
    pip install --upgrade pip
}

# 安装依赖
install_dependencies() {
    log_info "安装基础依赖..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    log_info "安装其他依赖..."
    pip install pycocotools tensorboard matplotlib numpy pillow

    log_info "安装开发依赖..."
    pip install flake8 black isort mypy
}

# 运行环境测试
run_test() {
    log_info "运行环境测试..."
    python test_setup.py

    if [ $? -ne 0 ]; then
        log_error "环境测试失败，请检查配置"
        exit 1
    fi
}

# 准备数据
prepare_data() {
    log_warn "请确保数据集已准备就绪："
    echo "   - 训练集: path/to/train/images/"
    echo "   - 训练标注: path/to/train/annotations.json"
    echo "   - 验证集: path/to/val/images/"
    echo "   - 验证标注: path/to/val/annotations.json"

    read -p "数据集已准备好吗？(y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_error "请先准备好数据集"
        exit 1
    fi
}

# 配置训练
configure_training() {
    log_info "配置训练参数..."

    # 选择配置文件
    echo "请选择配置文件："
    echo "1) 完整配置 (推荐生产环境)"
    echo "2) 轻量配置 (推荐快速测试)"

    read -p "请选择 (1/2): " choice
    case $choice in
        1)
            CONFIG_FILE="configs/finetune_config.json"
            ;;
        2)
            CONFIG_FILE="configs/finetune_config_light.json"
            ;;
        *)
            log_error "无效选择"
            exit 1
            ;;
    esac

    # 复制配置文件
    cp "$CONFIG_FILE" my_config.json

    # 编辑配置文件
    log_info "请编辑配置文件 my_config.json，设置正确的数据路径"
    read -p "按回车键继续..."

    # 检查配置文件
    if [ ! -f "my_config.json" ]; then
        log_error "配置文件不存在"
        exit 1
    fi
}

# 开始训练
start_training() {
    log_info "开始训练..."

    # 创建输出目录
    OUTPUT_DIR="./outputs/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"

    log_info "输出目录: $OUTPUT_DIR"

    # 启动训练
    python finetune.py \
        --config my_config.json \
        --save_dir "$OUTPUT_DIR" \
        --device cuda

    log_info "训练完成！"
    log_info "模型保存在: $OUTPUT_DIR/"
    log_info "TensorBoard日志: $OUTPUT_DIR/tensorboard/"
}

# 监控训练
monitor_training() {
    log_info "启动TensorBoard监控..."
    echo "在浏览器中访问: http://localhost:6006"
    tensorboard --logdir ./outputs --port 6006
}

# 显示帮助
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  install     安装依赖和环境"
    echo "  test        运行环境测试"
    echo "  configure   配置训练参数"
    echo "  train       开始训练"
    echo "  monitor     启动TensorBoard监控"
    echo "  all         完整流程（安装+测试+配置+训练）"
    echo "  help        显示此帮助信息"
}

# 主函数
main() {
    case "${1:-all}" in
        install)
            check_python
            create_venv
            install_dependencies
            ;;
        test)
            run_test
            ;;
        configure)
            prepare_data
            configure_training
            ;;
        train)
            start_training
            ;;
        monitor)
            monitor_training
            ;;
        all)
            check_python
            create_venv
            install_dependencies
            run_test
            prepare_data
            configure_training
            start_training
            ;;
        help)
            show_help
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"