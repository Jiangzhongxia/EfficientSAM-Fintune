@echo off
REM EfficientSAM 微调启动脚本 (Windows版本)
REM 自动化环境检查和训练启动

setlocal enabledelayedexpansion

echo 🚀 EfficientSAM 微调启动脚本
echo ==================================

REM 颜色定义 (Windows cmd限制)
set "INFO=[INFO]"
set "WARN=[WARN]"
set "ERROR=[ERROR]"

REM 检查Python
:check_python
echo %INFO% 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo %ERROR% Python未安装或不在PATH中
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %INFO% Python版本: %PYTHON_VERSION%

REM 检查CUDA
:check_cuda
echo %INFO% 检查CUDA环境...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo %WARN% nvidia-smi未找到，可能没有GPU或CUDA驱动
) else (
    echo %INFO% GPU信息:
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
)

REM 创建虚拟环境
:create_venv
echo %INFO% 创建Python虚拟环境...
if not exist "venv" (
    python -m venv venv
)

echo %INFO% 激活虚拟环境...
call venv\Scripts\activate.bat

echo %INFO% 升级pip...
python -m pip install --upgrade pip

REM 安装依赖
:install_dependencies
echo %INFO% 安装基础依赖...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo %INFO% 安装其他依赖...
pip install pycocotools tensorboard matplotlib numpy pillow

echo %INFO% 安装开发依赖...
pip install flake8 black isort mypy

REM 运行环境测试
:run_test
echo %INFO% 运行环境测试...
python test_setup.py
if errorlevel 1 (
    echo %ERROR% 环境测试失败，请检查配置
    pause
    exit /b 1
)

REM 准备数据
:prepare_data
echo %WARN% 请确保数据集已准备就绪：
echo    - 训练集: path/to/train/images/
echo    - 训练标注: path/to/train/annotations.json
echo    - 验证集: path/to/val/images/
echo    - 验证标注: path/to/val/annotations.json

set /p data_ready="数据集已准备好吗？(y/n): "
if /i not "%data_ready%"=="y" (
    echo %ERROR% 请先准备好数据集
    pause
    exit /b 1
)

REM 配置训练
:configure_training
echo %INFO% 配置训练参数...
echo 请选择配置文件：
echo 1) 完整配置 (推荐生产环境)
echo 2) 轻量配置 (推荐快速测试)

set /p config_choice="请选择 (1/2): "
if "%config_choice%"=="1" (
    set "CONFIG_FILE=configs\finetune_config.json"
) else if "%config_choice%"=="2" (
    set "CONFIG_FILE=configs\finetune_config_light.json"
) else (
    echo %ERROR% 无效选择
    pause
    exit /b 1
)

REM 复制配置文件
copy "%CONFIG_FILE%" my_config.json

REM 编辑配置文件
echo %INFO% 请编辑配置文件 my_config.json，设置正确的数据路径
pause

REM 检查配置文件
if not exist "my_config.json" (
    echo %ERROR% 配置文件不存在
    pause
    exit /b 1
)

REM 开始训练
:start_training
echo %INFO% 开始训练...

REM 创建输出目录
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set "DATE=%%c%%a%%b"
for /f "tokens=1-3 delims=:." %%a in ('time /t') do set "TIME=%%a%%b%%c"
set "OUTPUT_DIR=outputs\%DATE%_%TIME%"

if not exist "outputs" mkdir outputs
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo %INFO% 输出目录: %OUTPUT_DIR%

REM 启动训练
python finetune.py --config my_config.json --save_dir "%OUTPUT_DIR%" --device cuda

if errorlevel 1 (
    echo %ERROR% 训练过程中出现错误
    pause
    exit /b 1
)

echo %INFO% 训练完成！
echo %INFO% 模型保存在: %OUTPUT_DIR%\
echo %INFO% TensorBoard日志: %OUTPUT_DIR%\tensorboard\

REM 询问是否启动TensorBoard
set /p start_tb="是否启动TensorBoard监控？(y/n): "
if /i "%start_tb%"=="y" (
    echo %INFO% 启动TensorBoard监控...
    echo 在浏览器中访问: http://localhost:6006
    echo 按 Ctrl+C 停止TensorBoard
    tensorboard --logdir outputs --port 6006
)

echo.
echo 🎉 感谢使用EfficientSAM微调脚本！
pause
exit /b 0

REM 显示帮助
:show_help
echo 用法: %~nx0 [选项]
echo.
echo 选项:
echo   install     安装依赖和环境
echo   test        运行环境测试
echo   configure   配置训练参数
echo   train       开始训练
echo   monitor     启动TensorBoard监控
echo   all         完整流程（安装+测试+配置+训练）
echo   help        显示此帮助信息
echo.
echo 默认执行完整流程
goto :eof

REM 主函数
:main
if "%~1"=="" goto all
if "%~1"=="install" goto check_python
if "%~1"=="test" goto run_test
if "%~1"=="configure" goto configure_training
if "%~1"=="train" goto start_training
if "%~1"=="monitor" (
    echo %INFO% 启动TensorBoard监控...
    echo 在浏览器中访问: http://localhost:6006
    tensorboard --logdir outputs --port 6006
    goto :eof
)
if "%~1"=="all" goto check_python
if "%~1"=="help" goto show_help

echo %ERROR% 未知选项: %~1
goto show_help

:all
goto check_python