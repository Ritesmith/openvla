# OpenVLA 快速入门 - Windows PowerShell 安装脚本

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "OpenVLA 快速入门 - 安装脚本 (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 检查 conda 是否可用
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "✗ 错误: 未找到 conda，请先安装 Anaconda 或 Miniconda" -ForegroundColor Red
    exit 1
}

Write-Host "✓ 找到 conda" -ForegroundColor Green

# 检查是否已存在 openvla 环境
$envExists = conda env list | Select-String "^openvla "

if ($envExists) {
    Write-Host "⚠ 警告: openvla 环境已存在" -ForegroundColor Yellow
    $response = Read-Host "是否要删除并重建？(y/n)"
    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host "删除旧环境..."
        conda env remove -n openvla -y
    } else {
        Write-Host "使用现有环境"
    }
}

# 创建新的 conda 环境
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[1/4] 创建 conda 环境" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
conda create -n openvla python=3.10 -y
Write-Host "✓ 环境创建完成" -ForegroundColor Green

# 激活环境
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[2/4] 激活环境" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
conda activate openvla
Write-Host "✓ 环境已激活: openvla" -ForegroundColor Green

# 安装 PyTorch
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[3/4] 安装 PyTorch" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 检测 GPU
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "✓ 检测到 NVIDIA GPU" -ForegroundColor Green
    Write-Host "安装 CUDA 版本的 PyTorch..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
} else {
    Write-Host "⚠ 未检测到 NVIDIA GPU，安装 CPU 版本的 PyTorch" -ForegroundColor Yellow
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
}

Write-Host "✓ PyTorch 安装完成" -ForegroundColor Green

# 安装最小依赖
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[4/4] 安装 OpenVLA 依赖" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 从 requirements-min.txt 安装（如果文件存在）
if (Test-Path "requirements-min.txt") {
    Write-Host "从 requirements-min.txt 安装..."
    pip install -r requirements-min.txt
} else {
    # 如果文件不存在，手动安装核心依赖
    Write-Host "安装核心依赖包..."
    pip install torch transformers timm tokenizers accelerate
}

Write-Host "✓ 依赖安装完成" -ForegroundColor Green

# 安装包（开发模式）
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[5/5] 安装 OpenVLA 包" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
pip install -e .
Write-Host "✓ OpenVLA 安装完成" -ForegroundColor Green

# 可选：安装 Flash Attention
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Flash Attention（可选）" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
$response = Read-Host "是否安装 Flash Attention 2？(需要 CUDA) (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "安装 Flash Attention 2..."
    pip install packaging ninja
    ninja --version
    pip install "flash-attn==2.5.5" --no-build-isolation
    Write-Host "✓ Flash Attention 2 安装完成" -ForegroundColor Green
} else {
    Write-Host "跳过 Flash Attention 2（会使用默认的注意力机制）"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✓ 安装完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：" -ForegroundColor Yellow
Write-Host "  1. 激活环境: conda activate openvla"
Write-Host "  2. 运行演示: python quickstart_demo.py"
Write-Host "  3. 交互模式: python quickstart_demo.py interactive"
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
