#!/bin/bash

# OpenVLA 快速入门 - 安装和测试脚本

echo "============================================================"
echo "OpenVLA 快速入门 - 安装脚本"
echo "============================================================"

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "✗ 错误: 未找到 conda，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

echo "✓ 找到 conda"

# 检查是否已存在 openvla 环境
if conda env list | grep -q "^openvla "; then
    echo "⚠ 警告: openvla 环境已存在"
    read -p "是否要删除并重建？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除旧环境..."
        conda env remove -n openvla -y
    else
        echo "使用现有环境"
    fi
fi

# 创建新的 conda 环境
echo ""
echo "============================================================"
echo "[1/4] 创建 conda 环境"
echo "============================================================"
conda create -n openvla python=3.10 -y
echo "✓ 环境创建完成"

# 激活环境
echo ""
echo "============================================================"
echo "[2/4] 激活环境"
echo "============================================================"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate openvla
echo "✓ 环境已激活: openvla"

# 安装 PyTorch
echo ""
echo "============================================================"
echo "[3/4] 安装 PyTorch"
echo "============================================================"

# 检测 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到 NVIDIA GPU"
    echo "安装 CUDA 版本的 PyTorch..."
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    echo "⚠ 未检测到 NVIDIA GPU，安装 CPU 版本的 PyTorch"
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

echo "✓ PyTorch 安装完成"

# 安装最小依赖
echo ""
echo "============================================================"
echo "[4/4] 安装 OpenVLA 依赖"
echo "============================================================"

# 从 requirements-min.txt 安装（如果文件存在）
if [ -f "requirements-min.txt" ]; then
    echo "从 requirements-min.txt 安装..."
    pip install -r requirements-min.txt
else
    # 如果文件不存在，手动安装核心依赖
    echo "安装核心依赖包..."
    pip install torch transformers timm tokenizers accelerate
fi

echo "✓ 依赖安装完成"

# 安装包（开发模式）
echo ""
echo "============================================================"
echo "[5/5] 安装 OpenVLA 包"
echo "============================================================"
pip install -e .
echo "✓ OpenVLA 安装完成"

# 可选：安装 Flash Attention
echo ""
echo "============================================================"
echo "Flash Attention（可选）"
echo "============================================================"
read -p "是否安装 Flash Attention 2？(需要 CUDA) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装 Flash Attention 2..."
    pip install packaging ninja
    ninja --version
    pip install "flash-attn==2.5.5" --no-build-isolation
    echo "✓ Flash Attention 2 安装完成"
else
    echo "跳过 Flash Attention 2（会使用默认的注意力机制）"
fi

echo ""
echo "============================================================"
echo "✓ 安装完成！"
echo "============================================================"
echo ""
echo "下一步："
echo "  1. 激活环境: conda activate openvla"
echo "  2. 运行演示: python quickstart_demo.py"
echo "  3. 交互模式: python quickstart_demo.py interactive"
echo ""
echo "============================================================"
