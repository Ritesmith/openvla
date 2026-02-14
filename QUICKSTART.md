# OpenVLA 快速入门指南

欢迎学习 OpenVLA！本指南将帮助你快速上手并运行第一个推理示例。

---

## 📋 前置要求

### 必需
- Python >= 3.8
- conda（Anaconda 或 Miniconda）
- 约 15GB 磁盘空间（用于下载模型）

### 推荐
- NVIDIA GPU（至少 16GB 显存，推荐 A100 80GB）
- CUDA 12.1+

---

## 🚀 快速安装

### Windows 用户

在 PowerShell 中运行：

```powershell
# 运行安装脚本
.\install_and_test.ps1

# 如果遇到执行策略限制，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux/Mac 用户

在终端中运行：

```bash
# 运行安装脚本
chmod +x install_and_test.sh
./install_and_test.sh
```

### 手动安装（可选）

如果安装脚本不工作，可以手动执行：

```bash
# 1. 创建并激活 conda 环境
conda create -n openvla python=3.10 -y
conda activate openvla

# 2. 安装 PyTorch（根据你的平台选择）
# CUDA 版本（有 NVIDIA GPU）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# CPU 版本（无 GPU）
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 3. 安装最小依赖
pip install -r requirements-min.txt

# 4. 安装 OpenVLA 包（开发模式）
pip install -e .

# 5. 可选：安装 Flash Attention 2（仅 CUDA）
pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

---

## 🎯 运行演示

### 方式 1：自动演示（推荐首次使用）

```bash
# 激活环境
conda activate openvla

# 运行自动演示
python quickstart_demo.py
```

这将：
- 检查 GPU 可用性
- 下载 OpenVLA-7B 模型（约 14GB）
- 创建测试图像
- 对多个任务指令进行推理
- 显示预测的 7-DoF 动作

### 方式 2：交互模式（使用你自己的图像）

```bash
conda activate openvla
python quickstart_demo.py interactive
```

这将：
- 让你提供自己的图像文件
- 输入自定义的任务指令
- 实时预测并显示动作

---

## 📊 输出说明

### 动作格式（7-DoF）

```
位置 (XYZ):
  x: <位置 X>
  y: <位置 Y>
  z: <位置 Z>

旋转 (RPY):
  roll:  <滚转角度>
  pitch: <俯仰角度>
  yaw:   <偏航角度>

夹爪:
  state: <0=closed, 1=open>
```

### 推理速度参考

- **A100 80GB**: ~50-100 ms/动作
- **RTX 3090**: ~100-200 ms/动作
- **RTX 3080**: ~150-250 ms/动作
- **CPU**: ~5-10 s/动作

---

## 🔧 常见问题

### Q1: 模型下载很慢或失败？

**解决方案**:
1. 使用镜像源配置 HuggingFace
2. 手动下载模型并放在本地目录
3. 使用较小的模型（如 openvla-v01-7b）

```python
# 使用本地模型
model_path = "/path/to/local/model"
vla = AutoModelForVision2Seq.from_pretrained(model_path, ...)
```

### Q2: 显存不足（OOM）？

**解决方案**:
```python
# 使用低精度推理
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.float16,  # 或 torch.float32
    low_cpu_mem_usage=True,
    ...
)

# 或使用 CPU
device = "cpu"
```

### Q3: Flash Attention 安装失败？

**解决方案**:
Flash Attention 是可选的，不安装也能运行，只是推理速度稍慢。可以跳过该步骤继续使用。

### Q4: 推理结果不合理？

**原因**:
- 使用随机测试图像，模型无法理解场景
- 需要使用真实的机器人视角图像

**解决方案**:
```bash
# 使用交互模式，提供真实的机器人图像
python quickstart_demo.py interactive
```

---

## 📚 下一步学习

### 1. 理解模型架构
- 查看 `prismatic/models/` 目录
- 阅读模型定义文件

### 2. 学习微调
- 阅读 `vla-scripts/finetune.py`
- 准备你自己的数据集
- 运行 LoRA 微调

### 3. 部署到机器人
- 运行 `vla-scripts/deploy.py` 启动 REST API
- 编写机器人控制脚本调用 API

### 4. 深入研究
- 阅读 `README.md` 的完整文档
- 查看 `experiments/` 中的评估脚本
- 研究论文: https://arxiv.org/abs/2406.09246

---

## 🎓 示例代码

### 简单推理示例

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# 加载模型
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda:0")

# 准备输入
image = Image.open("robot_view.jpg")
prompt = "In: What action should the robot take to pick up the red cup?\nOut:"

# 预测动作
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"预测动作: {action}")
```

### 批量推理

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to("cuda:0")

# 多个任务
tasks = [
    ("image1.jpg", "pick up the red cup"),
    ("image2.jpg", "push the blue block"),
    ("image3.jpg", "open the drawer"),
]

for image_path, instruction in tasks:
    image = Image.open(image_path)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(f"{instruction}: {action}")
```

---

## 📖 参考资源

- **项目主页**: https://github.com/openvla/openvla
- **论文**: https://arxiv.org/abs/2406.09246
- **HuggingFace**: https://huggingface.co/openvla
- **项目网站**: https://openvla.github.io/

---

## 💡 提示

- 首次运行会下载模型，请确保网络连接稳定
- 如果没有 GPU，推理速度会很慢，但功能完全正常
- 使用真实机器人视角图像会得到更合理的动作预测
- 建议从简单的任务开始，逐步尝试更复杂的指令

---

## 🆘 需要帮助？

如果遇到问题：
1. 查看本指南的"常见问题"部分
2. 检查 README.md 的详细文档
3. 在 GitHub Issues 中搜索类似问题
4. 创建新的 Issue 描述你的问题

---

祝你学习愉快！🎉
