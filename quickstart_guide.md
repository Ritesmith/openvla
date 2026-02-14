# OpenVLA 快速入门 - 安装完成状态

## 当前状态

✓ **已安装的依赖：**
- Python 3.12.4
- PyTorch 2.9.1 (CPU 版本)
- Transformers 4.40.1
- timm 0.9.10
- tokenizers 0.19.1

✓ **创建的文件：**
- `quickstart_demo.py` - 完整演示脚本
- `test_env.py` - 环境测试脚本
- `QUICKSTART.md` - 详细使用指南

## 遇到的问题

### SSL 证书验证错误

你的网络无法访问 HuggingFace（SSL 证书验证失败）。这是网络配置问题，不是代码问题。

## 解决方案

### 方案 1：手动下载模型（推荐）

1. **访问 HuggingFace 网页下载**
   - 打开浏览器访问: https://huggingface.co/openvla/openvla-7b
   - 点击 "Files and versions"
   - 下载所有文件到本地目录，例如：`C:\openvla_models\openvla-7b\`

2. **修改脚本使用本地模型**

打开 `test_env.py` 或 `quickstart_demo.py`，修改模型路径：

```python
# 将这一行：
model_path = "openvla/openvla-7b"

# 改为：
model_path = r"C:\openvla_models\openvla-7b"
```

3. **再次运行**

```bash
cd d:\Stazica\Documents\GitHub\openvla
python test_env.py
```

### 方案 2：使用镜像站点

如果可以访问镜像站点，可以配置 HuggingFace 镜像：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModelForVision2Seq, AutoProcessor
# ... 正常使用
```

### 方案 3：使用代理（如果有）

如果你有代理服务器，可以配置环境变量：

```bash
set HTTP_PROXY=http://your-proxy:port
set HTTPS_PROXY=http://your-proxy:port
python test_env.py
```

### 方案 4：禁用 SSL 验证（不推荐，仅测试用）

```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import AutoModelForVision2Seq, AutoProcessor
# ... 正常使用
```

## 简化测试 - 无需下载模型

如果你想先验证环境是否正确，可以运行这个简化测试：

```python
import torch
import numpy as np
from PIL import Image

print("PyTorch 版本:", torch.__version__)
print("设备:", "CUDA" if torch.cuda.is_available() else "CPU")

# 创建测试张量
x = torch.randn(2, 3)
y = torch.randn(2, 3)
z = x + y
print("张量运算测试:", z.shape)

# 创建测试图像
img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
print("图像创建测试:", img.size)

print("\n环境基础功能正常！")
```

## 下一步

### 1. 解决网络问题

选择上述任一方案解决 HuggingFace 访问问题。

### 2. 下载模型

模型大小约 14GB，请确保有足够的磁盘空间。

### 3. 运行演示

```bash
# 自动演示模式
python quickstart_demo.py

# 交互模式（需要真实的机器人图像）
python quickstart_demo.py interactive
```

### 4. 学习微调

当环境完全配置好后，可以开始学习如何微调模型：

1. 准备你的机器人数据集
2. 参考 QUICKSTART.md 中的微调教程
3. 运行 `vla-scripts/finetune.py`

## 预期运行时间

- **首次运行**（下载模型）: 30-60 分钟（取决于网络速度）
- **后续运行**（CPU 推理）: 5-10 秒/动作
- **后续运行**（GPU 推理）: 0.05-0.1 秒/动作

## 需要帮助？

如果遇到其他问题：

1. 检查磁盘空间是否充足（至少 20GB）
2. 确认 Python 版本兼容性（3.8+）
3. 查看错误信息的完整堆栈跟踪
4. 可以尝试使用 conda 创建独立环境（见 QUICKSTART.md）

---

## 创建的文件说明

| 文件 | 说明 |
|------|------|
| `quickstart_demo.py` | 完整的推理演示，包含自动模式和交互模式 |
| `test_env.py` | 环境验证脚本 |
| `install_and_test.ps1` | Windows PowerShell 安装脚本 |
| `install_and_test.sh` | Linux/Mac 安装脚本 |
| `QUICKSTART.md` | 完整的快速入门指南 |
| `quickstart_guide.md` | 本文件 |

祝你学习顺利！🚀
