"""
简化测试 - 验证环境基础功能（无需下载模型）
"""
import torch
import numpy as np
from PIL import Image
import sys

print("="*60)
print("OpenVLA 环境基础测试")
print("="*60)

# 1. Python 版本
print(f"\n[1/5] Python: {sys.version.split()[0]}")

# 2. PyTorch
print(f"[2/5] PyTorch: {torch.__version__}")
print(f"      设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"      GPU: {torch.cuda.get_device_name(0)}")

# 3. PyTorch 运算测试
print("\n[3/5] PyTorch 运算测试...")
try:
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    z = torch.matmul(x, y)
    print(f"      [OK] 矩阵乘法成功，结果形状: {z.shape}")

    # GPU 测试
    if torch.cuda.is_available():
        x_gpu = x.to("cuda")
        y_gpu = y.to("cuda")
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print(f"      [OK] GPU 运算成功，结果形状: {z_gpu.shape}")

except Exception as e:
    print(f"      [ERROR] 运算失败: {e}")

# 4. 图像处理测试
print("\n[4/5] 图像处理测试...")
try:
    # 创建测试图像
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    print(f"      [OK] 图像创建成功: {img.size} {img.mode}")

    # 图像变换测试
    resized = img.resize((256, 256))
    rotated = img.rotate(90)
    print(f"      [OK] 图像变换成功")

except Exception as e:
    print(f"      [ERROR] 图像处理失败: {e}")

# 5. Transformers 基础测试
print("\n[5/5] Transformers 基础测试...")
try:
    from transformers import AutoTokenizer

    # 使用一个轻量级的 tokenizer 测试
    print("      正在加载 tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = "Hello, OpenVLA!"
    encoded = tokenizer(text, return_tensors="pt")
    print(f"      [OK] Tokenizer 加载成功")
    print(f"      输入文本: '{text}'")
    print(f"      编码形状: {encoded['input_ids'].shape}")

except Exception as e:
    print(f"      [ERROR] Transformers 测试失败: {e}")

print("\n" + "="*60)
print("基础测试完成")
print("="*60)

print("\n下一步:")
print("1. 如果所有测试通过，说明环境配置正确")
print("2. 现在需要解决网络问题以下载 OpenVLA 模型")
print("3. 查看 quickstart_guide.md 了解如何下载模型")
