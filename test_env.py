"""
简化的环境测试脚本
"""
import sys
import torch
import numpy as np
from PIL import Image

print("="*60)
print("OpenVLA 环境检查")
print("="*60)

# 1. Python 版本
print(f"\n[1/6] Python 版本: {sys.version}")

# 2. PyTorch
print(f"[2/6] PyTorch: {torch.__version__}")
print(f"      设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"      GPU: {torch.cuda.get_device_name(0)}")

# 3. Transformers
try:
    import transformers
    print(f"[3/6] Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"[3/6] Transformers: 未安装 ({e})")

# 4. 其他依赖
try:
    import timm
    print(f"[4/6] timm: {timm.__version__}")
except ImportError as e:
    print(f"[4/6] timm: 未安装 ({e})")

try:
    import tokenizers
    print(f"[4/6] tokenizers: {tokenizers.__version__}")
except ImportError as e:
    print(f"[4/6] tokenizers: 未安装 ({e})")

# 5. 加载 OpenVLA
print("\n[5/6] 尝试加载 OpenVLA 模型...")
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq

    print("      正在从 HuggingFace 下载模型（首次运行需要一些时间）...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    print("      [OK] 处理器加载成功")

    # CPU 模式加载模型
    device = torch.device("cpu")
    dtype = torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    print("      [OK] 模型加载成功")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {total_params / 1e9:.2f} B")

except Exception as e:
    print(f"      [ERROR] 加载失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 测试推理
print("\n[6/6] 测试推理...")
try:
    # 创建测试图像
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_image = Image.fromarray(image_array)
    prompt = "In: What action should the robot take to pick up the red cup?\nOut:"

    # 处理输入
    inputs = processor(prompt, test_image).to(device, dtype=dtype)

    # 预测
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    print(f"      [OK] 推理成功")
    print(f"      预测动作: {action}")
    print(f"      动作形状: {action.shape}")

except Exception as e:
    print(f"      [ERROR] 推理失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("环境检查完成")
print("="*60)
