"""
OpenVLA 本地模型推理演示
使用本地下载的模型进行推理
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def check_environment():
    """检查环境"""
    print("="*60)
    print("OpenVLA 推理演示 - 环境检查")
    print("="*60)
    print()

    # Python 版本
    print(f"[1/4] Python: {sys.version.split()[0]}")

    # PyTorch
    print(f"[2/4] PyTorch: {torch.__version__}")

    # GPU
    if torch.cuda.is_available():
        print(f"[3/4] GPU: {torch.cuda.get_device_name(0)}")
        print(f"       显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = torch.device("cuda:0")
    else:
        print("[3/4] GPU: 不可用 (使用 CPU)")
        device = torch.device("cpu")

    # Transformers
    import transformers
    print(f"[4/4] Transformers: {transformers.__version__}")

    print()
    print("="*60)
    return device


def load_model_local(model_path, device):
    """加载本地模型"""
    print("正在加载本地模型...")
    print()

    # 加载处理器
    print("[1/3] 加载处理器...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    print("[OK] 处理器加载完成")

    # 加载模型
    print()
    print("[2/3] 加载模型权重...")
    print("  (这可能需要几分钟时间)")

    # 根据设备选择数据类型
    if device.type == "cpu":
        dtype = torch.float32
        attn_impl = None
    else:
        dtype = torch.bfloat16
        attn_impl = "flash_attention_2"

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
        print("[OK] 模型加载完成")
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        print("尝试不使用 flash_attention_2...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
        print("[OK] 模型加载完成 (标准注意力)")

    print()
    print(f"[3/3] 设备: {device}")
    print(f"     数据类型: {dtype}")

    return model, processor, dtype


def create_test_image():
    """创建测试图像"""
    print()
    print("创建测试图像...")

    # 创建一个带有简单图案的测试图像
    # 模拟机器人视角 - 前景有一个"物体"，背景均匀
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)

    # 背景 - 灰色
    img_array[:, :] = [100, 100, 100]

    # 模拟一个"红色杯子"在中间
    # 红色圆圈
    for i in range(224):
        for j in range(224):
            if (i-112)**2 + (j-112)**2 < 50**2:
                img_array[i, j] = [200, 50, 50]

    # 杯子边缘 - 白色
    for i in range(224):
        for j in range(224):
            dist = (i-112)**2 + (j-112)**2
            if 48**2 < dist < 50**2:
                img_array[i, j] = [255, 255, 255]

    image = Image.fromarray(img_array)
    print("[OK] 测试图像创建完成 (224x224 RGB)")
    print("     模拟场景: 桌面上的红色杯子")

    return image


def run_inference(model, processor, image, prompt, device, dtype):
    """运行推理"""
    print()
    print("="*60)
    print("推理测试")
    print("="*60)
    print()
    print(f"任务描述: {prompt}")
    print()

    # 准备输入
    print("[1/2] 处理输入...")
    inputs = processor(prompt, image).to(device, dtype=dtype)

    # 预测动作
    print("[2/2] 预测动作...")
    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",  # 使用 BridgeData V2 归一化
            do_sample=False
        )

    print("[OK] 动作预测完成")
    print()

    return action


def display_action(action):
    """显示预测的动作"""
    print("="*60)
    print("预测动作 (7-DoF)")
    print("="*60)
    print()

    if isinstance(action, (list, np.ndarray, torch.Tensor)):
        action = np.array(action)
        if action.ndim == 1 and action.shape[0] >= 7:
            # 7-DoF 动作: [x, y, z, roll, pitch, yaw, gripper]
            print(f"位置 (x, y, z):      [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
            print(f"旋转 (roll, pitch):   [{action[3]:.4f}, {action[4]:.4f}]")
            print(f"旋转 (yaw):           [{action[5]:.4f}]")
            print(f"夹爪 (gripper):       [{action[6]:.4f}]")
            print()
            print("动作描述:")
            if action[6] > 0.5:
                print("  - 移动机器人末端到目标位置")
                print("  - 旋转到指定姿态")
                print("  - 打开/调整夹爪以抓取物体")
            else:
                print("  - 移动机器人末端到目标位置")
                print("  - 旋转到指定姿态")
                print("  - 关闭夹爪")
        else:
            print(f"原始动作: {action}")
    else:
        print(f"动作类型: {type(action)}")
        print(f"动作值: {action}")

    print()


def main():
    """主函数"""
    # 检查环境
    device = check_environment()

    # 本地模型路径
    model_path = os.path.join(project_root, "huggingface")

    # 检查模型文件是否存在
    print(f"模型路径: {model_path}")
    print()
    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors"
    ]

    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**3)  # GB
            print(f"  [OK] {file:40s} ({file_size:.2f} GB)")
        else:
            print(f"  [X]  {file:40s} 缺失!")
            missing_files.append(file)

    if missing_files:
        print()
        print("[ERROR] 缺少必要的模型文件!")
        print(f"缺失文件: {missing_files}")
        print("请确保模型下载完成后再运行此脚本")
        return

    print()
    print("="*60)

    # 加载模型
    model, processor, dtype = load_model_local(model_path, device)

    # 创建测试图像
    image = create_test_image()

    # 测试提示词
    test_prompts = [
        "In: What action should the robot take to pick up the red cup?\nOut:",
        "In: Pick up the red object\nOut:",
        "In: Grasp the red cup\nOut:",
    ]

    # 运行推理
    for i, prompt in enumerate(test_prompts, 1):
        print()
        print(f"测试 {i}/{len(test_prompts)}")
        print()

        action = run_inference(model, processor, image, prompt, device, dtype)
        display_action(action)

    print()
    print("="*60)
    print("[完成] 所有推理测试已完成")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[中断] 用户取消操作")
    except Exception as e:
        print(f"\n\n[错误] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
