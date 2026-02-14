"""
OpenVLA 快速入门 - 推理测试脚本
这个脚本演示如何加载 OpenVLA 模型并进行简单的动作预测
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np


def check_gpu():
    """检查 GPU 是否可用"""
    if torch.cuda.is_available():
        print(f"[OK] GPU 可用: {torch.cuda.get_device_name(0)}")
        print(f"  总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return torch.device("cuda:0")
    else:
        print("[WARNING] GPU 不可用，将使用 CPU（推理速度会较慢）")
        return torch.device("cpu")


def load_openvla_model(model_path="openvla/openvla-7b", device="cuda:0"):
    """加载 OpenVLA 模型和处理器"""
    print(f"\n{'='*60}")
    print(f"正在加载 OpenVLA 模型: {model_path}")
    print(f"{'='*60}")

    # 加载处理器
    print("\n[1/3] 加载处理器...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("[OK] 处理器加载完成")

    # 加载模型
    print("\n[2/3] 加载模型...")
    # CPU 不支持 flash_attention_2 和 bfloat16
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    attn_impl = None if device.type == "cpu" else "flash_attention_2"

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    print(f"[OK] 模型加载完成，设备: {device}")

    # 显示模型信息
    print("\n[3/3] 模型信息:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params / 1e9:.2f} B")
    print(f"  可训练参数: {trainable_params / 1e9:.2f} B")

    print(f"\n{'='*60}")
    print("[OK] 模型加载完成！")
    print(f"{'='*60}\n")

    return processor, model


def create_test_image():
    """创建一个测试图像（随机噪声）"""
    print("创建测试图像...")
    # 创建一个 224x224 的随机图像
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    print("[OK] 测试图像创建完成（224x224 RGB）")
    return image


def predict_action(processor, model, image, instruction, device="cuda:0"):
    """预测动作"""
    print(f"\n{'='*60}")
    print(f"任务指令: {instruction}")
    print(f"{'='*60}")

    # 构建提示词
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    print(f"\n提示词:\n{prompt}")

    # 准备输入
    print("\n正在处理输入...")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    inputs = processor(prompt, image).to(device, dtype=dtype)
    print("[OK] 输入处理完成")

    # 预测动作
    print("\n正在预测动作...")
    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",  # 用于 BridgeData V2 的归一化
            do_sample=False
        )

    print("[OK] 动作预测完成")

    return action


def display_action(action):
    """显示预测的动作"""
    print(f"\n{'='*60}")
    print("预测动作（7-DoF）:")
    print(f"{'='*60}")

    # BridgeData V2 的动作格式: [x, y, z, roll, pitch, yaw, gripper]
    print(f"\n位置 (XYZ):")
    print(f"  x: {action[0]:.4f}")
    print(f"  y: {action[1]:.4f}")
    print(f"  z: {action[2]:.4f}")

    print(f"\n旋转 (RPY):")
    print(f"  roll:  {action[3]:.4f}")
    print(f"  pitch: {action[4]:.4f}")
    print(f"  yaw:   {action[5]:.4f}")

    print(f"\n夹爪:")
    print(f"  state: {action[6]:.4f} (0=closed, 1=open)")

    print(f"\n动作向量: {action}")
    print(f"{'='*60}\n")


def run_demo():
    """运行完整演示"""
    print("\n" + "="*60)
    print("OpenVLA 快速入门 - 推理演示")
    print("="*60)

    # 1. 检查 GPU
    device = check_gpu()

    # 2. 加载模型
    processor, model = load_openvla_model(device=device)

    # 3. 创建测试图像
    test_image = create_test_image()

    # 4. 测试不同的任务指令
    test_instructions = [
        "pick up the red cup",
        "push the blue block to the left",
        "place the object on the table",
        "open the drawer",
    ]

    print("\n" + "="*60)
    print("开始推理测试")
    print("="*60)

    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n\n[测试 {i}/{len(test_instructions)}]")

        # 预测动作
        action = predict_action(processor, model, test_image, instruction, device)

        # 显示结果
        display_action(action)

        if i < len(test_instructions):
            input("\n按 Enter 继续下一个测试...")

    print("\n" + "="*60)
    print("✓ 所有测试完成！")
    print("="*60)


def interactive_demo(processor, model, device="cuda:0"):
    """交互式演示 - 使用用户提供的图像"""
    print("\n" + "="*60)
    print("交互式推理模式")
    print("="*60)

    print("\n提示: 准备一张包含机器人视角的图像（推荐 224x224 或更大）")

    while True:
        try:
            # 获取图像路径
            image_path = input("\n请输入图像路径（或输入 'q' 退出）: ").strip()

            if image_path.lower() == 'q':
                print("退出交互模式...")
                break

            # 加载图像
            try:
                image = Image.open(image_path).convert("RGB")
                print(f"✓ 图像加载成功: {image.size}")
            except FileNotFoundError:
                print(f"✗ 错误: 找不到文件 '{image_path}'")
                continue
            except Exception as e:
                print(f"✗ 错误: 无法加载图像 - {e}")
                continue

            # 获取指令
            instruction = input("\n请输入任务指令（例如: 'pick up the red cup'）: ").strip()

            if not instruction:
                print("✗ 错误: 指令不能为空")
                continue

            # 预测动作
            action = predict_action(processor, model, image, instruction, device)
            display_action(action)

        except KeyboardInterrupt:
            print("\n\n中断，退出...")
            break


def main():
    """主函数"""
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        # 交互模式
        device = check_gpu()
        processor, model = load_openvla_model(device=device)
        interactive_demo(processor, model, device)
    else:
        # 自动演示模式
        run_demo()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("OpenVLA 快速入门")
    print("="*60)
    print("\n使用方法:")
    print("  python quickstart_demo.py              # 运行自动演示")
    print("  python quickstart_demo.py interactive   # 交互模式（使用你自己的图像）")
    print("\n" + "="*60 + "\n")

    main()
