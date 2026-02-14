"""
使用自定义图像进行 OpenVLA 推理
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import sys
import os
import argparse

project_root = os.path.dirname(os.path.abspath(__file__))


def load_model(model_path, device):
    """加载模型"""
    print("正在加载模型...")

    # CPU 使用 float32
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

    print(f"[OK] 模型加载完成 (设备: {device})")
    return model, processor, dtype


def predict_action(model, processor, image_path, instruction, device, dtype):
    """预测动作"""
    print(f"\n图像: {image_path}")
    print(f"指令: {instruction}")

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    print(f"图像尺寸: {image.size}")

    # 准备输入
    prompt = f"In: {instruction}\nOut:"
    inputs = processor(prompt, image).to(device, dtype=dtype)

    # 预测
    print("正在预测...")
    with torch.no_grad():
        action = model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False
        )

    # 显示结果
    action = np.array(action)
    print(f"\n[预测动作]")
    print(f"位置: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
    print(f"旋转: [{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
    print(f"夹爪: [{action[6]:.4f}]")

    return action


def main():
    parser = argparse.ArgumentParser(description="OpenVLA 自定义图像推理")
    parser.add_argument("--image", required=True, help="图像文件路径")
    parser.add_argument("--instruction", default="pick up the object",
                        help="机器人任务指令")
    args = parser.parse_args()

    # 检查设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 加载模型
    model_path = os.path.join(project_root, "huggingface")
    model, processor, dtype = load_model(model_path, device)

    # 预测
    predict_action(
        model, processor,
        args.image, args.instruction,
        device, dtype
    )


if __name__ == "__main__":
    main()
