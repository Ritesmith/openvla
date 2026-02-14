# OpenVLA 迁移指南 - 文件路径汇总

## 📂 项目文件结构

### 核心项目路径
```
项目根目录: d:\Stazica\Documents\GitHub\openvla
```

---

## 📦 关键文件和目录

### 1. 模型文件（已下载完成）

**本地模型路径**:
```
d:\Stazica\Documents\GitHub\openvla\huggingface\
```

**模型文件清单** (14.04 GB):
```
d:\Stazica\Documents\GitHub\openvla\huggingface\
├── config.json                                    (配置文件)
├── generation_config.json                         (生成配置)
├── model.safetensors.index.json                  (权重索引)
├── model-00001-of-00003.safetensors              (6.47 GB)
├── model-00002-of-00003.safetensors              (6.49 GB)
├── model-00003-of-00003.safetensors              (1.08 GB)
├── configuration_prismatic.py                     (模型配置类)
├── modeling_prismatic.py                          (模型定义)
├── preprocessor_config.json                      (预处理器配置)
├── processor_config.json                         (处理器配置)
├── processing_prismatic.py                        (处理逻辑)
├── tokenizer.json                                (Tokenizer 词表)
├── tokenizer.model                               (Tokenizer 模型)
├── tokenizer_config.json                         (Tokenizer 配置)
├── special_tokens_map.json                       (特殊 token 映射)
└── added_tokens.json                             (自定义 token)
```

**迁移提示**:
- 将 `huggingface/` 整个目录复制到你的新项目
- 或者在代码中使用绝对路径加载模型

---

### 2. 推理脚本

**已创建的演示脚本**:

| 文件 | 路径 | 用途 |
|------|------|------|
| `demo_local.py` | `d:\Stazica\Documents\GitHub\openvla\demo_local.py` | 本地模型推理演示（已测试成功） |
| `demo_custom.py` | `d:\Stazica\Documents\GitHub\openvla\demo_custom.py` | 自定义图像推理 |
| `simple_test.py` | `d:\Stazica\Documents\GitHub\openvla\simple_test.py` | 基础环境测试 |
| `test_env.py` | `d:\Stazica\Documents\GitHub\openvla\test_env.py` | 完整环境测试 |

**使用示例**:
```bash
cd d:\Stazica\Documents\GitHub\openvla

# 运行本地演示
python demo_local.py

# 使用自定义图像
python demo_custom.py --image "path/to/your/image.jpg" --instruction "pick up the object"
```

---

### 3. 核心源代码

#### 3.1 Prismatic 包（VLA 实现）
```
d:\Stazica\Documents\GitHub\openvla\prismatic\
```

**关键子目录**:

| 子目录 | 路径 | 用途 |
|--------|------|------|
| `conf/` | `d:\Stazica\Documents\GitHub\openvla\prismatic\conf\` | 配置文件（YAML） |
| `extern/` | `d:\Stazica\Documents\GitHub\openvla\prismatic\extern\` | 外部接口（HuggingFace 集成） |
| `models/` | `d:\Stazica\Documents\GitHub\openvla\prismatic\models\` | 模型定义（29 个文件） |
| `vla/` | `d:\Stazica\Documents\GitHub\openvla\prismatic\vla\` | VLA 相关代码（19 个文件） |
| `preprocessing/` | `d:\Stazica\Documents\GitHub\openvla\prismatic\preprocessing\` | 数据预处理 |
| `training/` | `d:\Stazica\Documents\GitHub\openvla\prismatic\training\` | 训练工具 |
| `util/` | `d:\Stazica\Documents\GitHub\openvla\prismatic\util\` | 通用工具 |

**迁移必需文件**:
```
# 核心模型实现
d:\Stazica\Documents\GitHub\openvla\prismatic\models\

# VLA 功能
d:\Stazica\Documents\GitHub\openvla\prismatic\vla\

# HuggingFace 接口
d:\Stazica\Documents\GitHub\openvla\prismatic\extern\hf\

# 配置文件
d:\Stazica\Documents\GitHub\openvla\prismatic\conf\
```

#### 3.2 VLA 脚本（训练/微调/部署）
```
d:\Stazica\Documents\GitHub\openvla\vla-scripts\
```

| 文件 | 路径 | 用途 |
|------|------|------|
| `train.py` | `d:\Stazica\Documents\GitHub\openvla\vla-scripts\train.py` | 完整模型训练 |
| `finetune.py` | `d:\Stazica\Documents\GitHub\openvla\vla-scripts\finetune.py` | LoRA 微调脚本 |
| `deploy.py` | `d:\Stazica\Documents\GitHub\openvla\vla-scripts\deploy.py` | REST API 部署 |
| `extern/` | `d:\Stazica\Documents\GitHub\openvla\vla-scripts\extern\` | 外部工具 |

**迁移必需文件**:
```python
# 推理和部署
d:\Stazica\Documents\GitHub\openvla\vla-scripts\deploy.py

# 微调
d:\Stazica\Documents\GitHub\openvla\vla-scripts\finetune.py

# 训练
d:\Stazica\Documents\GitHub\openvla\vla-scripts\train.py
```

---

### 4. 数据处理

#### 4.1 RLDS 数据集配置
```
d:\Stazica\Documents\GitHub\openvla\prismatic\vla\datasets\rlds\
```

**关键文件**:

| 文件 | 路径 | 用途 |
|------|------|------|
| `oxe/configs.py` | `d:\Stazica\Documents\GitHub\openvla\prismatic\vla\datasets\rlds\oxe\configs.py` | 数据集配置（需要添加你的数据集） |
| `oxe/transforms.py` | `d:\Stazica\Documents\GitHub\openvla\prismatic\vla\datasets\rlds\oxe\transforms.py` | 数据转换函数（需要添加转换） |
| `oxe/mixtures.py` | `d:\Stazica\Documents\GitHub\openvla\prismatic\vla\datasets\rlds\oxe\mixtures.py` | 数据集混合配置 |

**微调时需要修改**:
```python
# 在 configs.py 中添加你的机器人数据集配置
d:\Stazica\Documents\GitHub\openvla\prismatic\vla\datasets\rlds\oxe\configs.py

# 在 transforms.py 中添加你的数据转换函数
d:\Stazica\Documents\GitHub\openvla\prismatic\vla\datasets\rlds\oxe\transforms.py
```

---

### 5. 评估脚本

```
d:\Stazica\Documents\GitHub\openvla\experiments\robot\
```

| 文件 | 路径 | 用途 |
|------|------|------|
| `widowx/eval_bridge_v2.py` | `d:\Stazica\Documents\GitHub\openvla\experiments\robot\widowx\eval_bridge_v2.py` | BridgeData V2 评估 |
| `libero/` | `d:\Stazica\Documents\GitHub\openvla\experiments\robot\libero\` | LIBERO 模拟评估 |

---

### 6. 项目配置文件

| 文件 | 路径 | 用途 |
|------|------|------|
| `pyproject.toml` | `d:\Stazica\Documents\GitHub\openvla\pyproject.toml` | 项目依赖和配置 |
| `requirements-min.txt` | `d:\Stazica\Documents\GitHub\openvla\requirements-min.txt` | 最小依赖（推理用） |
| `Makefile` | `d:\Stazica\Documents\GitHub\openvla\Makefile` | 构建工具 |

---

## 🚀 迁移到你的项目

### 方案 A: 最小化迁移（仅推理）

**所需文件**:
```
你的项目/
├── models/
│   └── openvla-7b/           # 从 huggingface/ 复制
├── prismatic/
│   ├── models/
│   ├── vla/
│   └── extern/hf/
├── inference.py              # 使用 demo_local.py 的代码
└── requirements.txt          # 从 requirements-min.txt 复制
```

**代码示例**:
```python
import sys
import os

# 添加 OpenVLA 路径
openvla_path = "d:/Stazica/Documents/GitHub/openvla"
sys.path.insert(0, openvla_path)

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# 加载本地模型
model_path = "d:/Stazica/Documents/GitHub/openvla/huggingface"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True
)

# 推理
image = Image.open("your_image.jpg")
prompt = "In: pick up the object\nOut:"
inputs = processor(prompt, image).to("cpu", dtype=torch.float32)

with torch.no_grad():
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(action)
```

---

### 方案 B: 完整迁移（推理 + 微调）

**所需文件**:
```
你的项目/
├── models/
│   └── openvla-7b/           # 从 huggingface/ 复制
├── prismatic/                 # 完整复制
├── vla-scripts/               # 复制 train.py, finetune.py, deploy.py
├── pyproject.toml            # 复制
├── requirements-min.txt       # 复制
└── your_data/                # 你的机器人数据
    └── rlds_datasets/
```

**安装依赖**:
```bash
pip install -r d:/Stazica/Documents/GitHub/openvla/requirements-min.txt
```

---

### 方案 C: 原地使用（推荐）

直接在原项目中工作，只需添加你的代码和数据：

```
d:\Stazica\Documents\GitHub\openvla\
├── huggingface/              # 模型（已有）
├── prismatic/                # 核心代码（已有）
├── vla-scripts/              # 脚本（已有）
├── your_robot_code.py        # 你的机器人接口代码
└── your_data/                # 你的数据集
    └── rlds/
```

---

## 📋 环境配置总结

### 已安装的依赖

| 包 | 版本 | 用途 |
|---|------|------|
| Python | 3.12.4 | 运行环境 |
| PyTorch | 2.9.1+cpu | 深度学习框架 |
| Transformers | 4.40.1 | HuggingFace 模型库 |
| timm | 0.9.10 | 视觉模型库 |
| tokenizers | 0.19.1 | Tokenizer |
| accelerate | 最新版本 | 分布式训练 |
| pillow | 最新版本 | 图像处理 |

### 安装命令（在新环境中）
```bash
# 从 OpenVLA 项目复制 requirements 文件
pip install torch transformers timm tokenizers accelerate pillow

# 或使用 requirements-min.txt
pip install -r d:/Stazica/Documents/GitHub/openvla/requirements-min.txt
```

---

## 🔗 关键路径速查表

| 用途 | 路径 |
|------|------|
| **本地模型** | `d:\Stazica\Documents\GitHub\openvla\huggingface\` |
| **推理演示** | `d:\Stazica\Documents\GitHub\openvla\demo_local.py` |
| **自定义推理** | `d:\Stazica\Documents\GitHub\openvla\demo_custom.py` |
| **核心代码** | `d:\Stazica\Documents\GitHub\openvla\prismatic\` |
| **微调脚本** | `d:\Stazica\Documents\GitHub\openvla\vla-scripts\finetune.py` |
| **部署脚本** | `d:\Stazica\Documents\GitHub\openvla\vla-scripts\deploy.py` |
| **数据集配置** | `d:\Stazica\Documents\GitHub\openvla\prismatic\vla\datasets\rlds\oxe\configs.py` |
| **依赖列表** | `d:\Stazica\Documents\GitHub\openvla\requirements-min.txt` |
| **项目配置** | `d:\Stazica\Documents\GitHub\openvla\pyproject.toml` |

---

## 💡 快速开始代码模板

### 推理模板

```python
import sys
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

# 设置路径
OPENVLA_ROOT = r"d:\Stazica\Documents\GitHub\openvla"
sys.path.insert(0, OPENVLA_ROOT)

# 加载模型
MODEL_PATH = os.path.join(OPENVLA_ROOT, "huggingface")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32 if device.type == "cpu" else torch.bfloat16

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=dtype, low_cpu_mem_usage=True,
    trust_remote_code=True, local_files_only=True
).to(device)

# 推理函数
def predict_action(image_path, instruction):
    image = Image.open(image_path).convert("RGB")
    prompt = f"In: {instruction}\nOut:"
    inputs = processor(prompt, image).to(device, dtype=dtype)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    return action.numpy()

# 使用
action = predict_action("image.jpg", "pick up the red cup")
print(f"Action: {action}")
```

---

## 📚 文档路径

| 文档 | 路径 |
|------|------|
| 项目说明 | `d:\Stazica\Documents\GitHub\openvla\README.md` |
| 快速开始 | `d:\Stazica\Documents\GitHub\openvla\QUICKSTART.md` |
| 本地指南 | `d:\Stazica\Documents\GitHub\openvla\quickstart_guide.md` |
| 迁移指南 | `d:\Stazica\Documents\GitHub\openvla\MIGRATION_GUIDE.md` (本文件) |

---

## ✅ 迁移检查清单

### 最小化迁移（推理）
- [ ] 复制 `huggingface/` 到新项目
- [ ] 复制 `prismatic/models/` 到新项目
- [ ] 复制 `prismatic/vla/` 到新项目
- [ ] 复制 `prismatic/extern/hf/` 到新项目
- [ ] 安装 `requirements-min.txt` 中的依赖
- [ ] 使用 `demo_local.py` 代码模板

### 完整迁移（微调）
- [ ] 复制整个 `prismatic/` 目录
- [ ] 复制 `vla-scripts/` (train.py, finetune.py, deploy.py)
- [ ] 复制 `pyproject.toml`
- [ ] 安装所有依赖（包括 TensorFlow 用于 RLDS）
- [ ] 在 `configs.py` 中添加你的数据集配置
- [ ] 在 `transforms.py` 中添加数据转换函数
- [ ] 准备 RLDS 格式的机器人数据

---

## 🎯 下一步行动

1. **选择迁移方案**（A/B/C）
2. **复制必要文件**到新项目
3. **安装依赖**到新环境
4. **测试推理**使用 `demo_local.py` 代码
5. **（可选）准备数据集**用于微调
6. **（可选）运行微调**使用 `finetune.py`

---

**最后更新**: 2026-02-14
**OpenVLA 版本**: openvla-7b
