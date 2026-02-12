# AGENTS.md - LLM 微调项目

## 项目概述
这是一个使用 LoRA 和 DeepSpeed 进行大语言模型参数高效微调（PEFT）的项目。

## 技术栈
- PyTorch 2.2.2（CUDA 12.1）
- Transformers（HuggingFace）
- PEFT（LoRA）
- DeepSpeed（ZeRO Stage 2）
- Datasets（HuggingFace）
- uv（环境与依赖管理）

## 代码规范
- 使用 4 个空格缩进
- Python 代码遵循 PEP 8
- 为函数和类添加 docstring
- 在合适场景使用类型注解
- 每行长度不超过 100 个字符

## 项目结构
- `train.py` - 标准全参数微调脚本
- `train_peft.py` - 集成 DeepSpeed 的 LoRA/PEFT 微调脚本
- `ds_config.json` - DeepSpeed 配置（ZeRO Stage 2，FP16）
- `requirements.txt` - Python 依赖列表

## 常用命令

### 使用 uv 初始化环境
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 安装依赖（已激活虚拟环境时）
```bash
uv pip install -r requirements.txt
```

### 运行 LoRA 训练（单 GPU）
```bash
uv run python train_peft.py \
  --model_name_or_path ./pythia-410m \
  --dataset_name ./wikitext-2-raw-v1 \
  --output_dir ./outputs_pythia_peft \
  --per_device_train_batch_size 2 \
  --max_length 512 \
  --num_train_epochs 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
```

### 使用 DeepSpeed 运行 LoRA 训练（多 GPU）
```bash
CUDA_VISIBLE_DEVICES=0,1 uv run deepspeed train_peft.py \
  --model_name_or_path ./pythia-410m \
  --dataset_name ./wikitext-2-raw-v1 \
  --output_dir ./outputs_pythia_peft \
  --deepspeed ./ds_config.json \
  --per_device_train_batch_size 2 \
  --max_length 512 \
  --num_train_epochs 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
```

## LoRA 配置指南
- `lora_r`：LoRA 矩阵秩（默认 16）。越大容量越强，但参数也更多。
- `lora_alpha`：缩放系数（默认 32）。通常设置为 `lora_r` 的 2 倍。
- `target_modules`：应用 LoRA 的模块。对于 GPT-NeoX/Pythia：
  `query_key_value,dense,dense_h_to_4h,dense_4h_to_h`

## 关键约定
- 本地数据集始终使用 `load_from_disk()`
- 若 `pad_token` 为 `None`，设置 `tokenizer.pad_token = tokenizer.eos_token`
- 因果语言模型使用 `DataCollatorForLanguageModeling` 且 `mlm=False`
- 应用 PEFT 后调用 `model.print_trainable_parameters()` 打印可训练参数

## 给 AI Agent 的注意事项
- 修改训练脚本时，保留 DeepSpeed 集成
- 确保 LoRA 配置针对当前模型架构使用合适的 `target_modules`
- 优先使用较小的 `num_train_epochs` 和 `max_length` 进行快速测试
- 注意 CUDA 显存占用，必要时使用梯度累积
