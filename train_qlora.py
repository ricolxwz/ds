"""
QLoRA 训练脚本 - 4-bit NF4 量化 + LoRA 微调

CUDA_VISIBLE_DEVICES=0 deepspeed train_qlora.py \
  --model_name_or_path ./pythia-410m \
  --dataset_name ./wikitext-2-raw-v1 \
  --output_dir ./outputs_pythia_qlora \
  --deepspeed ./ds_config.json \
  --per_device_train_batch_size 2 \
  --max_length 512 \
  --num_train_epochs 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
"""

import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer# # 计算时使用的精度
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--output_dir", type=str, default="./outputs_qlora")
    p.add_argument("--deepspeed", type=str, default="./ds_config.json")

    # QLoRA 量化参数
    p.add_argument("--bnb_4bit_quant_type", type=str, default="nf4",
                   help="Quantization type: nf4 or fp4")
    p.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16",
                   help="Compute dtype for quantization: float16, bfloat16, float32")
    p.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True,
                   help="Use double quantization for further memory reduction")

    # LoRA 参数
    p.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (rank)")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter for scaling")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
    p.add_argument("--target_modules", type=str, default="query_key_value,dense",
                   help="Comma-separated list of target modules to apply LoRA")
    p.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"],
                   help="Bias training type")

    # 训练参数
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--do_eval", action="store_true")
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 配置量化 (4-bit NF4)
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 模型以 4-bit 精度加载, 而非默认的 FP16/FP32
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,  # 使用 NF4 (Normal Float 4-bit) 算法, 针对正态分布的权重优化
        bnb_4bit_compute_dtype=compute_dtype,  # 虽然模型是 4-bit 存储, 但计算时升回 float16, 保证训练精度
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,  # 对量化常数再量化, 可额外节省约 0.5 bit/参数
    )

    print(f"量化配置: 4-bit {args.bnb_4bit_quant_type}, 计算类型: {compute_dtype}")
    print(f"双量化: {'启用' if args.bnb_4bit_use_double_quant else '禁用'}")

    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 禁用 KV cache 缓存以支持 gradient checkpointing
    model.config.use_cache = False
    # QLoRA 训练前必须做 k-bit 训练准备, 否则可能出现梯度为 None
    model = prepare_model_for_kbit_training(model)

    # 4. 配置 LoRA
    target_modules = args.target_modules.split(",") if args.target_modules else None

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    # 5. 应用 PEFT/LoRA 到量化模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. 加载数据集
    raw = load_from_disk(args.dataset_name)

    def tokenize_fn(examples):
        texts = examples[args.text_field]
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing",
    )

    # 过滤空样本
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)

    # 7. 数据整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 8. 训练参数 (启用 gradient checkpointing)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps if args.do_eval else None,
        fp16=True,
        bf16=False,
        deepspeed=args.deepspeed,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,  # QLoRA 必需: 节省显存
    )

    # 9. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None) if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 10. 训练
    trainer.train()

    # 11. 保存模型 (只保存 LoRA 权重)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"QLoRA 训练完成! 模型已保存到: {args.output_dir}")
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()
