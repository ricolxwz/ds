"""
CUDA_VISIBLE_DEVICES=0 deepspeed train_peft.py \
  --model_name_or_path ./pythia-410m \
  --dataset_name ./wikitext-2-raw-v1 \
  --dataset_config wikitext-2-raw-v1 \
  --output_dir ./outputs_pythia_peft \
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
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--output_dir", type=str, default="./outputs_peft")
    p.add_argument("--deepspeed", type=str, default="./ds_config.json")
    
    # LoRA 参数
    p.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension (rank)")  # 控制LoRA适配器的参数量和表达能力
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter for scaling")  # 控制LoRA适配器而输出强度, 平衡原模型能力和适配器的影响
    p.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability for LoRA layers")
    p.add_argument("--target_modules", type=str, default="query_key_value,dense", 
                   help="Comma-separated list of target modules to apply LoRA")
    p.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"],
                   help="Bias training type")  # 控制是否训练b, 权重矩阵的参数量远远大于b的参数量
    
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
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    # 配置 LoRA
    target_modules = args.target_modules.split(",") if args.target_modules else None
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用 PEFT/LoRA 到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数信息
    
    # 加载数据集
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
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 训练参数
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
        gradient_checkpointing=False,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None) if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 训练
    trainer.train()
    
    # 保存模型 (只保存 LoRA 权重)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"训练完成！模型已保存到: {args.output_dir}")
    print(f"可训练参数比例: {model.print_trainable_parameters()}")

if __name__ == "__main__":
    main()
