"""
deepspeed --num_gpus=1 train.py \
  --model_name_or_path gpt2 \
  --dataset_name wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --output_dir ./outputs \
  --deepspeed ./ds_config.json \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, default="gpt2")
    p.add_argument("--dataset_name", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--deepspeed", type=str, default="./ds_config.json")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--do_eval", action="store_true")
    return p.parse_args()

def main():

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)  # 会帮我们判断: 模型为args.model_name_or_path, 下载对应的tokenizer. Tokenizer的作用是将文本变为模型能够理解的token id. tokenizer必须和模型100%配套. 注意, 这里开启了use_fast, 这是让tokenzier使用Rust实现的高速版本, 而不是Python版本. 注意, use_fast并不总是可用的, 如果某个模型没有对应的fast tokenizer, 那么就会自动回退到普通的tokenizer. 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    """
    什么是padding?

    假设batch里面有两句话:
    "I love deep learning"
    "Hello"

    模型要求一个batch内的tensor必须等长, 所以要进行补齐:
    "I love deep learning"
    "Hello <PAD> <PAD> <PAD>"

    否则:
    ❌ tensor 无法堆叠
    ❌ GPU 无法并行计算

    很多decoder-only的模型没有PAD的概念, 他们只有<BOS>和<EOS>, 由于<EOS>本来就代表后面没内容了, padding的语义其实也是, 后面没有有效token, 所以在语义上是兼容的, 可以等效替换.
    """
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)  # 干了很多事情, 下载模型, 加载config, 构建网络结构, 填充权重, 切换到eval状态, 本质是一键恢复完整模型
    raw = load_dataset(args.dataset_name, args.dataset_config)  # dataset的数据存在磁盘, 但是看起来像在内存, 后面的args.dataset_config可用于指定数据集版本. 它没有使用传统的f.read(), 而是使用了内存映射的方式, 只有在真正访问数据的时候才会从磁盘读取, 这也是为什么它能处理大规模数据的原因. 

    def tokenize_fn(examples):
        texts = examples[args.text_field]
        return tokenizer(
            texts,  # 一次性tokenize
            truncation=True,  # 超过max_length截断
            max_length=args.max_length,
            padding=False,  # 这段代码是在训练之前, 把原始文本数据集一次性转换为input_ids这种token序列, 并把结果存到磁盘缓存里面, 如果在这个阶段就填充到max_length, 训练的时候DataLoader每次读的时候会耗费大量的计算资源去读取这些padding. 所以这里不填充, 让它保持原始长度, 训练的时候再动态填充
        )

    tokenized = raw.map(  # 这是datasets的核心api
        tokenize_fn,
        batched=True,  # 注意, 这里的batch_size和训练时的batch_size没有关系, 这里的batch_size是指map函数在处理数据集的时候, 一次性传递给tokenize_fn函数多少条数据. 这个值越大, tokenize_fn一次性处理的数据就越多, 效率就越高, 但是同时也会占用更多的内存. 这个值的选择需要根据你的数据集大小和内存情况来调整.
        remove_columns=raw["train"].column_names,  # 只保留tokenize_fn返回的列, 将原始列比如说text, title等都删除掉, 这样可以节省磁盘空间和内存
        desc="Tokenizing",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    """
    负责三件事情:

    1. padding. 刚才没有padding, 它会在每个batch里面, 找到最长的样本, 把其他的样本pad到最长样本的长度; 2. 生成labels. 对于causal language modeling, labels就是input_ids的一个shifted版本, 也就是说每个位置的label是下一个位置的token id, 这个shift的逻辑是在模型内部完成的, 所以这里只需要把labels设置为input_ids的一个副本就行了, 它会帮我们完成; 3. 屏蔽padding token的loss, 也就是说在计算loss的时候, padding token的预测结果不参与loss的计算, 这也是通过把labels里面padding token的位置设置为-100来实现的, 因为在transformers的loss计算中, label为-100的位置会被自动忽略掉.

    mlm的意思是masked language modeling, 这是BERT等encoder-only模型使用的预训练任务, 但是GPT等decoder-only模型使用的是causal language modeling, 所以这里设置mlm=False.
    """

    train_args = TrainingArguments(  # 这是Trainer的训练配置对象, 用来集中描述如何训, 训练多久, 用多少资源, 怎么存, 怎么打日志等等
        output_dir=args.output_dir,  # 所有checkpoint/日志的根目录
        overwrite_output_dir=True,  # 如果目录已经存在, 则不报错, 不中断, 直接覆盖

        # batch & optimization
        per_device_train_batch_size=args.per_device_train_batch_size,  # 每张GPU上的batch_size, 不是全局batch_size, 全局batch_size = per_device_train_batch_size * num_gpus * gradient_accumulation_steps
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,  # 优化器的基础lr, 会传给optimizer
        num_train_epochs=args.num_train_epochs,  # 训练多少轮
        warmup_ratio=0.03,  # 前3%用于warmup预热
        weight_decay=0.0,

        # logging / saving
        logging_steps=args.logging_steps,  # 每多少次step打一次log
        save_steps=args.save_steps,  # 每save_steps保存一次模型checkpoint
        save_total_limit=2,  # 最多保留两个

        # eval
        evaluation_strategy="steps" if args.do_eval else "no",  # 评估的策略, 每多少步评估一次, 这里设置为每eval_steps评估一次, 如果不评估则设置为no
        eval_steps=args.eval_steps if args.do_eval else None,  # 每多少步评估一次, 只有当evaluation_strategy不为no时才有效

        # precision
        fp16=True,  # 也可以关掉，用 deepspeed 控制
        bf16=False,

        # deepspeed
        deepspeed=args.deepspeed,  # args.deepspeed是一个json路径, Trainer会自动初始化DeepSpeed, 接管optimizer, ZeRO, 通信. 

        # misc
        report_to="none",  # 不用tensorboard, wandb, 纯命令行跑
        dataloader_num_workers=2,  # DataLoader子进程数, 用来加速读取
        gradient_checkpointing=False,  # 正常训练, forward保存每一层的activation, backward直接用保存的activation算梯度; 缺点是显存占用大, 优点是快; 开启gradient_checkpointing后, forward只保存关键节点的activation, backward需要重新计算缺失的activation来算梯度; 优点是显存占用小, 缺点是慢, 适合显存受限的情况
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation", None) if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
