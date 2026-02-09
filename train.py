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
    raw = load_dataset(args.dataset_name, args.dataset_config)  # dataset的数据存在磁盘, 但是看起来像在内存, 后面的args.dataset_config可用于指定数据集版本
