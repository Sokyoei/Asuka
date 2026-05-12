"""
Fine Tune 大模型微调
"""

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, TrainingArguments
from transformers.trainer_utils import SaveStrategy
from transformers.training_args import OptimizerNames
from trl import SFTTrainer

from Ahri.Asuka.config.config import settings
from Ahri.Asuka.utils import DEVICE

MODEL_DIR = settings.MODELS_DIR / "your_local_model_dir"
OUTPUT_DIR = settings.MODELS_DIR / "your_local_output_dir"
DATA_PATH = settings.DATA_DIR / "your_local_data_path"


# 加载模型和分词器
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map=DEVICE)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # LoRA 的秩 r
    lora_alpha=32,  # 缩放系数
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 微调的层
    bias="none",
)

# 数据集配置
dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # 批次大小
    gradient_accumulation_steps=1,
    learning_rate=2e-4,  # 学习率
    num_train_epochs=20,  # 训练次数
    weight_decay=0.01,
    warmup_steps=0.1,
    save_strategy=SaveStrategy.EPOCH,
    optim=OptimizerNames.ADAMW_TORCH,
    # fp16=DEVICE == "cuda",
    bf16=True,
    use_cpu=DEVICE.type == "cpu",
    report_to="none",  # 关闭 wandb 日志
)


def formatting_func(example):
    # 不同模型的对话模板不一定相同
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ],
        tokenize=False,
    )


# 启用微调
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
    formatting_func=formatting_func,
    processing_class=tokenizer,
    # dataset_text_field="text",
)

print("开始微调")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("微调完成")
