import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from Ahri.Asuka.config.config import settings
from Ahri.Asuka.utils import DEVICE

BASE_MODEL = settings.MODELS_DIR / "your_local_model_dir"
LORA_MODEL = settings.MODELS_DIR / "your_local_lora_model_dir"

# 加载模型
tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL), dtype=torch.bfloat16, device_map=DEVICE)
model = PeftModel.from_pretrained(model, str(LORA_MODEL))

# 构造输入
prompt = "your_prompt"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(text)
inputs: BatchEncoding = tokenizer([text], return_tensors="pt").to(DEVICE)

# 推理
outputs = model.generate(
    **inputs,
    max_new_tokens=6,  # 最多生成 token 数
    repetition_penalty=1.1,  # 惩罚，防止复读
    # temperature=0.7,
    do_sample=False,  # 贪心搜索，不随机
    pad_token_id=tokenizer.eos_token_id,  # 填充标记，对齐长度
    eos_token_id=tokenizer.eos_token_id,  # 遇到结束符立刻停止
)
# print(outputs)

# 解析成文本
result = tokenizer.decode(outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)
print(result)
