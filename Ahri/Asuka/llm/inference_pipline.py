import torch
import transformers

# 大模型文件夹路径
MODEL_DIR = "./Llama3-Chinese-8B-Instruct"
messages = [{"role": "system", "content": ""}]

pipeline = transformers.pipeline(
    "text-generation", model=MODEL_DIR, model_kwargs={"torch_dtype": torch.float16}, device="cuda"
)

while True:
    text = input("您好，有什么问题吗？请输入\n")
    messages.append({"role": "user", "content": text})

    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    outputs = pipeline(prompt, max_new_tokens=512, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)

    content = outputs[0]["generated_text"][len(prompt) :]

    print(content)
