from transformers import AutoModelForCausalLM, AutoTokenizer

# 大模型文件夹路径
MODEL_DIR = "./Llama3-Chinese-8B-Instruct"

messages = [{"role": "system", "content": ""}, {"role": "user", "content": "你好，请介绍下自己。"}]

# 加载因果语言模型
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto")

# 加载与模型相匹配的分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in generated_ids]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(response)
