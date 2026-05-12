import torch
from peft import PeftModel
from transformers import TextGenerationPipeline, pipeline

from Ahri.Asuka.config.config import settings
from Ahri.Asuka.utils import DEVICE

BASE_MODEL = settings.MODELS_DIR / "your_local_model_dir"
LORA_MODEL = settings.MODELS_DIR / "your_local_lora_model_dir"

pipe: TextGenerationPipeline = pipeline("text-generation", model=BASE_MODEL, dtype=torch.bfloat16, device=DEVICE)
pipe.model = PeftModel.from_pretrained(pipe.model, LORA_MODEL)


prompt = "your_prompt"
messages = [{"role": "user", "content": prompt}]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe(
    prompt,
    max_new_tokens=32,
    do_sample=False,
    pad_token_id=pipe.tokenizer.eos_token_id,
    eos_token_id=pipe.tokenizer.eos_token_id,
    repetition_penalty=1.1,
)

result = outputs[0]["generated_text"][len(prompt) :].strip()
print(result)
