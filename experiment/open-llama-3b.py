import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'

tokenizer = LlamaTokenizer.from_pretrained(model_path, low_cpu_mem_usage=True)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)

print("\nOutput:")
print(tokenizer.decode(generation_output[0]))
