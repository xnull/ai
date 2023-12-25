# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import time

device='cpu'

print("INIT TOKENIZER")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", device_map=device)

print("INIT MODEL")
#low_cpu_mem_usage=True
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map=device)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
print("GENERATE tokens")
start = time.time()
generate_ids = model.generate(inputs.input_ids, max_length=30)
end = time.time()
print("Time: " +  str(end - start))

print("DECODE tokens")
res = tokenizer.decode(
    generate_ids, 
    skip_special_tokens=True, 
    clean_up_tokenization_spaces=False)

print("\nOutput:")
print(res)