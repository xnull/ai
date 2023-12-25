import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import time

torch.set_default_device("mps")

#model_path='bert-base-cased'
model_path="microsoft/phi-1_5"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True, device_map='mps')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device_map='mps')

print("create tokens")
inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

start = time.time()
print("GENERATE tokens")
outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
end = time.time()
print(text)
print("Time: " +  str(end - start))


start = time.time()
print("GENERATE tokens 2")
inputs2 = tokenizer('hi how are you doing?', return_tensors="pt", return_attention_mask=False)
outputs2 = model.generate(**inputs2, max_length=200)
text2 = tokenizer.batch_decode(outputs2)[0]
end = time.time()
print(text2)
print("Time: " +  str(end - start))
