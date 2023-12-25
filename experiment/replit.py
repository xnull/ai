import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained(
    "replit/replit-code-v1_5-3b",
    trust_remote_code=True
)
config.attn_config['attn_impl'] = 'triton'

# load model
tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1_5-3b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1_5-3b', config=config, trust_remote_code=True)
model.to(device='mps:0', dtype=torch.bfloat16)

# forward pass
x = tokenizer.encode('def fibonacci(n): ', return_tensors='pt').to(device='mps:0')
x = x.to(device='mps:0')
y = model.generate(x, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)


# decoding
generated_code = tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(generated_code)
