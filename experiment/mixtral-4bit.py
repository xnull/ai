from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(model_id)

