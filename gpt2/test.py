from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

import warnings
warnings.filterwarnings("ignore")

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2") #.to(torch_device)

text = 'hi how are you doing?'
encoded_input = tokenizer(text, return_tensors='pt') #.to(torch_device)

sample_outputs = model.generate(
    **encoded_input,
    do_sample=True,
    top_k=10,
    top_p=0.95,
    num_return_sequences=3,
)

print("\nOutput:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
