from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
inputs = tokenizer(["Nine judges currently serve the Supreme"], return_tensors="pt")

# simple forward
outputs = model(**inputs)
logits = outputs.logits
input_ids = inputs["input_ids"].tolist()[0]
next_word = logits[0][-1].argmax()

t, p = tokenizer.decode(input_ids), tokenizer.decode(next_word)
print(f'prefix: {t} \nnext word: {p}')
