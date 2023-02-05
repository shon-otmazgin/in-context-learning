from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
# inputs = tokenizer(["Nine judges currently serve the Supreme"], return_tensors="pt")
inputs = tokenizer(["seven of the nine sitting justices were"], return_tensors="pt")
# inputs = tokenizer(["Michael Jack, born 17 September"], return_tensors="pt")
# inputs = tokenizer(["Michael Jack was born in"], return_tensors="pt")

# simple forward
outputs = model(**inputs)
logits = outputs.logits
input_ids = inputs["input_ids"].tolist()

for i, ids in enumerate(input_ids):
    try:
        idx = ids.index(50256)
    except ValueError:
        idx = len(ids)

    next_word = logits[i][idx - 1].argmax()
    t, p = tokenizer.decode(ids[:idx]), tokenizer.decode(next_word)
    print(f'prefix: {t} \nnext word: {p}')
    print()
