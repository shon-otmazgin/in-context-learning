from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
inputs = tokenizer(["Nine judges currently serve the Supreme"], return_tensors="pt")

# simple forward
outputs = model(**inputs)
logits = outputs.logits
input_ids = inputs["input_ids"].tolist()[0]
ids = logits.argmax(dim=-1).tolist()[0]

# simple forward

texts = []
for i, _id in enumerate(ids):
    t, p = tokenizer.decode(input_ids[:i+1]), tokenizer.decode(_id)
    texts.append(t)
    print(f'prefix: {t} \nnext word: {p}')
    print()

# using generate

for t in texts:
    inputs = tokenizer([t], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    print(f'prefix: {t}')
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        # | token | token string | logits | probability
        print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
    print()