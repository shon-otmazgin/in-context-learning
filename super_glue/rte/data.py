import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import trange, tqdm
from sklearn.metrics import accuracy_score


def prepare_for_prompt(file_path, prompt_template, label=True):
    df = pd.read_json(file_path, lines=True)

    df['str_label'] = df.apply(lambda ex: "True" if ex['label'] == "entailment" else "False", axis=1)

    df['prompt'] = df.apply(
        lambda ex: prompt_template.format(
            premise=ex['premise'], hypothesis=ex['hypothesis'], label=ex['str_label'] if label else ''
        ).strip()
        , axis=1
    )

    return df


prompt_template = '{premise}\nQuestion: {hypothesis} True or False?\nAnswer: {label}'
seperator = '\n\n'

train_df = prepare_for_prompt('super_glue/rte/train.jsonl', prompt_template)
dev_df = prepare_for_prompt('super_glue/rte/val.jsonl', prompt_template, label=False)

few_shot = 16
_, few_shot_df = train_test_split(train_df, test_size=few_shot, random_state=few_shot, shuffle=True)
few_shot_prompt = f'{seperator}'.join(few_shot_df['prompt'].tolist())

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
opt_size = 'facebook/opt-13b'
model = AutoModelForCausalLM.from_pretrained(opt_size, device_map="auto")#.to(device)
tokenizer = AutoTokenizer.from_pretrained(opt_size, use_fast=False)

print(model.hf_device_map)

preds = []
labels = []
prompts = []
for i, row in dev_df.iterrows():
    prompt = few_shot_prompt + f'{seperator}{row["prompt"]}'
    prompts.append(prompt)
    if row['label'] == "entailment":
        labels.append(1)
    elif row['label'] == "not_entailment":
        labels.append(0)
    else:
        raise Exception()

print(prompts[0])
print(labels[0])
print()

print(prompts[1])
print(labels[1])

true_id = tokenizer.encode(" True", add_special_tokens=False)[0]
false_id = tokenizer.encode(" False", add_special_tokens=False)[0]
batch_size = 1
batches = []
for i in trange(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    encoded_batch = tokenizer(batch, return_tensors="pt", padding=True)
    batches.append(encoded_batch)

model.eval()
for encoded_batch in tqdm(batches, desc='inference'):
    encoded_batch = encoded_batch.to(0)
    with torch.no_grad():
        outputs = model(**encoded_batch)

    logits = outputs['logits']
    batch_size, seq_len, vocab_size = logits.size()

    last_hidden_state_indices = (encoded_batch['input_ids'] != 1).sum(dim=-1) - 1

    last_hidden_state_indices = last_hidden_state_indices.unsqueeze(-1).unsqueeze(-1).expand((batch_size, 1, vocab_size))
    # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    last_hidden_state_logits = torch.gather(logits, dim=1, index=last_hidden_state_indices)
    classes_logits = last_hidden_state_logits[:, :, [false_id, true_id]].squeeze(1)
    preds += classes_logits.argmax(dim=-1).tolist()

acc = accuracy_score(labels, preds)
print(f'Accuracy: {acc}')

# mext_tok_ids = []
# for i, idx in enumerate(last_hidden_state_indices.tolist()):
#     tok_id = logits[i, idx].argmax()
#     mext_tok_ids.append(tok_id)
# for tok_id in mext_tok_ids:
#     if tok_id == true_id:
#         preds.append(1)
#     elif tok_id == false_id:
#         preds.append(0)
#     else:
#         raise Exception()



