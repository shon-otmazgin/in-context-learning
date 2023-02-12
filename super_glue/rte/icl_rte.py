import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.auto import trange, tqdm
from sklearn.metrics import accuracy_score

from super_glue.data import get_datasets


def rte_prompt(ex, add_completion):
    prompt_template = '{premise} question: {hypothesis} Yes or No? answer:{completion}'
    label_to_completion_map = {
        1: ' No',  # not_entailment
        0: ' Yes'  # entailment
    }

    completion = label_to_completion_map[ex['class_label']]
    prompt = prompt_template.format(
        premise=ex['premise'], hypothesis=ex['hypothesis'], completion=completion if add_completion else ''
    ).strip()

    return prompt, completion


if __name__ == '__main__':
    # indices = [1432, 1711, 383, 1742, 31, 2304, 391, 380, 1607, 703, 1814, 2082, 2379, 1189, 1573, 1455]
    few_shot_dataset, dev_dataset = get_datasets(
        task_name='rte', prompt_func=rte_prompt, n_shots=16, seed=0
    )

    seperator = '\n\n'
    few_shot_prompt = f'{seperator}'.join(few_shot_dataset['prompt'])

    preds = []
    labels = []
    prompts = []
    for example in dev_dataset:
        prompt = few_shot_prompt + f'{seperator}{example["prompt"]}'
        prompts.append(prompt)
        labels.append(example['class_label'])

    print(prompts[0])
    print(labels[0])
    print()


    # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    opt_size = 'facebook/opt-13b'
    model = AutoModelForCausalLM.from_pretrained(opt_size, device_map="auto")#.to(device)
    tokenizer = AutoTokenizer.from_pretrained(opt_size, use_fast=False)

    print(model.hf_device_map)

    yes_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode(" No", add_special_tokens=False)[0]
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
        classes_logits = last_hidden_state_logits[:, :, [yes_id, no_id]].squeeze(1)
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



