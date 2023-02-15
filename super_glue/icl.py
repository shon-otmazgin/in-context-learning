import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, EvalPrediction

from data import get_datasets
from config import prompts, seperator
from utils import Project2TargetTokens


def prepare_dataset_for_inference(few_shot_dataset, dev_dataset, tokenizer):
    few_shot_prompt = f'{seperator}'.join(few_shot_dataset['prompt'])

    for example in tqdm(dev_dataset):
        prompt = few_shot_prompt + f'{seperator}{example["prompt"]}'
        encoded_prompts = tokenizer(prompt, return_tensors='pt')
        yield encoded_prompts


if __name__ == '__main__':
    task_name = 'rte'

    opt_size = 'facebook/opt-13b'
    cache_dir = '/home/nlp/shon711/.cache'
    experiment_name = f'ICL-{task_name}-{opt_size}'

    wandb.init(project='in-context-learning', name=experiment_name)

    prompter = prompts[task_name]
    tokenizer = AutoTokenizer.from_pretrained(opt_size, use_fast=False, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        opt_size, device_map="auto", cache_dir=cache_dir,
        max_memory={0: "25GiB", 1: "25GiB", "cpu": "300GiB"}
    )
    model.eval()
    print(model.hf_device_map)

    few_shot_dataset, dev_dataset = get_datasets(task_name=task_name, prompter=prompter, n_shots=16, seed=0)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    target_tokens = list(prompter.target_tokens.values())
    encoded_target_tokens = [x[0] for x in tokenizer(target_tokens, add_special_tokens=False)['input_ids']]
    target_tokens_logits_processor = Project2TargetTokens(encoded_target_tokens)

    all_preds, all_labels = [], []
    for i, encoded_prompt in enumerate(prepare_dataset_for_inference(few_shot_dataset, dev_dataset, tokenizer)):
        encoded_prompt = encoded_prompt.to(0)
        with torch.no_grad():
            output = model(**encoded_prompt, output_hidden_states=True)

        hidden_states = output['hidden_states']
        hidden_states = hidden_states[-1][:, -2:, :].cpu().numpy()
        np.save(f'icl_outputs/{task_name}/{i}_hidden_states.npy', hidden_states)

        logits, labels = output['logits'], encoded_prompt['input_ids']
        target_tokens_logits = target_tokens_logits_processor(logits, labels)

        target_tokens_logits = target_tokens_logits.cpu().numpy()
        labels = labels.cpu().numpy()

        labels, preds = target_tokens_logits_processor.eval(
            EvalPrediction(predictions=target_tokens_logits, label_ids=labels), return_accuracy=False
        )
        all_preds += preds
        all_labels += labels

    acc = accuracy_score(y_true=all_labels, y_pred=all_preds)
    wandb.log({'accuracy': acc})
    print(acc)
