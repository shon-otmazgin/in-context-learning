import torch
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, EvalPrediction

from data import get_datasets
from config import prompts, seperator
from utils import Project2TargetTokens

from peft import PeftModel, PeftConfig

def prepare_dataset_for_inference(dev_dataset, tokenizer):
    for example in tqdm(dev_dataset):
        encoded_prompts = tokenizer(example["prompt"], return_tensors='pt')
        yield encoded_prompts


if __name__ == '__main__':
    task_name = 'rte'
    n_shots = 16
    seed = 0

    opt_size = 'facebook/opt-13b'
    cache_dir = '/home/nlp/shon711/.cache'
    experiment_name = f'FT-{task_name}-{opt_size}'
    adapter_path = f'{experiment_name}/checkpoint-32'

    wandb.init(project='in-context-learning', name=adapter_path)

    prompter = prompts[task_name]
    tokenizer = AutoTokenizer.from_pretrained(opt_size, use_fast=False, cache_dir=cache_dir)

    config = PeftConfig.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(opt_size, device_map="auto", cache_dir=cache_dir)
    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(model.hf_device_map)

    _, dev_dataset = get_datasets(task_name=task_name, prompter=prompter, n_shots=n_shots, seed=seed)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    target_tokens = list(prompter.target_tokens.values())
    encoded_target_tokens = [x[0] for x in tokenizer(target_tokens, add_special_tokens=False)['input_ids']]
    target_tokens_logits_processor = Project2TargetTokens(encoded_target_tokens)

    all_preds, all_labels = [], []
    for encoded_prompt in prepare_dataset_for_inference(dev_dataset, tokenizer):
        encoded_prompt = encoded_prompt.to(0)
        with torch.no_grad():
            output = model(**encoded_prompt)

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