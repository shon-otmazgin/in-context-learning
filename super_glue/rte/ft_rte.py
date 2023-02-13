import wandb
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, TrainerCallback

from data import get_datasets
from icl_rte import rte_prompt, rte_config
from peft import get_peft_model, LoraConfig


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


class Project2TargetTokens:
    def __init__(self, target_tokens):
        self.target_tokens = target_tokens

    def __call__(self, logits, labels):
        batch_size, seq_len, vocab_size = logits.size()

        # -1 is the label, -2 is the last token before the label
        last_token_indices = (labels != -100).sum(dim=-1) - 2

        last_token_indices = last_token_indices.unsqueeze(-1).unsqueeze(-1).expand(
            (batch_size, 1, vocab_size))
        # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        last_token_logits = torch.gather(logits, dim=1, index=last_token_indices)
        logits_target_tokens = last_token_logits[:, :, self.target_tokens].squeeze(1)

        return logits_target_tokens

    def eval_rte(self, eval_perds):
        labels = torch.from_numpy(eval_perds.label_ids)
        logits = eval_perds.predictions

        # -1 is the label
        last_token_indices = (labels != -100).sum(dim=-1) - 1
        true_target_tokens = torch.gather(labels, dim=1, index=last_token_indices.unsqueeze(-1)).squeeze(-1).tolist()
        pred_target_tokens = [self.target_tokens[x] for x in logits.argmax(axis=-1).tolist()]
        return {
            'accuracy': accuracy_score(y_true=true_target_tokens, y_pred=pred_target_tokens)
        }


if __name__ == '__main__':
    cache_dir = '/home/nlp/shon711/.cache'
    opt_size = 'facebook/opt-30b'
    experiment_name = f'rte-{opt_size}'

    wandb.init(project='in-context-learning', name=experiment_name)

    tokenizer = AutoTokenizer.from_pretrained(opt_size, use_fast=False, cache_dir=cache_dir)

    few_shot_dataset, dev_dataset = get_datasets(
        task_name='rte', prompt_func=rte_prompt, n_shots=16, seed=0, finetune=True, tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=experiment_name,
        logging_dir=experiment_name,
        overwrite_output_dir=True,
        learning_rate=3e-4,
        max_steps=len(few_shot_dataset),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_strategy='no',
        evaluation_strategy='no',
        # eval_steps=1,
        # logging_first_step=True,
        logging_strategy='steps',
        logging_steps=4,
        remove_unused_columns=True,
        seed=0,
        fp16=True,
        report_to='wandb',
        run_name=experiment_name,
        deepspeed='ds_config.json'
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    target_tokens = [x[0] for x in tokenizer(rte_config['target_tokens'], add_special_tokens=False)['input_ids']]
    logits_processor = Project2TargetTokens(target_tokens)

    model = AutoModelForCausalLM.from_pretrained(opt_size, cache_dir=cache_dir)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=few_shot_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=logits_processor.eval_rte,
        preprocess_logits_for_metrics=logits_processor
    )
    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)


