import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

from data import get_datasets
from config import rte_prompt, rte_config
from utils import Project2TargetTokens, EvaluateFirstStepCallback

from peft import get_peft_model, LoraConfig


if __name__ == '__main__':
    cache_dir = '/home/nlp/shon711/.cache'
    opt_size = 'facebook/opt-13b'
    experiment_name = f'rte-{opt_size}'

    wandb.init(project='in-context-learning', name=experiment_name)

    tokenizer = AutoTokenizer.from_pretrained(opt_size, use_fast=False, cache_dir=cache_dir)

    few_shot_dataset, dev_dataset = get_datasets(
        task_name='rte', prompter=rte_prompt, n_shots=16, seed=0, finetune=True, tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=experiment_name,
        logging_dir=experiment_name,
        overwrite_output_dir=True,
        learning_rate=3e-4,
        # gradient_accumulation_steps=len(few_shot_dataset),
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
        deepspeed='ds_zero2_config.json'
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
        compute_metrics=logits_processor.eval,
        preprocess_logits_for_metrics=logits_processor
    )
    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)


