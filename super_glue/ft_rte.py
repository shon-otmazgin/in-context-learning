import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

from data import get_datasets
from config import prompts
from utils import MyTrainer, Project2TargetTokens, EvaluateFirstStepCallback

from peft import get_peft_model, LoraConfig


if __name__ == '__main__':
    task_name = 'rte'
    n_shots = 16
    seed = 0

    opt_size = 'facebook/opt-13b'
    cache_dir = '/home/nlp/shon711/.cache'
    experiment_name = f'FT-{task_name}-{opt_size}'

    wandb.init(project='in-context-learning', name=experiment_name)

    training_args = TrainingArguments(
        output_dir=experiment_name,
        logging_dir=experiment_name,
        overwrite_output_dir=True,
        learning_rate=3e-4,
        gradient_accumulation_steps=2,
        max_steps=64,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        evaluation_strategy='steps',
        eval_steps=n_shots,
        logging_strategy='steps',
        logging_steps=4,
        seed=seed,
        fp16=True,
        report_to=['wandb'],
        run_name=experiment_name,
        save_strategy='steps',
        save_steps=n_shots,
        save_total_limit=1,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        deepspeed='ds_zero2_config.json'
    )

    prompter = prompts[task_name]
    tokenizer = AutoTokenizer.from_pretrained(opt_size, use_fast=False, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(opt_size, cache_dir=cache_dir)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    few_shot_dataset, dev_dataset = get_datasets(
        task_name=task_name, prompter=prompter, n_shots=n_shots, seed=seed, tokenizer=tokenizer
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    target_tokens = list(prompter.target_tokens.values())
    encoded_target_tokens = [x[0] for x in tokenizer(target_tokens, add_special_tokens=False)['input_ids']]
    target_tokens_logits_processor = Project2TargetTokens(encoded_target_tokens)

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=few_shot_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        preprocess_logits_for_metrics=target_tokens_logits_processor,
        compute_metrics=target_tokens_logits_processor.eval,
    )
    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

    # model.save_pretrained(experiment_name)


