from datasets import load_dataset


def prepare_prompt(ex, prompt_template, label_to_completion_map, add_completion=True):
    completion = label_to_completion_map[ex['label']]
    prompt = prompt_template.format(
        premise=ex['premise'], hypothesis=ex['hypothesis'], completion=completion if add_completion else ''
    ).strip()
    ex['prompt'] = prompt
    ex['completion'] = completion

    return ex


def get_datasets(task_name, prompt_template, label_to_completion_map=None, n_shots=32):
    dataset = load_dataset("super_glue", task_name)

    few_shot_dataset = dataset['train'].train_test_split(test_size=n_shots, shuffle=True, seed=n_shots)['test']
    dev_dataset = dataset['validation']

    few_shot_dataset = few_shot_dataset.map(
        prepare_prompt, batched=False, fn_kwargs={'prompt_template': prompt_template,
                                                  'label_to_completion_map': label_to_completion_map,
                                                  'add_completion': True}
    )
    dev_dataset = dev_dataset.map(
        prepare_prompt, batched=False, fn_kwargs={'prompt_template': prompt_template,
                                                  'label_to_completion_map': label_to_completion_map,
                                                  'add_completion': False}
    )

    return few_shot_dataset, dev_dataset


if __name__ == '__main__':
    prompt_template = '{premise}\nQuestion: {hypothesis} True or False?\nAnswer:{completion}'
    label_to_completion_map = {
        1: ' False',     # not_entailment
        0: ' True'       # entailment
    }

    few_shot_dataset, dev_dataset = get_datasets(
        task_name='rte', prompt_template=prompt_template,
        label_to_completion_map=label_to_completion_map, n_shots=22
    )

    print(few_shot_dataset)
    print(dev_dataset)
    print(few_shot_dataset[0]['prompt'])
    print(few_shot_dataset[0]['completion'])
    print()
    print(dev_dataset[0]['prompt'])
    print(dev_dataset[0]['completion'])

