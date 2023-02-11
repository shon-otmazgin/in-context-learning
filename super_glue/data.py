from datasets import load_dataset


def prepare_prompt(ex, prompt_func, add_completion=True):
    prompt, completion = prompt_func(ex, add_completion)

    ex['prompt'] = prompt
    ex['completion'] = completion

    return ex


def get_datasets(task_name, prompt_func, n_shots=32, seed=None, shots_indices=None):
    dataset = load_dataset("super_glue", task_name)

    if not shots_indices:
        few_shot_dataset = dataset['train'].train_test_split(
            test_size=n_shots, shuffle=True, seed=seed if seed else n_shots
        )['test']
    else:
        few_shot_dataset = dataset['train'].select(shots_indices)
    dev_dataset = dataset['validation']

    few_shot_dataset = few_shot_dataset.map(
        prepare_prompt, batched=False, fn_kwargs={'prompt_func': prompt_func, 'add_completion': True}
    )
    dev_dataset = dev_dataset.map(
        prepare_prompt, batched=False, fn_kwargs={'prompt_func': prompt_func, 'add_completion': False}
    )

    return few_shot_dataset, dev_dataset


if __name__ == '__main__':
    def rte_prompt(ex, add_completion):
        prompt_template = '{premise} question: {hypothesis} Yes or No? answer:{completion}'
        label_to_completion_map = {
            1: ' No',       # not_entailment
            0: ' Yes'       # entailment
        }

        completion = label_to_completion_map[ex['label']]
        prompt = prompt_template.format(
            premise=ex['premise'], hypothesis=ex['hypothesis'], completion=completion if add_completion else ''
        ).strip()

        return prompt, completion

    indices = [1432, 1711, 383, 1742, 31, 2304, 391, 380, 1607, 703, 1814, 2082, 2379, 1189, 1573, 1455]
    few_shot_dataset, dev_dataset = get_datasets(
        task_name='rte', prompt_func=rte_prompt, shots_indices=indices
    )

    seperator = '\n\n'
    few_shot_prompt = f'{seperator}'.join(few_shot_dataset['prompt'])

    with open('prompt.txt', 'r') as f:
        other_prompt = f.read()
    # print(few_shot_prompt)
    # print(other_prompt)


    print(few_shot_dataset)
    print(dev_dataset)
    print(few_shot_dataset[0]['prompt'])
    print(few_shot_dataset[0]['completion'])
    print()
    print(dev_dataset[0]['prompt'])
    print(dev_dataset[0]['completion'])

