class RTEPrompt:
    def __init__(self):
        self.prompt_template = '{premise} question: {hypothesis} Yes or No? answer:{completion}'
        self.target_tokens = {
            1: ' No',  # not_entailment
            0: ' Yes'  # entailment
        }

    def __call__(self, example, add_completion):
        completion = self.target_tokens[example['class_label']]
        prompt = self.prompt_template.format(
            premise=example['premise'], hypothesis=example['hypothesis'],
            completion=completion if add_completion else ''
        ).strip()

        return prompt, completion


seperator = '\n\n'
prompts = {
    'rte': RTEPrompt(),
}


