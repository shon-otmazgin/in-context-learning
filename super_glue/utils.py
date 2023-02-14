import torch
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback


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

    def eval(self, eval_perds, return_accuracy=True):
        labels = torch.from_numpy(eval_perds.label_ids)
        logits = eval_perds.predictions

        # -1 is the label
        last_token_indices = (labels != -100).sum(dim=-1) - 1
        true_target_tokens = torch.gather(labels, dim=1, index=last_token_indices.unsqueeze(-1)).squeeze(-1).tolist()
        pred_target_tokens = [self.target_tokens[x] for x in logits.argmax(axis=-1).tolist()]

        if return_accuracy:
            return {
                'accuracy': accuracy_score(y_true=true_target_tokens, y_pred=pred_target_tokens)
            }
        else:
            return true_target_tokens, pred_target_tokens


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True
