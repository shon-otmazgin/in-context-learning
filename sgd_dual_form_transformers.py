import torch
from torch import nn
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

bias = torch.ones((1, 1, 2048, 2048), dtype=torch.int).tril()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def attention(query, key, value, linear=False):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # masking
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = bias[:, :, key_length - query_length: key_length, :key_length].to(torch.bool)
    mask_value = torch.finfo(attn_weights.dtype).min
    # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
    # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
    mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    if not linear:
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights


def linear_attn_to_sgd(query, key, value):
    (batch, head, seq_length, head_features) = key.size()

    # causal_mask = torch.ones((1, head, seq_length)).tril().bool()
    # causal_mask = causal_mask.unsqueeze(-1).expand((batch, head, seq_length, head_features))
    # mask_value = torch.finfo(key.dtype).min
    # mask_value = torch.tensor(mask_value, dtype=key.dtype).to(key.device)
    # key = torch.where(causal_mask, key, mask_value)

    delta_w = torch.zeros((batch, head, head_features, head_features), dtype=query.dtype)
    for b in range(batch):
        for h in range(head):
            for i in range(seq_length):
                e_i = value[b, h, i, :]
                x_tag_i = key[b, h, i, :]
                outer_product = torch.outer(e_i, x_tag_i)
                delta_w[b][h] += outer_product

    # causal_mask = torch.ones_like(delta_w).triu().bool()
    # mask_value = torch.finfo(delta_w.dtype).min
    # mask_value = torch.tensor(mask_value, dtype=delta_w.dtype).to(delta_w.device)
    # delta_w = torch.where(causal_mask, delta_w, mask_value)

    result = torch.matmul(delta_w, query.transpose(-2, -1)).transpose(-2, -1)

    return result, delta_w


if __name__ == '__main__':
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    inputs = tokenizer(["Nine judges currently serve the Supreme"], return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    # simple forward
    outputs = model(**inputs, output_attentions=True)
    logits = outputs.logits
    next_word = logits[0][-1].argmax()

    t, p = tokenizer.decode(input_ids), tokenizer.decode(next_word)
    print(f'prefix: {t} \nnext word: {p}')

    for i, (attn_weights, query, key, value) in enumerate(outputs['attentions']):
        attn_output = torch.matmul(attn_weights, value)
        attn_output2, attn_weights2 = attention(query, key, value, linear=True if i == 11 else False)
        attn_output3, delta_w = linear_attn_to_sgd(query, key, value)

        attn_output = attn_output
        attn_output2 = attn_output2
        print(f'Layer {i}: attn_weights  == using_torch: {torch.equal(attn_weights, attn_weights2)}')
        print(f'Layer {i}: attn_output   == using_torch: {torch.equal(attn_output, attn_output2)}')
        print(f'Layer {i}: attn_output     == dual_form: {torch.equal(attn_output, attn_output3)}')
        print(f'Layer {i}: attn_output     == dual_form: {torch.isclose(attn_output, attn_output3).sum() / torch.ones_like(attn_output3).sum()}')
        print(f'Layer {i}: attn_output     == dual_form: (last) {torch.isclose(attn_output[:, :, -1, :], attn_output3[:, :, -1, :]).sum() / torch.ones_like(attn_output3[:, :, -1, :]).sum()}')
        print()






