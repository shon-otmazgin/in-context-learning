import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_attention(value, key, query):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    # attn_weights = torch.where(mask, attn_weights, mask_value)

    return torch.matmul(attn_weights, value)


def transformers_sgd_dual_form(value, key, query):
    (seq_length, features) = key.shape
    W_delta = torch.zeros((features, features))

    for i in range(seq_length):
        e_i = value[i, :]
        x_i_tag = key[i, :]

        W_delta[:, :] += torch.outer(e_i, x_i_tag)

    result = torch.matmul(W_delta, query.T).T

    return result, W_delta


def sgd_dual_form(E, X, x):
    # E   = (out_dim, samples)
    # X   = (in_dim, samples)
    # x   = (in_dim, 1)

    in_dim, samples = X.shape
    W_delta = torch.zeros((1, in_dim))
    for i in range(samples):
       e_i = E[:, i]
       x_i = X[:, i]
       W_delta += torch.outer(e_i, x_i)

    result = torch.matmul(W_delta, x)

    return result, W_delta


if __name__ == '__main__':
    set_seed(42)

    # hidden_dim = 768
    # num_samples = 8
    # out_dim = 1
    #
    # # gradient for each sample w.r.t to output
    # E = torch.rand((out_dim, num_samples))
    # X_tag = torch.rand((hidden_dim, num_samples))
    # x = torch.rand((hidden_dim, 1))
    #
    # attn_outputs = linear_attention(value=E, key=X_tag, query=x)
    # print(f'results from linear_attention: \n{attn_outputs}')
    # result, W_delta = sgd_dual_form(E, X_tag, x)
    # print(f'results from dual form: \n{result}')
    # print(f'W_delta: \n{W_delta.shape}')
    # print()

    # # (batch, head, seq_length, head_features)
    query = torch.from_numpy(np.load('query.npy'))[0].reshape(6, 768)[-1, :].unsqueeze(0)
    key = torch.from_numpy(np.load('key.npy'))[0].reshape(6, 768)
    value = torch.from_numpy(np.load('value.npy'))[0].reshape(6, 768)
    mask_value = torch.from_numpy(np.load('mask_value.npy'))
    causal_mask = torch.from_numpy(np.load('causal_mask.npy'))[0]
    attn_weights = torch.from_numpy(np.load('attn_weights.npy'))[0]
    # attn_output = torch.matmul(attn_weights.T, value).view(1, -1)

    print(query.shape)
    print(key.shape)
    print(value.shape)
    # print(mask_value, mask_value.shape)
    # print(causal_mask.shape)
    # print(attn_weights.shape)
    # print(attn_output.shape)

    linear_attn_outputs = linear_attention(value, key, query)
    print(f'results from linear_attention: \n{linear_attn_outputs.shape}')
    result, W_delta = transformers_sgd_dual_form(value, key, query)
    print(f'results from dual_form: \n{result.shape}')

    print(torch.equal(linear_attn_outputs, result))
    # print(torch.equal(linear_attn_outputs[:, -1, :].reshape(1, -1), result[:, -1, :].reshape(1, -1)))





