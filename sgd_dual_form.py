import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_attention(V, K, q):
    weights = torch.matmul(K.T, q)
    # is this is not linear attention here we need to do softmax

    return torch.matmul(V, weights)


def sgd_dual_form(W_0, E, X, x):
    # W_0 = (out_dim, in_dim)
    # E   = (out_dim, samples)
    # X   = (in_dim, samples)
    # x   = (in_dim, 1)

    in_dim, samples = X.shape
    W_delta = torch.zeros(W_0.shape)
    for i in range(samples):
       e_i = E[:, i]
       x_i = X[:, i]
       W_delta += torch.outer(e_i, x_i)

    result = torch.matmul((W_0 + W_delta), x)

    return result, W_delta


if __name__ == '__main__':
    set_seed(42)

    # each input vector is is size of 768
    hidden_dim = 768
    out_dim = 2
    num_samples = 8

    # init weights
    W_0 = torch.rand((out_dim, hidden_dim))

    # gradient for each sample w.r.t to output
    E = torch.rand((out_dim, num_samples))

    # inputs + new sample
    X_prev = torch.rand((hidden_dim, num_samples))
    x = torch.rand((hidden_dim, 1))

    attn_logits = linear_attention(V=E, K=X_prev, q=x)
    attn_result = torch.matmul(W_0, x) + attn_logits
    print(f'results from linear_attention: \n{attn_result}')

    print()

    result, W_delta = sgd_dual_form(W_0, E, X_prev, x)
    print(f'results from dual form: \n{result}')




