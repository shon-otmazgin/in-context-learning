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


if __name__ == '__main__':
    set_seed(42)

    # each input vector is is size of 768
    hidden_dim = 768
    num_samples = 8

    X_prev = torch.rand((hidden_dim, num_samples))
    x_n = torch.rand((hidden_dim, 1))
    E = torch.rand((hidden_dim, num_samples))

    attn_logits = linear_attention(V=E, K=X_prev, q=x_n)
    print(attn_logits)




