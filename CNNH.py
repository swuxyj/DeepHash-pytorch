# CNNH(AAAI2014)
# paper [Supervised Hashing for Image Retrieval via Image Representation Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/download/8137/8861)

import torch
from itertools import product
from random import shuffle

n = 128
q = 64
label = (torch.rand((n, 21)) - 0.8).sign() + 1
S = (label @ label.t() > 0).float()
H = 2 * torch.rand((n, q)) - 1
L = H @ H.t() - q * S
T = 30
permutaion = list(product(range(n), range(q)))
for t in range(T):
    H_temp = H.clone()
    L_temp = L.clone()
    shuffle(permutaion)
    for i, j in permutaion:
        g_prime_Hij = 4 * L[i, :] @ H[:, j]
        g_prime_prime_Hij = 4 * (H[:, j].t() @ H[:, j] + H[i, j].pow(2) + L[i, i])
        d = (-g_prime_Hij / g_prime_prime_Hij).clamp(min=-1 - H[i, j], max=1 - H[i, j])

        L[i, :] = L[i, :] + d * H[:, j].t()
        L[:, i] = L[:, i] + d * H[:, j]
        L[i, i] = L[i, i] + d * d

        H[i, j] = H[i, j] + d
    if L.pow(2).mean() >= L_temp.pow(2).mean():
        H = H_temp
        L = L_temp
    else:
        print(t, L.pow(2).mean().item())