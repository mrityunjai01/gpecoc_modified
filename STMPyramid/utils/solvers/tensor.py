import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from random import seed
np.random.seed(1)
seed(1)

def rank_R_decomp(X, rank = 3):
    X_t = tl.tensor(X)
    _, factors = parafac(X_t, int(rank))
    fact_np = [tl.to_numpy(f) for f in factors]
    return fact_np


def inner_prod_decomp(Ai, Aj):
    s = 0.0
    R = Ai[0].shape[1]
    for p in range(R):
        for q in range(R):
            prod = 1.0
            for ai, aj in zip(Ai, Aj):  
                prod *= np.dot(ai[:, p], aj[:, q])
            s += prod
    return s

def make_kernel(data_decomp):
    K = np.zeros((len(data_decomp), len(data_decomp)))
    for i in range(len(data_decomp)):
        for j in range(i+1):
            K[i, j] = inner_prod_decomp(data_decomp[i], data_decomp[j])
            K[j, i] = K[i, j]
    return K

def construct_W_from_mat(data_decomp, l, eps=1e-100):
    R = data_decomp[0][1].shape[1]
    W = tl.zeros([data_decomp[0][i].shape[0] for i in range(len(data_decomp[0]))])
    for i, flag in enumerate((np.abs(l) > eps)):
        if flag:
            W += l[i]*tl.cp_to_tensor((np.ones(R), data_decomp[i]))
    return tl.to_numpy(W)