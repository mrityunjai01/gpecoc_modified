import numpy as np
import cvxpy as cp
from random import seed
np.random.seed(1)
seed(1)

def construct_W_from_vec(W, shape):
    W = W.reshape(shape)
    return W
    

def inner_prod(A, B):
    if A.shape == B.shape:
        return np.sum(A*B)
    if A.shape == B.shape[1:]:
        ans = []
        for b in B:
            ans.append(np.sum(A*b))
        return ans
    
def inner_prod_cp(A, B):
    if A.shape == B.shape:
        return cp.sum(cp.multiply(A,B))
    if A.shape == B.shape[1:]:
        ans = []
        for b in B:
            ans.append(cp.sum(cp.multiply(A,B)))
        return ans