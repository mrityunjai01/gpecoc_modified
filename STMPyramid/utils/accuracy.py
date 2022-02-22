import numpy as np
from random import seed
np.random.seed(1)
seed(1)

def accuracy(a,b):
    return np.mean(np.array(a)==np.array(b))