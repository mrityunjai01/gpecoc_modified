import numpy as np
from random import seed
np.random.seed(1)
seed(1)

def visualise(w):
    wsq=w.copy()
    wsq = (wsq-np.min(wsq))/(np.max(wsq)-np.min(wsq))
    return wsq


def visualise_pos(w):
    wsq=w.copy()
    wsq=np.maximum(wsq,0)
    wsq = (wsq-np.min(wsq))/(np.max(wsq)-np.min(wsq))
    return wsq

def visualise_neg(w):
    wsq=w.copy()
    wsq=np.minimum(wsq, 0)
    wsq = (wsq-np.min(wsq))/(np.max(wsq)-np.min(wsq))

    return wsq
