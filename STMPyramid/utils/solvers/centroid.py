import numpy as np
from random import seed
np.random.seed(1)
seed(1)
def centroid(points):
    c = np.sum(points, axis=0)
    return c/points.shape[0]
