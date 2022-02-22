from . import utils.solvers.solvers as solvers
from . import utils.loader as load
from .utils.crossval import grid_search_cv
from .constants import *
import numpy as np
from random import seed
np.random.seed(1)
seed(1)
if __name__ == "__main__":
    
    if dataset == 'mnist':
        Xtrain, ytrain = load.bin_mnist(class1, class2, totalsamp)
    if dataset == 'cifar10':
        Xtrain, ytrain = load.bin_CIFAR(class1, class2, totalsamp)
    if dataset == 'custom':
        Xtrain = np.load(Xtrain_file)
        ytrain = np.load(ytrain_file)
    
    if normalised == True:
        Xtrain, _ = load.normer(Xtrain)
    
    xa = None
    xb = None
    if len(Xtrain.shape) == 2:
        Xtrain = Xtrain.reshape((Xtrain.shape[0],Xtrain.shape[1],1))
    grid_search_cv(Xtrain,ytrain)






