import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import mnist
from random import seed
np.random.seed(1)
seed(1)
def bin_mnist(class1 = 3, class2 = 8, totalsamp = None):
    XALL = mnist.train_images()[:totalsamp]/255
    yALL = mnist.train_labels().reshape(-1,1)[:totalsamp].astype(np.int8)
    yn = (yALL == class1)
    yp = (yALL == class2)
    ytot = (yn + yp).reshape(-1)
    Xtrain = XALL[ytot]
    ytrain = np.where(yALL[ytot] == class1, -1, 1).reshape(-1,1)
    return Xtrain, ytrain

def bin_CIFAR(class1 = 3, class2 = 8, totalsamp = None):
    transform = transforms.Compose([transforms.ToTensor(),])
    train_dataset = torchvision.datasets.CIFAR10(root = 'utils/data', train = True, download = True, transform=transform) #Training data set
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    XALL = next(iter(train_loader))[0].numpy().transpose((0,2,3,1))[:totalsamp]
    yALL = next(iter(train_loader))[1].numpy().reshape((-1,1))[:totalsamp]    
    yn = (yALL == class1)
    yp = (yALL == class2)
    ytot = (yn + yp).reshape(-1)
    Xtrain = XALL[ytot]
    ytrain = np.where(yALL[ytot] == class1, -1, 1).reshape(-1,1)
    return Xtrain, ytrain

def normer(Xtrain):
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain.reshape(-1, Xtrain.shape[-1])).reshape(Xtrain.shape)
    return Xtrain, scaler

