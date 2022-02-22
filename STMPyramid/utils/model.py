from STMPyramid.utils.solvers.vector import inner_prod
from STMPyramid.utils.solvers.solvers import getHyperPlaneFromTwoPoints
from STMPyramid.utils.accuracy import accuracy as accuracy_

import numpy as np
from random import seed
import os
import STMPyramid.solvers.solvers as solvers
import pickle
np.random.seed(1)
seed(1)

class Node:
    def __init__(self,indim,sol_name='STM',C = 1.0,rank = 3,wconst='maxmax',xa = None,xb = None,constrain = 'lax',wnorm = 'L1',tree_height=3,path=None):
        self.weight = np.zeros(indim)
        self.bias = 0
        self.A = None
        self.B = None
        self.wA = 0
        self.wB = 0
        self.dim = indim
        self.C1=[]
        self.C2=[]
        self.C3=[]
        self.C4=[]
        self.labels = []
        self.X = []
        self.height = 0
        self.sol_name = sol_name
        self.C = C
        self.rank = rank
        self.wconst = wconst
        self.xa = xa
        self.xb = xb
        self.constrain = constrain
        self.wnorm = wnorm
        self.path = path
        self.tree_height = tree_height
        self.solver = solvers.STM
        if(sol_name == 'MCM'):
            self.solver = solvers.MCM
        if(sol_name == 'MCTM'):
            self.solver = solvers.MCTM
        if(sol_name == 'STM'):
            self.solver = solvers.STM
        if(sol_name == 'SHTM'):
            self.solver = solvers.SHTM

    def insert(self,neuron_type, weight=0, bias=0, w=0):
        if neuron_type == 'A':
            self.A = Node(indim=self.dim,sol_name=self.solver,C=self.C,rank=self.rank,wconst=self.wconst,
                          xa=self.xa,xb=self.xb,constrain=self.constrain,wnorm=self.wnorm,
                          tree_height=self.tree_height,path=self.path)
            self.A.weight = weight
            self.A.bias = bias
            self.A.height = self.height+1
            return self.A
        else:
            self.B = Node(indim=self.dim,sol_name=self.solver,C=self.C,rank=self.rank,wconst=self.wconst,
                          xa=self.xa,xb=self.xb,constrain=self.constrain,wnorm=self.wnorm,
                          tree_height=self.tree_height,path=self.path)
            self.B.weight = weight
            self.B.bias = bias
            self.B.height = self.height+1
        return self.B

    def update_weights_and_bias(self,weight, bias, wA = 0, wB = 0):
        self.weight = weight
        self.bias = bias
        self.wA = wA
        self.wB = wB

    def update_classes(self,ypred,ytrue):
        ypred=ypred.copy()
        ypred=np.reshape(ypred,(ypred.shape[0],1))
        yf  = np.add(2*ypred, ytrue)
        self.C1 = np.argwhere(yf==3)[:,0] #1,1              #In order: predicted, true
        self.C2 = np.argwhere(yf==-3)[:,0] #-1,-1
        self.C3 = np.where(yf==1)[0]   #1,-1
        self.C4 = np.where(yf==-1)[0] #-1,1

    def forward(self, X):
        y=[]
        X=X.copy()
        w = self.weight
        b = self.bias
        wA = np.asarray([self.wA]).copy()
        wB=np.asarray([self.wB]).copy()

        if(self==None):
            return []
        if(self.A==None and self.B==None):
            y = np.sign(np.array(inner_prod(w, X))+np.array(b)).reshape(-1,1)
        if(self.A==None):
            xA = np.zeros((X.shape[0],1))
        else:
            xA = self.A.forward(X)
            xA=np.reshape(xA,(xA.shape[0],1))
        if(self.B==None):
            xB = np.zeros((X.shape[0],1))
        else:
            xB = self.B.forward(X)
            xB=np.reshape(xB,(xB.shape[0],1))
        if(self.A!=None and self.B!=None):
            wA = np.asarray([wA.item()])
            wB = np.asarray([wB.item()])
            y = np.sign(np.asarray(inner_prod(w, X))+np.asarray(inner_prod(wA, xA))+np.asarray(inner_prod(wB, xB))+np.asarray(b)).reshape(-1,1)
        if(self.A!=None and self.B==None):
            wA = np.asarray([wA.item()])
            y = np.sign(np.asarray(inner_prod(w, X))+np.asarray(inner_prod(wA, xA))+np.asarray(b)).reshape(-1,1)
        if(self.A==None and self.B!=None):
            wB = np.asarray([wB.item()])
            y = np.sign(np.asarray(inner_prod(w, X))+np.asarray(inner_prod(wB, xB))+np.asarray(b)).reshape(-1,1)
        return y

    def accuracy(self, xtrain, ytrain):
        return accuracy_(self.forward(xtrain),ytrain)


    def fine_tune_weights(self):
        l=self.labels.copy()
        X = self.X.copy()
        xA = np.zeros((X.shape[0],1))
        xB = np.zeros((X.shape[0],1))
        if(self==None):
            return
        if(self.A!=None):
            self.A.fine_tune_weights()
            xA = self.A.forward(X)
            xA=np.reshape(xA,(xA.shape[0],1))
        if(self.B!=None):
            self.B.fine_tune_weights()
            xB = self.B.forward(X)
            xB=np.reshape(xB,(xB.shape[0],1))

        weight, bias, wA, wB = self.solver(X=X,y=l,C=self.C,rank=self.rank,xa=self.xa,xb=self.xb,
                                           constrain=self.constrain,wnorm=self.wnorm,wconst=self.wconst)

        self.update_weights_and_bias(weight, bias, wA, wB)


    def recursive(self, X, labels):
        h = self.tree_height
        self.X = X
        self.labels = labels
        labels=labels.copy()
        X=X.copy()
        weight, bias, _, _1_ = self.solver(X=X,y=labels,C=self.C,rank=self.rank,xa=self.xa,xb=self.xb,
                                           constrain=self.constrain,wnorm=self.wnorm,wconst=self.wconst)
        self.update_weights_and_bias(weight, bias)
        ypred=self.forward(X)
        self.update_classes(ypred,labels)
        C1=self.C1
        C2=self.C2
        C3=self.C3
        C4=self.C4
        if(len(C3)==0 and len(C4)==0):
            return
        if(self.height>h-1):
            return
        if(len(C1)==0 or len(C2)==0):
            if(len(C1)!=0):
                X_positive=np.take(X,np.hstack((C1,C4)),axis=0)
                X_negative=np.take(X,np.hstack((C3)),axis=0)
            elif(len(C2)!=0):
                X_positive=np.take(X,np.hstack((C4)),axis=0)
                X_negative=np.take(X,np.hstack((C2,C3)),axis=0)
            else:
                X_positive=np.take(X,np.hstack((C4)),axis=0)
                X_negative=np.take(X,np.hstack((C3)),axis=0)
            weight, bias = getHyperPlaneFromTwoPoints(X_positive, X_negative)
            self.update_weights_and_bias(weight, bias)
            ypred = self.forward(X)
            self.update_classes(ypred,labels)
            C1=self.C1
            C2=self.C2
            C3=self.C3
            C4=self.C4

        if(len(C3)!=0):
            X_new=np.take(X,np.hstack((C1,C3,C4)),axis=0)
            labels[C1]=-1
            labels[C3]=1
            labels[C4]=-1
            y_new=np.take(labels,np.hstack((C1,C3,C4)),axis=0)
            NodeA = self.insert('A')
            NodeA.recursive(X_new, y_new)

        if(len(C4) != 0):
            X_new=np.take(X,np.hstack((C2,C3,C4)),axis=0)
            labels[C2]=-1
            labels[C3]=-1
            labels[C4]=1
            y_new=np.take(labels,np.hstack((C2,C3,C4)),axis=0)
            NodeB = self.insert('B')
            NodeB.recursive(X_new, y_new)


    def store_weight(self):
        narr = np.zeros(self.X.shape[1:])
        npos = np.zeros(2)
        narr = narr[None,:,:]
        npos = npos[None,:]
        narr,npos = self.dfs(narr=narr,npos=npos,x1=0,x2=0)
        narr = narr[1:]
        npos = npos[1:]
        dic = dict()
        leng = npos.shape[0]
        i = 0
        while(i < leng):
            tp = (npos[i,0],npos[i,1])
            dic[tp] = narr[i]
            i = i+1
        if(self.path != None):
            parent_dir = self.path
            directory = 'solver='+str(self.sol_name)+'_Tree_Height='+str(self.tree_height)+'_C='+str(self.C)+'_rank='+str(self.rank) + '_constrain=' + str(self.constrain) + '_wnorm=' + str(self.wnorm)
            pa = os.path.join(parent_dir,directory)
            try:
                os.mkdir(pa)
            except OSError as error:
                error = 1
            s1 = os.path.join(pa,'weight.npy')
            np.save(s1,narr)
            try:
                s2 = os.path.join(pa,'dictionary_weight')
                dic_weight_file = open(s2,'wb')
                pickle.dump(dic,dic_weight_file)
                dic_weight_file.close()
            except:
                print("Dictionary Weight File not save")
        return narr,dic

    def dfs(self,narr,npos,x1,x2):
        if (self == None):
            return
        pos = np.zeros(2)
        pos[0] = x1
        pos[1] = x2
        npos = np.append(npos,[pos],axis=0)
        sh = self.X.shape[1:]
        wf = np.reshape(self.weight,sh)
        '''if len(shape) == 3:
                wf = np.average(self.weight,axis = -1)'''
        narr = np.append(narr,[wf],axis=0)
        if(self.A != None):
            x1 = x1-1
            x2 = x2+1
            narr,npos = self.A.dfs(narr,npos,x1,x2)
        if(self.B != None):
            x1 = x1+1
            x2 = x2+1
            narr,npos = self.B.dfs(narr,npos,x1,x2)
        return narr,npos
