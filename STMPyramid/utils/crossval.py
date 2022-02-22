from STMPyramid.utils.model import Node
import numpy as np
from STMPyramid.utils.accuracy import accuracy
from random import seed
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV
import joblib
import os
from STMPyramid.utils.visualiser import visualise, visualise_neg, visualise_pos
from constants import *
import pickle
import matplotlib.pyplot as plt
np.random.seed(1)
seed(1)

class model_cv(ClassifierMixin):
    def __init__(self,indim,sol_name='STM',C = 1.0,rank = 3,wconst='maxmax',xa = None,xb = None,constrain = 'lax',wnorm = 'L1',tree_height=3,path=None):
        self.indim = indim
        self.sol_name = sol_name
        self.C = C
        self.rank = rank
        self.wconst = wconst
        self.xa = xa
        self.xb = xb
        self.constrain = constrain
        self.wnorm = wnorm
        self.tree_height = tree_height
        self.path = path
        self.model = Node(indim=self.indim,sol_name=self.sol_name,C=self.C,rank=self.rank,wconst=self.wconst,
                          xa=self.xa,xb=self.xb,constrain=self.constrain,wnorm=self.wnorm,
                          tree_height=self.tree_height,path=self.path)
    def fit(self,X,Y):
        self.model.recursive(X,Y)
        self.model.fine_tune_weights()
        self.model.store_weight()
    def predict(self,X):
        return self.model.forward(X)
    def decision_function(X):
        return np.zeros(X.shape[0])
    def get_params(self,deep=True):
        return {"indim":self.indim,"sol_name":self.sol_name,"C":self.C,"rank":self.rank,"wconst":self.wconst,"constrain":self.constrain,"wnorm":self.wnorm,"tree_height":self.tree_height,"path":self.path}
    def set_params(self, **parameters):
        for parameter,value in list(parameters.items()):
            setattr(self,parameter,value)
        self.model = Node(indim=self.indim,sol_name=self.sol_name,C=self.C,rank=self.rank,wconst=self.wconst,
                          xa=self.xa,xb=self.xb,constrain=self.constrain,wnorm=self.wnorm,
                          tree_height=self.tree_height,path=self.path)
        return self

def grid_search_cv(Xtrain,ytrain):
    model = model_cv(indim=Xtrain.shape[1:],path=path,wconst=wconst)
    grid = GridSearchCV(estimator=model,param_grid=param_grid,n_jobs=-1,verbose=gridsearch_verbose,error_score='raise')
    grid.fit(Xtrain,ytrain)
    best_param = grid.best_params_
    test_acc = grid.best_score_
    print(("Best Parameters : ",best_param))
    print(("Best 5 Fold Cross Validation Accuracy : ",test_acc))
    joblib.dump(grid,path+'/Grid_Search_CV_Results.pkl')
    if(visualiser == True):
        model = model_cv(indim=Xtrain.shape[1:],sol_name=best_param['sol_name'],C = best_param['C'],rank = best_param['rank'],
                         wconst=wconst,xa = None,xb = None,constrain = best_param['constrain'],
                         wnorm = best_param['wnorm'],tree_height=best_param['tree_height'],path=None)
        model.fit(Xtrain,ytrain)
        narr,dic = model.model.store_weight()
        Ypred = model.predict(Xtrain)
        train_acc = accuracy(Ypred,ytrain)
        parent_dir = path
        directory = "Best_Parameters"
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
        st1 = 'Train Accuracy : ' + str(train_acc)+'\n'
        st2 = 'Test Accuracy : ' + str(test_acc)+'\n'
        st3 = 'Best Parameters : ' + str(best_param)+'\n'
        L = [st1, st2, st3]
        s3 = os.path.join(pa,'Best_Parameters.txt')
        f = open(s3,"w")
        f.writelines(L)
        f.close()
        directory = "Images"
        pa = os.path.join(pa,directory)
        try:
            os.mkdir(pa)
        except OSError as error:
            error = 1
        for s in list(dic.keys()):
            fol_name = os.path.join(pa,str(s))
            try:
                os.mkdir(fol_name)
            except OSError as error:
                error = 1
            w_arr = dic[s]
            w1 = visualise(w_arr)
            s1 = os.path.join(fol_name,'weight.png')
            w2 = visualise_pos(w_arr)
            s2 = os.path.join(fol_name,'weight_pos.png')
            w3 = visualise_neg(w_arr)
            s3 = os.path.join(fol_name,'weight_neg.png')
            plt.imsave(s1,w1,cmap='gray',dpi=100)
            plt.imsave(s2,w2,cmap='gray',dpi=100)
            plt.imsave(s3,w3,cmap='gray',dpi=100)
