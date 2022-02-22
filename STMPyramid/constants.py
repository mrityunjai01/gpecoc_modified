# TODO make a way to save output data rather than print it out (especially for images)
dataset = 'mnist'                  # {mnist, cifar10, custom}

# For mnist 
if dataset == 'mnist':
    class1 = 3                       # {0,1,2,3,4,5,6,7,8,9}
    class2 = 8                       # {0,1,2,3,4,5,6,7,8,9}
    totalsamp = 2000                 # int within [50,60000] or None (for max)

# For cifar10
if dataset == 'cifar10':
    class1 = 3                      # {0,1,2,3,4,5,6,7,8,9}
    class2 = 8                      # {0,1,2,3,4,5,6,7,8,9}
    totalsamp = 3000                # int within [50,10000] or None (for max)


# For custom
Xtrain_file = '/path/to/data.npy'   # (Xtrain.shape[0] = total samples and Xtrain is numpy array, no other constraint)
ytrain_file = '/path/to/label.npy'  # (ytrain.shape = (Total Samples, 1) and ytrain is numpy array, no other constraint)
normalised = False                  # normalised data? {True, False}

#The Following are the Hyper-Parameters :
#1. sol_name : Current options are STM, SHTM, MCM, MCTM
#2. C : positive float
#3. rank : positive int
#4. constrain : lax constrain on W {'lax', #ANYTHING} using MCM / MCTM #ANYTHING for not lax
#5. wnorm : {'L1', 'L2'} for L1 or L2 norm using STM / SHTM
#6. tree_height : positive int

#wconst : {'minmax', 'maxmax', #ANYTHING} #ANYTHING for no constraint
wconst = 'maxmax'
#Parameters Grid in Grid Search CV :
param_grid = {'sol_name':['STM'],'C':[0.1],'rank':[3],'constrain':['!lax'],'wnorm':['L1','L2'],'tree_height':[3]}

#Visualiser to visualise images of nodes using best hyperparamters.
visualiser = True
#path : Folder to save all the results.
path = 'Results'
#verbose of grid search CV :
gridsearch_verbose = 3

