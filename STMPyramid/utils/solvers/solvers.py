# TODO give option to constrain the weights from min to max OR 0 to max OR -max to max etc
# TODO ask if the max/min of the constraints is AFTER or BEFORE decomposition
# TODO code up primal form of SHTM (is the 'w' on the decomposed matrices of X or on the reconstructed X after the outerproduct summation)
# TODO SHTM is not working (error that expressions with dimensions more than 2 are not supported)
# TODO MCTM is giving mediocre results and doesn't get better with height or C so maybe some bug
from utils.solvers.tensor import make_kernel, rank_R_decomp, construct_W_from_mat
from utils.solvers.vector import inner_prod, construct_W_from_vec, inner_prod_cp
from utils.solvers.centroid import centroid
import cvxpy as cp
import numpy as np
from random import seed
np.random.seed(1)
seed(1)

def SHTM(X,y,C = 1.0,rank = 3,xa = None,xb = None,constrain = 'lax',wnorm = 'L1',wconst='maxmax'):
    M = len(X)
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)
    constrain = M if constrain == 'lax' else 1
    rank = 3 if rank is None else rank

    data_fact = [rank_R_decomp(x, rank) for x in X]
    K = make_kernel(data_fact)
    
    l = cp.Variable(M)
    b = cp.Variable()
    wa = cp.Variable()
    wb = cp.Variable()
    q = cp.Variable(M)
    objfun = C*cp.sum(q)
    if wnorm == 'L1':
        objfun += cp.sum(cp.abs(l@X)) + cp.abs(wa) + cp.abs(wb)
    elif wnorm == 'L2':
        objfun += 1/2*(cp.sum(cp.square(l@X)) + cp.square(wa) + cp.square(wb))
    constraints = []
    if wconst == 'maxmax':
        constraints.append(l >= -1/M*constrain)
        constraints.append(l <= 1/M*constrain)
    elif wconst == 'minmax':
        constraints.append(l >= 0)
        constraints.append(l <= 1/M*constrain)
    constraints.append(q >= 0)
    for i in range(M):
        constraints.append(y[i]*(cp.sum(cp.multiply(l,K[:, i])) + b + cp.multiply(wa,xa[i]) + cp.multiply(wb,xb[i])) + q[i] >= 1)
    
    problem = cp.Problem(cp.Minimize(objfun),constraints)
    problem.solve()
    
    W = construct_W_from_mat(data_fact, l.value, 1e-9)
    
    return W, b.value, wa.value, wb.value

def STM(X, y, C = 1.0, rank = 3, xa = None, xb = None, constrain = 'lax', wnorm = 'L1',wconst = 'maxmax'):
    M = len(X)
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)
    
    wshape = X.shape[1:]
    
    w = cp.Variable(len(X[0].reshape(-1)))
    b = cp.Variable()
    wa = cp.Variable()
    wb = cp.Variable()
    q = cp.Variable(M)
    objfun = C*cp.sum(q)
    if wnorm == 'L1':
        objfun += cp.sum(cp.abs(w)) + cp.abs(wa) + cp.abs(wb)
    elif wnorm == 'L2':
        objfun += 1/2*(cp.sum(cp.square(w)) + cp.square(wa) + cp.square(wb))
    constraints = []
    maxes = np.max(X, axis = 0).reshape(-1)
    mines = np.min(X, axis = 0).reshape(-1)
    if wconst == 'maxmax':
        abmaxes = np.maximum(np.abs(maxes),np.abs(mines))
        constraints.append(w <= abmaxes)
        constraints.append(w >= -abmaxes)
    elif wconst == 'minmax':
        constraints.append(w <= maxes)
        constraints.append(w >= mines)
    constraints.append(q >= 0)
    for i in range(M):
        constraints.append(y[i]*(inner_prod_cp(w,X[i].reshape(-1)) + b + cp.multiply(wa,xa[i]) + cp.multiply(wb,xb[i])) + q[i] >= 1.0)
    constraints.append(wa >= 0)
    constraints.append(wb <= 0)
    problem = cp.Problem(cp.Minimize(objfun),constraints)
    problem.solve()
    
    W = construct_W_from_vec(w.value, wshape)
    return W, b.value, wa.value, wb.value


def MCM(X, y, C = 1.0, rank = 3, xa = None, xb = None, constrain = 'lax', wnorm = 'L1',wconst = 'maxmax'):
    M = len(X)
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)
    
    wshape = X.shape[1:]
    
    w = cp.Variable(len(X[0].reshape(-1)))
    b = cp.Variable()
    wa = cp.Variable()
    wb = cp.Variable()
    q = cp.Variable(M)
    h = cp.Variable()
    objfun = h + C*cp.sum(q)
    constraints = []
    maxes = np.max(X, axis = 0).reshape(-1)
    mines = np.min(X, axis = 0).reshape(-1)
    if wconst == 'maxmax':
        abmaxes = np.maximum(np.abs(maxes),np.abs(mines))
        constraints.append(w <= abmaxes)
        constraints.append(w >= -abmaxes)
    elif wconst == 'minmax':
        constraints.append(w <= maxes)
        constraints.append(w >= mines)
    constraints.append(q >= 0)
    for i in range(M):
        constraints.append(y[i]*(inner_prod_cp(w,X[i].reshape(-1)) + b + cp.multiply(wa,xa[i]) + cp.multiply(wb,xb[i])) + q[i] >= 1.0)
        constraints.append(y[i]*(inner_prod_cp(w,X[i].reshape(-1)) + b + cp.multiply(wa,xa[i]) + cp.multiply(wb,xb[i])) + q[i] <= h)

    
    problem = cp.Problem(cp.Minimize(objfun),constraints)
    problem.solve()
    
    W = construct_W_from_vec(w.value, wshape)
    
    return W, b.value, wa.value, wb.value
        




def MCTM(X, y, C = 1.0, rank = 3, xa = None, xb = None, constrain = 'lax', wnorm = 'L1',wconst = 'maxmax'):
    '''
    If solver doesn't work, then hyperparameters chosen are faulty.
    '''
    M = len(X)
    
    xa = xa if xa is not None else np.zeros(M)
    xb = xb if xb is not None else np.zeros(M)
    rank = 3 if rank is None else rank
    constrain = M if constrain == 'lax' else 1
    data_fact = [rank_R_decomp(x, rank) for x in X]
    K = make_kernel(data_fact)

    h = cp.Variable()
    b = cp.Variable()
    q = cp.Variable(M)
    l = cp.Variable(M)
    wa = cp.Variable()
    wb = cp.Variable()

    obj = h + C*cp.sum(q)

    constraints = []
    for i in range(M):
        constraints.append(h >= y[i]*(cp.sum(cp.multiply(l,K[:, i])) + b + cp.multiply(wa,xa[i]) + cp.multiply(wb,xb[i])) + q[i])
        constraints.append(y[i]*(cp.sum(cp.multiply(l,K[:, i])) + b + cp.multiply(wa,xa[i]) + cp.multiply(wb,xb[i])) + q[i] >= 1)
    constraints.append(q >= 0)
    if wconst == 'maxmax':
        constraints.append(l >= -1/M*constrain)
        constraints.append(l <= 1/M*constrain)
    elif wconst == 'minmax':
        constraints.append(l >= 0)
        constraints.append(l <= 1/M*constrain)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    W = construct_W_from_mat(data_fact, l.value, 1e-9)
    return W, b.value, wa.value, wb.value



def getHyperPlaneFromTwoPoints(xp, xn):
    x1 = centroid(xp)
    x2 = centroid(xn)
    w = (2) * (x2 - x1) / (np.linalg.norm(x1 - x2) ** 2)
    b = -1 * inner_prod(w,(0.5 * (x1 + x2)))  
    return -w, -b
