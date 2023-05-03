from errno import ENETUNREACH
import numpy as np
import numpy.linalg as la
import numba as nb
from time import time
from sklearn.utils import extmath
from collections import namedtuple
import warnings
from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from numba import prange


def skl_svd(X):
    return extmath.randomized_svd(X,n_components=1)[1][0]

@nb.njit(cache=True)
def prox_L1(beta, beta_tilde1, lambda1):
    beta_sub = beta-beta_tilde1
    abs_beta = np.abs(beta_sub)
    return beta_tilde1 + np.where(abs_beta>lambda1, abs_beta-lambda1, 0) * np.sign(beta_sub)

@nb.njit(cache=True)
def clip(beta, M):
    abs_beta = np.abs(beta)
    return np.where(abs_beta>M, M, abs_beta) * np.sign(beta)


@nb.njit
def mvm(A,b,index,transpose=False):
    n,p = A.shape
    if not transpose:
        res = np.zeros(n,dtype=b.dtype)
        for j,i in enumerate(index):
            res += A[:,i]*b[j]
    else:
        res = np.zeros(len(index),dtype=b.dtype)
        for j,i in enumerate(index):
            res[j] = A[:,i]@b
    return res
@nb.njit(parallel=True)
def pmvm(A,b,index,transpose=False):
    n,p = A.shape
    if not transpose:
        res = np.zeros(n,dtype=b.dtype)
        for j in prange(len(index)):
            res += A[:,index[j]]*b[j]
        return res
    else:
        res = np.zeros(len(index),dtype=b.dtype)
        for j in prange(len(index)):
            res[j] = A[:,index[j]]@b
        return res

@nb.njit
def mmm(A,index):
    n,p = A.shape
    res = np.zeros((n,n),dtype=A.dtype)
    for i in range(n):
        res[i] = mvm(A,A[i][index],index,False)

    return res


@nb.njit(parallel=True)
def pmmm(A,index):
    n,p = A.shape
    res = np.zeros((n,n),dtype=A.dtype)
    for i in prange(n):
        res[i] = mvm(A,A[i][index],index,False)

    return res


@nb.njit(cache=True)
def hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M=np.inf):
    
    beta_hat = (1-2*lambda2/L)*beta + (X.T@r-alpha+2*lambda2*beta_tilde2)/L
    beta_new = clip(prox_L1(beta_hat, beta_tilde1, lambda1/L), M)
    rec_obj = (1/2)*(beta_new - beta_hat)**2+lambda1/L*np.abs(beta_new-beta_tilde1) - ((1/2)*(beta_hat)**2+lambda1/L*np.abs(beta_tilde1))
    argsort = np.argsort(rec_obj)
    beta_new[argsort[k:]] = 0
    
    _, p = X.shape
    if p > 5*k:
        r = y - X[:,argsort[:k]]@beta_new[argsort[:k]]
    else:
        r = y - X@beta_new
    
    return beta_new, r

#@nb.njit(cache=True)
def hard_thresholding_ls(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M=np.inf,sea_max_itr=5):
    
    _, p = X.shape
    support = np.where(beta!=0)[0]
    support_inv = np.where(beta==0)[0]
    XTr = X.T@r
    beta_sub2 = beta_tilde2 - beta
    grad = -XTr + alpha - 2*lambda2*beta_sub2
    grad_supp = np.zeros(p)
    grad_supp[support] = grad[support]
    
    max_suppinv = np.max(np.abs(grad[support_inv]))
    same_sign = np.sign(beta[support])==np.sign(grad[support])
    L_change = np.min(np.where( same_sign + (np.abs(grad[support])<max_suppinv),
         np.abs(beta[support])/(np.abs(grad[support])*(2*same_sign-1)+max_suppinv),np.inf))
   
    
    if p > 5*k:
        Xgrad = X[:,support]@grad[support]
    else:
        Xgrad = X@grad_supp
    
 
    opt_step = (grad_supp@alpha - r@Xgrad - 2*lambda2*beta_sub2@grad_supp)/(Xgrad@Xgrad+2*lambda2*grad_supp@grad_supp)
    
    #print("----opt step is",opt_step,"Lchange is",L_change)
    if opt_step < L_change:
        beta_new = beta - opt_step*grad
        argsort = np.argsort(-np.abs(beta_new))
        beta_new[argsort[k:]] = 0
        if p > 5*k:
            r = y - X[:,argsort[:k]]@beta_new[argsort[:k]]
        else:
            r = y - X@beta_new
    else:
        f_best = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)   
        sea_itr = 0
        #L_step = np.maximum(L_change,1/L)
        L_step = L_change/(1+1e-4)
        beta_new = np.copy(beta)
        while sea_itr < sea_max_itr:
            
            beta_tmp = beta - L_step*grad
            argsort = np.argsort(-np.abs(beta_tmp))
            beta_tmp[argsort[k:]] = 0
            if p > 5*k:
                r_tmp = y - X[:,argsort[:k]]@beta_tmp[argsort[:k]]
            else:
                r_tmp = y - X@beta_tmp
                
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
            #print("f_new is",f_new,"f_best is",f_best,"step is",L_step)
            if f_new < f_best:
                f_best = f_new
                beta_new = np.copy(beta_tmp)
                r = np.copy(r_tmp)
            else:
                break
         
            L_step *= 2
            sea_itr += 1
    
    return beta_new, r

@nb.njit(cache=True)
def prune_ls(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=np.inf,sea_max_itr=5):
    
    _, p = X.shape
    argsort = np.argsort(-np.abs(beta))
    support = argsort[:k]
    support_inv = argsort[k:]
    
    XTr = X.T@r
    beta_sub2 = beta_tilde2 - beta
    grad = -XTr + alpha - 2*lambda2*beta_sub2
    grad = grad.astype(X.dtype)
    grad_supp = np.zeros(p,dtype=X.dtype)
    grad_supp[support] = grad[support]
    
    if p > 5*k:
        Xgrad = X[:,support]@grad[support]
    else:
        Xgrad = X@grad_supp
        
    opt_step = ((grad_supp@alpha) - (r@Xgrad) - 2*lambda2*(beta_sub2@grad_supp))/((Xgrad@Xgrad) +2*lambda2*(grad_supp@grad_supp))
        
    sup_max = np.max(np.abs(beta[support]-opt_step*grad[support]))
    supinv_max = np.max(np.abs(beta[support_inv]-opt_step*grad[support_inv]))
    
    if sup_max >= supinv_max - 1e-10:
        # opt_step is less change step
        #print("Use opt step",opt_step)
        beta_new = beta - opt_step*grad
        beta_new = beta_new.astype(X.dtype)
        beta_new[argsort[k:]] = 0 
        if p > 5*k:
            r = y - X[:,argsort[:k]]@beta_new[argsort[:k]]
        else:
            r = y - X@beta_new
    else:
        #print("Use line search step, opt step is",opt_step)
        L_step = opt_step/2
        sea_itr = 0
        while sea_itr < 100:
            sup_max = np.max(np.abs(beta[support]-L_step*grad[support]))
            supinv_max = np.max(np.abs(beta[support_inv]-L_step*grad[support_inv]))
            if sup_max >= supinv_max - 1e-10:
                break
            L_step /= 2
            sea_itr += 1
        
        beta_mp = np.zeros(p)
        beta_mp[support] = beta[support]
        beta_mp = beta_mp.astype(X.dtype)
        f_best = evaluate_obj(beta_mp,y-X@beta_mp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2) 
        sea_itr = 0
        while sea_itr < sea_max_itr:
            
            beta_tmp = beta - L_step*grad
            beta_tmp = beta_tmp.astype(X.dtype)
            argsort = np.argsort(-np.abs(beta_tmp))
            beta_tmp[argsort[k:]] = 0
            if p > 5*k:
                r_tmp = y - X[:,argsort[:k]]@beta_tmp[argsort[:k]]
            else:
                r_tmp = y - X@beta_tmp
                
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
            #print("f_new is",f_new,"f_best is",f_best,"step is", L_step)
            if f_new < f_best:
                f_best = f_new
                beta_new = np.copy(beta_tmp)
                r = np.copy(r_tmp)
            else:
                break
            L_step *= 2
            sea_itr += 1
            
    return beta_new, r


@nb.njit(cache=True)
def DIHT_update(y,X,r,gamma,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,eta,M=np.inf):
    
    gamma = (1-eta)*gamma - eta*r
    quad_term = 2*lambda2
    linear_term = 2*lambda2*beta_tilde2-alpha-X.T@gamma
    beta = clip(prox_L1(linear_term, quad_term*beta_tilde1, lambda1)/(quad_term), M)
    rec_obj = (1/2)*quad_term*beta**2 - linear_term*beta + lambda1*np.abs(beta-beta_tilde1) - lambda1*np.abs(beta_tilde1)
    argsort = np.argsort(rec_obj)
    beta[argsort[k:]] = 0
    r = y - X[:,argsort[:k]]@beta[argsort[:k]]
    
    return beta, r, gamma

@nb.njit(cache=True)
def SDIHT_update(y,X,X_idx,beta,gamma,Xgamma,idx_s,idx_e,nnz_idx,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,eta,M=np.inf):
    
    _, p = X.shape
    if p > 10*k:
        r_idx = y[idx_s:idx_e] - X_idx[:,nnz_idx]@beta[nnz_idx]
    else:
        r_idx = y[idx_s:idx_e] - X_idx@beta
        
    #r_idx = y[idx_s:idx_e] - X_idx[:,nnz_idx]@beta[nnz_idx]
    grad_idx = gamma[idx_s:idx_e]+r_idx
    Xgamma -= eta*(X_idx.T@grad_idx)
    gamma[idx_s:idx_e] -= eta*grad_idx
    
    quad_term = 2*lambda2
    linear_term = 2*lambda2*beta_tilde2-alpha-Xgamma
    beta = clip(prox_L1(linear_term, quad_term*beta_tilde1, lambda1)/(quad_term), M)
    rec_obj = (1/2)*quad_term*beta**2 - linear_term*beta + lambda1*np.abs(beta-beta_tilde1) - lambda1*np.abs(beta_tilde1)
    argsort = np.argsort(rec_obj)
    beta[argsort[k:]] = 0
    
    return beta, gamma, Xgamma, argsort[:k]
    

@nb.njit(cache=True)
def sto_BCD(y,X,beta,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,gradB,stepL,M=np.inf,sto_iter=100,sto_m="cyc"):
    
    r = y - X@beta
    objs = list()
    for i in range(sto_iter):
        
        if sto_m == "cyc":
            for j in range(len(stepL)):
                L = stepL[j]
                beta_cur = beta[gradB[j]:gradB[j+1]]
                beta_hat = (1-2*lambda2/L)*beta_cur + (X[:,gradB[j]:gradB[j+1]].T@r-alpha[gradB[j]:gradB[j+1]]+2*lambda2*beta_tilde2[gradB[j]:gradB[j+1]])/L
                beta_new = clip(prox_L1(beta_hat, beta_tilde1[gradB[j]:gradB[j+1]], lambda1/L), M)
   
                r -= X[:,gradB[j]:gradB[j+1]]@(beta_new-beta_cur)
                beta[gradB[j]:gradB[j+1]] = beta_new
                objs.append(evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2))
        elif sto_m == "sto":
            for _ in range(len(stepL)):
                j = np.random.randint(len(stepL))
                L = stepL[j]
                beta_cur = beta[gradB[j]:gradB[j+1]]
                beta_hat = (1-2*lambda2/L)*beta_cur + (X[:,gradB[j]:gradB[j+1]].T@r-alpha[gradB[j]:gradB[j+1]]+2*lambda2*beta_tilde2[gradB[j]:gradB[j+1]])/L
                beta_new = clip(prox_L1(beta_hat, beta_tilde1[gradB[j]:gradB[j+1]], lambda1/L), M)
   
                r -= X[:,gradB[j]:gradB[j+1]]@(beta_new-beta_cur)
                beta[gradB[j]:gradB[j+1]] = beta_new
                objs.append(evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2))
    
    return beta, r, objs

@nb.njit(cache=True)
def coordinate_descent(y,X,beta,r,i,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=np.inf):

    beta_inew = clip(prox_L1(2*lambda2*beta_tilde2[i]+X[:,i]@r+S_diag[i]*beta[i]-alpha[i], beta_tilde1[i]*(S_diag[i]+2*lambda2), lambda1)/(S_diag[i]+2*lambda2), M)
    
    return beta_inew

@nb.njit(cache=True)
def CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=np.inf, cd_itr=1, sto_m = "cyc", cd_tol = -1):
    
    f_old = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    for i in range(cd_itr):
        support = np.where(beta!=0)[0]
        if sto_m == "cyc":
            for j in support:
                beta_inew = coordinate_descent(y,X,beta,r,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
                r = r - (beta_inew - beta[j])*X[:,j]
                beta[j] = beta_inew
        elif sto_m == "sto":
            for _ in range(len(support)):
                j = np.random.randint(len(support))
                beta_inew = coordinate_descent(y,X,beta,r,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
                r = r - (beta_inew - beta[j])*X[:,j]
                beta[j] = beta_inew
        
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)* cd_tol):
            break
        f_old = f
                        
            
    return beta, r


@nb.njit(cache=True)
def Check_Swap1(y,X,beta,r,i,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=np.inf):
    
    quad_term = S_diag[j]+2*lambda2
    linear_term = 2*lambda2*beta_tilde2[j]+X[:,j]@r+(X[:,j]@X[:,i])*beta[i]-alpha[j]
    beta_jnew = clip(prox_L1(linear_term, quad_term*beta_tilde1[j], lambda1)/(quad_term), M)
    beta_reduce = (1/2)*quad_term*beta_jnew**2 - linear_term*beta_jnew + lambda1*(np.abs(beta_jnew-beta_tilde1[j]) - np.abs(beta_tilde1[j]))
    
    return beta_jnew, beta_reduce


@nb.njit(cache=True)
def Check_Swaprow(y,X,beta,r,support,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=np.inf):
    
    bst_imp, bst_i, bst_v, f = 0, -1, 0, evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    for i in support:
        beta_jnew, beta_reduce = Check_Swap1(y,X,beta,r,i,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
        
        beta_iold = beta[i]
        beta[i] = 0
        beta[j] = beta_jnew
        r_old = np.copy(r)
        r = r - X[:,j]*beta_jnew + X[:,i]*beta_iold
        cur_imp = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2) - f
        beta[i] = beta_iold
        beta[j] = 0
        r = r_old
        
        if cur_imp < bst_imp:
            bst_imp = cur_imp
            bst_i = i
            bst_v = beta_jnew
            break
            
    return bst_imp, bst_i, bst_v


@nb.njit(cache=True)
def Check_Swapall(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M=np.inf):
    
    j, bst_i, bst_v = -1,-1,-1
    support_set = set(np.where(beta!=0)[0])
    support_c = np.where(beta==0)[0] 
    for j in support_c:
        bst_imp, bst_i, bst_v = Check_Swaprow(y,X,beta,r,support_set,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
        if bst_imp < 0 and bst_v != 0:
            break
                
    return j, bst_i, bst_v


@nb.njit(cache=True)
def evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2):
    beta_sub1 = beta - beta_tilde1
    beta_sub2 = beta - beta_tilde2
    return 0.5*(r@r) + lambda2*(beta_sub2@beta_sub2) + lambda1*(np.sum(np.abs(beta_sub1))) + alpha@beta


def initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M=np.inf,buget=None,kimp=2.,act_itr=1):
    
    p = beta.shape[0]
    buget = p if buget is None else buget
    ksupp = int(np.max([np.min([kimp*k, buget, p]),k]))
    beta_tmp, r_tmp = np.copy(beta), np.copy(r)
    for i in range(act_itr):
        beta_tmp,r_tmp = hard_thresholding(y,X,beta_tmp,r_tmp,ksupp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
    active_set = set(np.where(beta_tmp)[0])    
    active_set = np.array(sorted(active_set),dtype=int)
    
    return active_set



def Vanilla_DIHTCD(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,eta=1e-3,M=np.inf,diht_max_itr=100,cd_refine = 100):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    r = y - X@beta
    S_diag = np.linalg.norm(X, axis=0)**2
    gamma = -np.copy(r)
    diht_cur_itr = 0
    objs = []
    priobj = []
    duaobj = []
    f_best = np.inf
    beta_best = np.copy(beta)
    r_best = np.copy(r)
    f_old = np.inf
    while diht_cur_itr < diht_max_itr:
        
        beta, r, gamma = DIHT_update(y,X,r,gamma,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,eta/(1+diht_cur_itr)**0.5,M)
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        
        if diht_cur_itr%100==0:
            print("primal obj ",f)
            print("dual   obj ",(X.T@gamma)@beta+alpha@beta+lambda2*(beta-beta_tilde2)@(beta-beta_tilde2)+
                  lambda1*np.sum(np.abs(beta-beta_tilde1))-(1/2)*gamma@gamma-gamma@y)
            
        
        if f < f_best:
            beta_best = np.copy(beta)
            r_best = np.copy(r)
            f_best = f
                
        objs.append(f)
        priobj.append(f)
        duaobj.append((X.T@gamma)@beta+alpha@beta+lambda2*(beta-beta_tilde2)@(beta-beta_tilde2)+
                  lambda1*np.sum(np.abs(beta-beta_tilde1))-(1/2)*gamma@gamma-gamma@y)
        f_old = f
        diht_cur_itr += 1
        
    beta,r = CD_loop(y,X,beta_best,r_best,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_refine)
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    sol_time = time()-st
    
    return beta, f, objs, r, diht_cur_itr, sol_time, priobj, duaobj


def Vanilla_SDIHTCD(y,X,beta,k,batch,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,eta=1e-3,M=np.inf,diht_max_itr=100,cd_refine = 100):
    
    st = time()

    n,p = X.shape
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    r = y - X@beta
    S_diag = np.linalg.norm(X, axis=0)**2
    gamma = -np.copy(r)
    Xgamma = X.T@gamma
    diht_cur_itr = 0
    objs = []
    f_best = np.inf
    beta_best = np.copy(beta)
    r_best = np.copy(r)
    idx_list = list(range(0,n,int(n/batch)))
    idx_list.append(n)
    nnz_idx = np.where(beta)[0]
    while diht_cur_itr < diht_max_itr:
        
        for batch_iter in range(len(idx_list)-1):
            beta, gamma, Xgamma, nnz_idx = SDIHT_update(y,X,X[idx_list[batch_iter]:idx_list[batch_iter+1],:],beta,gamma,Xgamma,idx_list[batch_iter],
                            idx_list[batch_iter+1],nnz_idx,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,eta,M)
        
        
        
        
        if diht_cur_itr%100==0:
            r = y-X[:,nnz_idx]@beta[nnz_idx]
            f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
            print("primal obj ",f)
            print("dual   obj ",(X.T@gamma)@beta+alpha@beta+lambda2*(beta-beta_tilde2)@(beta-beta_tilde2)+
                  lambda1*np.sum(np.abs(beta-beta_tilde1))-(1/2)*gamma@gamma-gamma@y)
            
        
            if f < f_best:
                beta_best = np.copy(beta)
                r_best = np.copy(r)
                f_best = f
                
        objs.append(f)
        diht_cur_itr += 1
        
    beta,r = CD_loop(y,X,beta_best,r_best,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_refine)
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    sol_time = time()-st
    
    return beta, f, objs, r, diht_cur_itr, sol_time


def Active_SDIHTCD(y,X,beta,k,batch=1,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, eta=1e-3, L=None, M=np.inf, 
                diht_max_itr=100, act_max_itr=10, buget=None,kimp=2.,act_itr=1,cd_refine=0, sea_max_itr=10):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X)**2+lambda2*2) if L is None else L
    r = y - X@beta
    S_diag = np.linalg.norm(X, axis=0)**2
    gamma = -np.copy(r)
    f_best = np.inf
    beta_best = np.copy(beta)
    r_best = np.copy(r)
    act_cur_itr = 0
    _sol_str = 'active_set objs, r, diht_cur_itr sol_time search_itr outliers'
    Solution = namedtuple('Solution', _sol_str)
    sols = []
    active_set = initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M,buget,kimp,act_itr)
    
    while act_cur_itr < act_max_itr:
        
        X_act = X[:,active_set]
        beta_act = beta[active_set]
        
        st = time()
        print("active iter:",act_cur_itr)

    
        #===================================
        # SIHTCD
        
        S_diag_act = np.linalg.norm(X_act, axis=0)**2
        Xgamma_act = X_act.T@gamma
        diht_cur_itr = 0
        f_best = np.inf
        objs = []
        n_act = len(active_set)
        idx_list = list(range(0,n_act,int(n_act/batch)))
        idx_list.append(n_act)
        nnz_idx = np.where(beta_act)[0]
        
        lambda2_other = lambda2 * beta_tilde2@beta_tilde2 - lambda2 * beta_tilde2[active_set]@beta_tilde2[active_set]
        lambda1_other = lambda1 * np.sum(np.abs(beta_tilde1)) - lambda1 * np.sum(np.abs(beta_tilde1[active_set])) 
        
        while diht_cur_itr < diht_max_itr:
        
            for batch_iter in range(len(idx_list)-1):
                beta_act, gamma, Xgamma_act, nnz_idx = SDIHT_update(y,X_act,X_act[idx_list[batch_iter]:idx_list[batch_iter+1],:],beta_act,gamma,
                                      Xgamma_act,idx_list[batch_iter],idx_list[batch_iter+1],nnz_idx,k,alpha[active_set],
                                      lambda1,lambda2,beta_tilde1[active_set],beta_tilde2[active_set],eta,M)
        
            if diht_cur_itr%100==0:
                r = y-X_act[:,nnz_idx]@beta_act[nnz_idx]
                f = evaluate_obj(beta_act,r,alpha[active_set],lambda1,lambda2,beta_tilde1[active_set],beta_tilde2[active_set]) +\
                      lambda2_other + lambda1_other
                print("primal obj ",f)
                print("dual   obj ",(X_act.T@gamma)@beta_act+alpha[active_set]@beta_act+
                      lambda2*(beta_act-beta_tilde2[active_set])@(beta_act-beta_tilde2[active_set]) + lambda2_other + lambda1_other+\
                      lambda1*np.sum(np.abs(beta_act-beta_tilde1[active_set]))-(1/2)*gamma@gamma-gamma@y)
            
        
                if f < f_best:
                    beta_best = np.copy(beta_act)
                    r_best = np.copy(r)
                    f_best = f
                
            objs.append(f)
            diht_cur_itr += 1
        
        beta_act, r = CD_loop(y,X_act,beta_best,r_best,S_diag_act,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,50)
        f = evaluate_obj(beta_act,r,alpha[active_set],lambda1,lambda2,beta_tilde1[active_set],beta_tilde2[active_set]) +\
                      lambda2_other + lambda1_other
        sol_time = time()-st
        #=========================================
        
        L_init = 2*(skl_svd(X_act)**2+lambda2*2) 
        beta = np.zeros(p)
        beta[active_set] = beta_act
        r = y - X@beta
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        active_set_set = set(active_set)
        search_flag = False
        search_cur_itr = 0
        outliers = set()
        beta_update,r_update = beta,r
        while search_cur_itr < sea_max_itr:
            beta_tmp, r_tmp = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L_init,M)
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
            outliers = set(np.where(beta_tmp)[0]) - active_set_set
            search_cur_itr += 1
            print(f_new,f,len(outliers),search_cur_itr)
            if len(outliers) >= 1 and f_new < f:
                search_flag = True
                beta_update = beta_tmp
                r_update = r_tmp
            elif f_new >= f:
                beta = beta_update 
                r = r_update
                break
            L_init /= 2
            
        sols.append(Solution(active_set=active_set,objs=objs,r=np.copy(r),diht_cur_itr=diht_cur_itr,sol_time=sol_time,
                             search_itr=search_cur_itr,outliers=len(outliers)))
        if not search_flag:
            break
        active_set = np.array(sorted(active_set_set | outliers))
        act_cur_itr += 1
    
    beta,r = CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_refine)
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    tot_time = time()-st
    return beta, f, sols, r, act_cur_itr, tot_time



def Vanilla_IHTCD(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,L=None,M=np.inf,iht_max_itr=100,ftol=1e-8,
                  cd_itr=0,ctol=1e-4,search_max_itr=1):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X)**2+lambda2*2) if L is None else L
    r = y - X@beta
    S_diag = np.linalg.norm(X, axis=0)**2
    iht_cur_itr = 0
    objs = []
    beta, r = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
    f_old = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    while iht_cur_itr < iht_max_itr:
        
        search_flag = False
        search_cur_itr = 0
        L_init = L
        while True:
            beta_tmp, r_tmp = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L_init,M)
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
            if f_new < f_old - L_init/10 * (beta_tmp-beta)@(beta_tmp-beta) or search_cur_itr == 0:
                beta_update = beta_tmp
                r_update = r_tmp
            else:
                beta = beta_update
                r = r_update
                break
            search_cur_itr += 1
            if search_cur_itr >= search_max_itr:
                beta = beta_update
                r = r_update
                break
            L_init /= 2
   
        #beta, r = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
        
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)* ctol) and iht_cur_itr > 0:
            beta,r = CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_itr)
            f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
                
        objs.append(f)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)*ftol) and iht_cur_itr > 0:
            break
        f_old = f
        iht_cur_itr += 1
        
    sol_time = time()-st
    return beta, f, objs, r, iht_cur_itr, sol_time

def Vanilla_IHTCDLS(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,L=None,M=np.inf,iht_max_itr=100,ftol=1e-8,
                  cd_itr=0,ctol=1e-4,search_max_itr=1):
    
    assert lambda1==0
    assert M==np.inf
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X)**2+lambda2*2) if L is None else L
    r = y - X@beta
    S_diag = np.linalg.norm(X, axis=0)**2
    iht_cur_itr = 0
    objs = []
    beta, r = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M)
    f_old = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
    while iht_cur_itr < iht_max_itr:
        
        beta, r = hard_thresholding_ls(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M, search_max_itr)
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)* ctol) and iht_cur_itr > 0:
            beta,r = CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_itr)
            f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
                
        objs.append(f)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)*ftol) and iht_cur_itr > 0:
            break
        f_old = f
        iht_cur_itr += 1
        
    sol_time = time()-st
    return beta, f, objs, r, iht_cur_itr, sol_time

def Active_IHTCD(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=np.inf, 
                iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
                sea1_max_itr=5, sea2_max_itr=10):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X)**2+lambda2*2) if L is None else L
    r = y - X@beta
    f_old = np.inf
    act_cur_itr = 0
    _sol_str = 'active_set objs, r, iht_cur_itr sol_time search_itr outliers'
    Solution = namedtuple('Solution', _sol_str)
    sols = []
    
    active_set = initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M,buget,kimp,act_itr)
    
    while act_cur_itr < act_max_itr:
        
        X_act = X[:,active_set]
        beta_act = beta[active_set]
        L_act = 1.05*(skl_svd(X_act)**2+lambda2*2) 
        
        beta_act, f, objs, r_act, iht_cur_itr, sol_time = Vanilla_IHTCD(y,X_act,beta_act,k,alpha[active_set],lambda1,lambda2,beta_tilde1[active_set],
                                                                     beta_tilde2[active_set], L_act,M,iht_max_itr,ftol,cd_itr,ctol,sea1_max_itr)
        
        print("Num of iter:",act_cur_itr+1," num of inner iter:",iht_cur_itr,"\n Finding new active set")
        L_init = 2*L_act
        beta = np.zeros(p)
        beta[active_set] = beta_act
        r = y - X@beta
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        active_set_set = set(active_set)
        search_flag = False
        search_cur_itr = 0
        outliers = set()
        beta_update,r_update = beta,r
        while search_cur_itr < sea2_max_itr:
            beta_tmp, r_tmp = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L_init,M)
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
            outliers = set(np.where(beta_tmp)[0]) - active_set_set
            search_cur_itr += 1
            print(f_new,f,len(outliers),search_cur_itr)
            if len(outliers) >= 1 and f_new < f:
                search_flag = True
                beta_update = beta_tmp
                r_update = r_tmp
            elif f_new >= f:
                beta = beta_update 
                r = r_update
                break
            L_init /= 2
            
        sols.append(Solution(active_set=active_set,objs=objs,r=np.copy(r_act),iht_cur_itr=iht_cur_itr,sol_time=sol_time,
                             search_itr=search_cur_itr,outliers=len(outliers)))
        if not search_flag:
            break
        active_set = np.array(sorted(active_set_set | outliers))
        act_cur_itr += 1
    
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    tot_time = time()-st
    return beta, f, sols, r, act_cur_itr, tot_time

def Active_IHTCDLS(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=np.inf, 
                iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
                sea1_max_itr=5, sea2_max_itr=10):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X)**2+lambda2*2) if L is None else L
    r = y - X@beta
    f_old = np.inf
    act_cur_itr = 0
    _sol_str = 'active_set objs, r, iht_cur_itr sol_time search_itr outliers'
    Solution = namedtuple('Solution', _sol_str)
    sols = []
    
    active_set = initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M,buget,kimp,act_itr)
    
    while act_cur_itr < act_max_itr:
        
        X_act = X[:,active_set]
        beta_act = beta[active_set]
        L_act = 1.05*(skl_svd(X_act)**2+lambda2*2) 
        
        beta_act, f, objs, r_act, iht_cur_itr, sol_time = Vanilla_IHTCDLS(y,X_act,beta_act,k,alpha[active_set],lambda1,lambda2,beta_tilde1[active_set],
                                                                     beta_tilde2[active_set], L_act,M,iht_max_itr,ftol,cd_itr,ctol,sea1_max_itr)
        
        print("Num of iter:",act_cur_itr+1," num of inner iter:",iht_cur_itr,"\n Finding new active set")
        L_init = 2*L_act
        beta = np.zeros(p)
        beta[active_set] = beta_act
        r = y - X@beta
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        active_set_set = set(active_set)
        search_flag = False
        search_cur_itr = 0
        outliers = set()
        beta_update,r_update = beta,r
        while search_cur_itr < sea2_max_itr:
            beta_tmp, r_tmp = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L_init,M)
            f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
            outliers = set(np.where(beta_tmp)[0]) - active_set_set
            search_cur_itr += 1
            print(f_new,f,len(outliers),search_cur_itr)
            if len(outliers) >= 1 and f_new < f:
                search_flag = True
                beta_update = beta_tmp
                r_update = r_tmp
            elif f_new >= f:
                beta = beta_update 
                r = r_update
                break
            L_init /= 2
            
        sols.append(Solution(active_set=active_set,objs=objs,r=np.copy(r_act),iht_cur_itr=iht_cur_itr,sol_time=sol_time,
                             search_itr=search_cur_itr,outliers=len(outliers)))
        if not search_flag:
            break
        active_set = np.array(sorted(active_set_set | outliers))
        act_cur_itr += 1
    
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    tot_time = time()-st
    return beta, f, sols, r, act_cur_itr, tot_time

def Vanilla_CDPSI(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,L=None,M=np.inf,iht_init_itr=100,
                  cd_itr=10,psi_max_itr=100, ftol=1e-8):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X)**2+lambda2*2) if L is None else L
    r = y - X@beta
    S_diag = np.linalg.norm(X, axis=0)**2
    f_old = np.inf
    psi_cur_itr = 0
    objs = []
    beta, f, _, r, _, _ = Vanilla_IHTCD(y,X,beta,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M,
                                        iht_init_itr,1e-8,0,0)
    
    support_set = set(np.where(beta>0)[0])
    len_support = len(support_set)
    while psi_cur_itr < psi_max_itr:
        
        beta, r = CD_loop(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,cd_itr)
        support_set = set(np.where(beta>0)[0])
        len_support = len(support_set)
               
        support_c = np.where(beta==0)[0]
        for j in support_c:
            if len_support < k:
                beta_inew = coordinate_descent(y,X,beta,r,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
                r = r - (beta_inew - beta[j])*X[:,j]
                beta[j] = beta_inew
                if beta[j] != 0:
                    support_set.add(j)
                    len_support += 1
                continue
                
            bst_imp, bst_i, bst_v = Check_Swaprow(y,X,beta,r,support_set,j,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
            if bst_imp < 0:
                r = r - bst_v*X[:,j] + beta[bst_i]*X[:,bst_i]
                beta[j] = bst_v
                beta[bst_i] = 0
                support_set.remove(bst_i)
                if bst_v != 0:
                    support_set.add(j)
                else:
                    len_support -= 1
                break
                    
        f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
        objs.append(f)
        if (abs(f_old - f) <= max(abs(f),abs(f_old),1)*ftol) and psi_cur_itr > 0:
            break
        f_old = f
        psi_cur_itr += 1
        
    sol_time = time()-st
    return beta, f, objs, r, psi_cur_itr, sol_time

def Active_IHTCDPSI(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=np.inf, 
                iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
                sea1_max_itr=5,sea2_max_itr=10,psi_max_itr=1):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    L = 1.05*(skl_svd(X)**2+lambda2*2) if L is None else L
    S_diag = np.linalg.norm(X, axis=0)**2
    r = y - X@beta
    f_old = np.inf
    act_cur_itr = 0
    psi_cur_itr = 0
    _sol_str = 'active_set objs, r, iht_cur_itr sol_time search_itr outliers'
    Solution = namedtuple('Solution', _sol_str)
    sols = []
    
    active_set = initial_active_set(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L,M,buget,kimp,act_itr)
    
    while psi_cur_itr < psi_max_itr:
        
        act_cur_itr = 0
        while act_cur_itr < act_max_itr:
        
            X_act = X[:,active_set]
            beta_act = beta[active_set]
            L_act = 1.05*(skl_svd(X_act)**2+lambda2*2) 
        
            beta_act, f, objs, r_act, iht_cur_itr, sol_time = Vanilla_IHTCD(y,X_act,beta_act,k,alpha[active_set],lambda1,lambda2,beta_tilde1[active_set],
                                                                     beta_tilde2[active_set], L_act,M,iht_max_itr,ftol,cd_itr,ctol,sea1_max_itr)
        
            L_init = 2*L_act
            beta = np.zeros(p)
            beta[active_set] = beta_act
            r = y - X@beta
            f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)
            active_set_set = set(active_set)
            search_flag = False
            search_cur_itr = 0
            outliers = set()
            beta_update,r_update = beta,r
            while search_cur_itr < sea2_max_itr:
                beta_tmp, r_tmp = hard_thresholding(y,X,beta,r,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,L_init,M)
                f_new = evaluate_obj(beta_tmp,r_tmp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
                outliers = set(np.where(beta_tmp)[0]) - active_set_set
                search_cur_itr += 1
                if len(outliers) >= 1 and f_new < f:
                    search_flag = True
                    beta_update = beta_tmp
                    r_update = r_tmp
                elif f_new >= f:
                    beta = beta_update 
                    r = r_update
                    break
                L_init /= 2
            
            sols.append(Solution(active_set=active_set,objs=objs,r=np.copy(r_act),iht_cur_itr=iht_cur_itr,sol_time=sol_time,
                             search_itr=search_cur_itr,outliers=len(outliers)))
            if not search_flag:
                break
            active_set = np.array(sorted(active_set_set | outliers))
            act_cur_itr += 1
            
        psi_cur_itr += 1
        j, bst_i, bst_v = Check_Swapall(y,X,beta,r,S_diag,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M)
        if j == -1:
            break
        else:
            r = r - bst_v*X[:,j] + beta[bst_i]*X[:,bst_i]
            beta[j] = bst_v
            beta[bst_i] = 0
            active_set_set = set(active_set)
            active_set = np.array(sorted(active_set_set | set([j])))
    
    
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    tot_time = time()-st
    return beta, f, sols, r, psi_cur_itr, tot_time

def Stochastic_BCD(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, M=np.inf, sto_max_itr=100,buget=None,kimp=1.5,batch=10,sto_m="cyc"):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    
    buget = p if buget is None else buget
    ksupp = int(np.max([np.min([kimp*k, buget, p]),k]))
    
    argsort = np.argsort(-np.abs(beta))
    active_set = argsort[:ksupp]
    

    gradB = list(range(0,ksupp,int(ksupp/batch)))
    gradB.append(ksupp)
    per_idx = np.random.permutation(range(ksupp))
    act_idx = active_set[per_idx]
    beta_act = beta[act_idx]
    X_act = X[:,act_idx]
    stepL = list()
    for i in range(len(gradB)-1):
        stepL.append(1.05*(skl_svd(X_act[:,gradB[i]:gradB[i+1]])**2+lambda2**2))
        
        
    beta_act, r,objs = sto_BCD(y,X_act,beta_act,alpha[act_idx],lambda1,lambda2,
                               beta_tilde1[act_idx],beta_tilde2[act_idx],gradB,stepL,M,sto_max_itr,sto_m)
    
    beta = np.zeros(p)
    beta[act_idx] = beta_act
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    sol_time = time()-st

    
    return beta, f, objs, r, sol_time


def Heuristic_CD(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, M=np.inf, cd_max_itr=100,buget=None,kimp=1.5,sto_m = "cyc",cd_tol=-1):
    
    st = time()
    p = beta.shape[0]
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    
    buget = p if buget is None else buget
    ksupp = int(np.max([np.min([kimp*k, buget, p]),k]))
    
    
    if np.linalg.norm(alpha) > 1e-8:
        beta, r = prune_ls(y,X,beta,y-X@beta,ksupp,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,10)
    else:
        r = y - X@beta
    active_set = np.argsort(-np.abs(beta))
    act_idx = active_set[:ksupp]
    beta_act = beta[act_idx]
    X_act = X[:,act_idx]
    S_diag = np.linalg.norm(X_act, axis=0)**2

    print('l2222 ', lambda2)
    
    beta,r = CD_loop(y,X_act,beta_act,r,S_diag,alpha[act_idx],lambda1,lambda2,
                     beta_tilde1[act_idx],beta_tilde2[act_idx],M,cd_max_itr,sto_m,cd_tol)
        
    
    beta = np.zeros(p)
    beta[act_idx] = beta_act
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    sol_time = time()-st

    
    return beta, f, r, sol_time

def compute_inverse(y, X, alpha, lambda2, beta_tilde2,act_idx=None):
    n, p = X.shape
    k = p if act_idx is None else len(act_idx)
    if n < k:
        if act_idx is None:
            solve_b = X.T@y + 2*lambda2*beta_tilde2 - alpha
            beta = solve_b / (2*lambda2) - \
            X.T@np.linalg.solve(np.eye(n)+X@(X.T)/(2*lambda2),X@solve_b)/(4*lambda2**2)
        else:
            solve_b = pmvm(X,y,act_idx,True) + 2*lambda2*beta_tilde2[act_idx] - alpha[act_idx]
            mmX = pmmm(X,act_idx)
            mvmX =pmvm(X,solve_b,act_idx,False)
            solve_tmp = (np.linalg.solve(np.eye(n)+mmX/(2*lambda2),mvmX)/(4*lambda2**2)).astype(X.dtype)
            beta = solve_b / (2*lambda2) - \
                pmvm(X,solve_tmp,act_idx,True)
    else:
        if act_idx is None:
            solve_b = X.T@y + 2*lambda2*beta_tilde2 - alpha
            beta = np.linalg.solve(2*lambda2*np.eye(k)+(X.T)@X,solve_b)
        else:
            solve_b = pmvm(X,y,act_idx,True) + 2*lambda2*beta_tilde2[act_idx] - alpha[act_idx]
            X_act = X[:,act_idx]
            beta = np.linalg.solve(2*lambda2*np.eye(k)+(X_act.T)@X_act,solve_b)
    return beta.astype(X.dtype)

def Heuristic_LS(y,X,beta,k,alpha,lambda1,lambda2, beta_tilde1, beta_tilde2, M=np.inf,
                use_prune = True):
    
    assert M == np.inf
    assert lambda1 == 0
    
    st = time()
    n, p = X.shape
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    if X.dtype == 'float32':
        lambda2 = np.float32(lambda2)
        lambda1 = np.float32(lambda1)
    if np.linalg.norm(alpha) > 1e-8 and use_prune:
        beta, r = prune_ls(y,X,beta,y-X@beta,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,10)
    active_set = np.argsort(-np.abs(beta))
    act_idx = active_set[:k]
    if p < 1e7:
        beta_act = beta[act_idx]
        X_act = X[:,act_idx]
        beta_act = compute_inverse(y, X_act, alpha[act_idx], lambda2, beta_tilde2[act_idx])
    else:
        beta_act = compute_inverse(y, X, alpha, lambda2, beta_tilde2,act_idx)
    
    beta = np.zeros(p,dtype=X.dtype)
    beta[act_idx] = beta_act
    r = y - X@beta
    f = evaluate_obj(beta,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    sol_time = time()-st
    
    return beta, f, r, sol_time

def Heuristic_LSBlock(w_bar,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, M=np.inf,
                use_prune = True, per_idx=None, num_block=1, block_list=None, split_type=0):
    
    assert M == np.inf
    assert lambda1 == 0
    
    st = time()
    n, p = X.shape
    alpha = np.zeros(p) if alpha is None else alpha
    beta_tilde1 = np.zeros(p) if beta_tilde1 is None else beta_tilde1
    beta_tilde2 = np.zeros(p) if beta_tilde2 is None else beta_tilde2
    per_idx = np.arange(p) if per_idx is None else per_idx
    if block_list is None:
        block_list = list(range(0,p+1,int(p/num_block))) 
        block_list[-1] = p

    if X.dtype == 'float32':
        lambda2 = np.float32(lambda2)
        lambda1 = np.float32(lambda1)
    
    y = X@w_bar
    X_per = X
    w_barper = w_bar
    beta_per = beta
    alpha_per = alpha
    beta_tilde1per = beta_tilde1
    beta_tilde2per = beta_tilde2
    
    beta_new = np.zeros(p,dtype=X.dtype)
    ksum = 0
    if split_type == 0:
        for ib in range(len(block_list)-1):
            idx_cur = np.arange(block_list[ib],block_list[ib+1])
            kcur = int(np.floor((block_list[ib+1]-block_list[ib])*k/p))
            ksum += kcur
            beta_new[per_idx[idx_cur]], _, _, _ = Heuristic_LS(X_per[:,idx_cur]@w_barper[idx_cur],X_per[:,idx_cur],beta_per[idx_cur],kcur,alpha_per[idx_cur],
                lambda1,lambda2,beta_tilde1per[idx_cur],beta_tilde2per[idx_cur],M,use_prune)
    else:
        if np.linalg.norm(alpha) > 1e-8 and use_prune:
            beta, r = prune_ls(y,X,beta,y-X@beta,k,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2,M,10)
            beta_per = beta[per_idx]
    
        active_set = np.argsort(-np.abs(beta))
        thres = np.abs(beta[active_set[k]])
        
        for ib in range(len(block_list)-1):
            idx_cur = np.arange(block_list[ib],block_list[ib+1])
            kcur = np.sum(np.abs(beta_per[idx_cur]) > thres)
            ksum += kcur
            if kcur == 0:
                continue
            beta_new[per_idx[idx_cur]], _, _, _ = Heuristic_LS(X_per[:,idx_cur]@w_barper[idx_cur],X_per[:,idx_cur],beta_per[idx_cur],kcur,alpha_per[idx_cur],
                lambda1,lambda2,beta_tilde1per[idx_cur],beta_tilde2per[idx_cur],M,use_prune)
    
    sol_time = time() - st
    r = y-X@beta_new
    f = evaluate_obj(beta_new,r,alpha,lambda1,lambda2,beta_tilde1,beta_tilde2)    
    
    print("Total non-zero:",ksum)
    return beta_new, f, r, sol_time

def Vanilla_IHTCD_PP(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0.,beta_tilde1=None,beta_tilde2=None,L=None,M=np.inf,iht_max_itr=100,ftol=1e-8,
                  cd_itr=0,ctol=1e-4,search_max_itr=1):
    
    nnz_idx = np.where(np.linalg.norm(X, axis=0)**2)[0]
    beta_new = np.zeros_like(beta)
    
    beta, f, objs, r, iht_cur_itr, sol_time = Vanilla_IHTCD(y,X[:,nnz_idx],beta[nnz_idx],k,alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,
                beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], L=L, M=M, iht_max_itr=iht_max_itr, ftol=ftol,       
                cd_itr=cd_itr,ctol=ctol,search_max_itr=search_max_itr)
    
    beta_new[nnz_idx] = np.copy(beta)
    
    return beta_new, f, objs, r, iht_cur_itr, sol_time

def Active_IHTCD_PP(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=np.inf, 
                iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
                sea1_max_itr=5, sea2_max_itr=10):
    
    
    nnz_idx = np.where(np.linalg.norm(X, axis=0)**2)[0]
    beta_new = np.zeros_like(beta)
    
    beta, f, sols, r, act_cur_itr, tot_time = Active_IHTCD(y,X[:,nnz_idx],beta[nnz_idx],k,alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,
                beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], L=L, M=M, iht_max_itr=iht_max_itr, ftol=ftol,       
                act_max_itr=act_max_itr,buget=buget,kimp=kimp,act_itr=act_itr,cd_itr=cd_itr,ctol=ctol,
                sea1_max_itr=sea1_max_itr, sea2_max_itr=sea2_max_itr)
    
    beta_new[nnz_idx] = np.copy(beta)
    
    return beta_new, f, sols, r, act_cur_itr, tot_time

def Active_IHTCDLS_PP(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, L=None, M=np.inf, iht_max_itr=100, ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4,
  sea1_max_itr=5, sea2_max_itr=10):
    
    
    nnz_idx = np.where(np.linalg.norm(X, axis=0)**2)[0]
    beta_new = np.zeros_like(beta)
    
    if len(nnz_idx) > k:
        beta, f, sols, r, act_cur_itr, tot_time = Active_IHTCDLS(y,X[:,nnz_idx],beta[nnz_idx],k,alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,
                    beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], L=L, M=M, iht_max_itr=iht_max_itr, ftol=ftol,       
                    act_max_itr=act_max_itr,buget=buget,kimp=kimp,act_itr=act_itr,cd_itr=cd_itr,ctol=ctol,
                    sea1_max_itr=sea1_max_itr, sea2_max_itr=sea2_max_itr)
        
        beta_new[nnz_idx] = np.copy(beta)
    else:
        beta_new[nnz_idx] = np.copy(beta[nnz_idx])
        f = 0
        r = np.zeros_like(y)
        sols = None
        act_cur_itr=0
        tot_time=0
    
    return beta_new, f, sols, r, act_cur_itr, tot_time

def Heuristic_CD_PP(y,X,beta,k,alpha=None,lambda1=0.,lambda2=0., beta_tilde1=None, beta_tilde2=None, M=np.inf, cd_max_itr=100,buget=None,kimp=1.5,sto_m="cyc",cd_tol=-1):
    
    
    nnz_idx = np.where(np.linalg.norm(X, axis=0)**2)[0]
    beta_new = np.zeros_like(beta)

    if len(nnz_idx) > k:
    
        beta, f, r, sol_time = Heuristic_CD(y,X[:,nnz_idx],beta[nnz_idx],k,alpha=alpha[nnz_idx],lambda1=lambda1,lambda2=lambda2,
                    beta_tilde1=beta_tilde1[nnz_idx], beta_tilde2=beta_tilde2[nnz_idx], M=M, cd_max_itr=cd_max_itr, buget=buget,kimp=kimp,sto_m=sto_m,cd_tol=cd_tol)
        
        beta_new[nnz_idx] = np.copy(beta)
        print(beta)
    else:
        beta_new[nnz_idx] = np.copy(beta[nnz_idx])
        f = 0
        r = np.zeros_like(y)
        sol_time=0
    
    return beta_new, f, r, sol_time