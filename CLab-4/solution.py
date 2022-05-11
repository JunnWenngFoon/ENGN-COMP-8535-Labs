import numpy as np
import cvxpy as cp

########################################################
## Complete functions in skeleton codes below
## following instructions in each function.
## Do not modify existing function name or inputs.
## Do not test your codes here - use main.py instead.
## You may use any built-in functions from NumPy.
## You may define and call new functions as you see fit.
########################################################


def hard_sparsity_bf(A, y, s):
    '''
    inputs: 
      - A: m-by-n matrix
      - y: m-dimensional vector
      - s: integer in range (0,m)
    returns:
      x: n-dimensional vector that minimises ||y-Ax||_2 subject to s-sparsity
    '''
    m,n = A.shape
    bf=combs_idx(n, s)
    store = None
    result = None
    for mask in bf:
        x,r, _, _ = np.linalg.lstsq(A[:,mask],y,rcond=None)
        if store==None or r<store:
            store=r
            result=np.zeros(n)
            result[mask]=x
    return result

def hard_sparsity_omp(A, y, s):
    '''
    inputs: 
      - A: m-by-n matrix
      - y: m-dimensional vector
      - s: integer in range (0,m)
    returns:
      x: n-dimensional vector that minimises ||y-Ax||_2 subject to s-sparsity
    ref: https://angms.science/doc/RM/OMP.pdf
    '''
    m,n = A.shape
    x =0
    r =y
    sig= set()
    for i in range(s):
        b=A.T@r
        sig.add(np.argmax(np.abs(b)))
        x,_,_,_ = np.linalg.lstsq(A[:,list(sig)],y,rcond=None)
        r = y-A[:,list(sig)]@x
    result=np.zeros(n)
    result[list(sig)]=x
    return result

def hard_equality_lp(A, y):
    '''
    inputs: 
      - A: m-by-n matrix
      - y: m-dimensional vector
    returns:
      x: n-dimensional vector that is as sparse as possible subject to y=Ax
         sparsity is approximated by minimising L1 norm ||x||_1 instead
    '''
    # you can use cvxpy to solve the linear programming
    m,n=A.shape
    var=cp.Variable(n)
    lp=cp.Problem(cp.Minimize(cp.norm(var,1)),[A@var==y])
    lp.solve()
    result=var.value
    return result



### you can optionally write your own functions like below ###

def combs_idx(n, k):
    '''
    returns an index array of n chooses k.
    arguments:
        -n: integer
        -k: integer
    returns:
        np array of shape (C,k)
        where C is number of combinations.
        each row has k integers in [0,n), representing indices of a k-combination.
    '''
    assert n>=k
    combs = []
    comb = np.arange(k)
    while comb[-1] < n: 
        combs.append(comb.copy())
        for i in range(1,k+1):
            if comb[-i] != n-i:
                break # find last occurance of non-maximum elem
        # reset this last part in increasing order
        comb[-i:] = np.arange(1+comb[-i], i+1+comb[-i])
    return np.array(combs)