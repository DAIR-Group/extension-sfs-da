import numpy as np
from . import lp_solver

def construct_Theta(ns, nt):
    return np.hstack((np.kron(np.identity(ns), np.ones((nt, 1))), np.kron(- np.ones((ns, 1)), np.identity(nt))))

def construct_cost(Ds, Dt):
    Xs = Ds[:,:-1]
    Xt = Dt[:,:-1]
    ys = Ds[:,-1:]
    yt = Dt[:,-1:]

    Xs_squared = np.sum(Xs**2, axis=1, keepdims=True)  # shape (n_s, 1)
    Xt_squared = np.sum(Xt**2, axis=1, keepdims=True).T  # shape (1, n_t)
    cross_term = Xs @ Xt.T  # shape (n_s, n_t)

    c_ = Xs_squared - 2 * cross_term + Xt_squared

    ys_squared = np.sum(ys**2, axis=1, keepdims=True)  # shape (n_s, 1)
    yt_squared = np.sum(yt**2, axis=1, keepdims=True).T  # shape (1, n_t)
    cross_term = ys @ yt.T  # shape (n_s, n_t)

    c__ = ys_squared - 2 * cross_term + yt_squared
    c = c_ + c__
    return c_.reshape(-1,1), c.reshape(-1,1)

def construct_H(ns, nt):
    Hr = np.zeros((ns, ns * nt))
    
    for i in range(ns):
        Hr[i:i+1, i*nt:(i+1)*nt] = np.ones((1, nt))
        
    Hc = np.identity(nt)
    for _ in range(ns - 1):
        Hc = np.hstack((Hc, np.identity(nt)))

    H = np.vstack((Hr, Hc))
    H = H[:-1,:]
    return H

def construct_h(ns, nt):
    h = np.vstack((np.ones((ns, 1)) / ns, np.ones((nt, 1)) / nt))
    h = h[:-1,:]
    return h

def fit(Ds, Dt, cost, H, h, method='highs'):
    ns, nt = Ds.shape[0], Dt.shape[0]
    T, B, Bc = lp_solver.solve(cost, H, h, method)
    T = T.reshape(ns, nt)
    return T, B, Bc