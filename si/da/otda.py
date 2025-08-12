from scipy.cluster.hierarchy import DisjointSet
import ot
import numpy as np
from ..utils import solve_quadratic_inequality, intersect

def construct_Theta(ns, nt):
    return np.hstack((np.kron(np.identity(ns), np.ones((nt, 1))), np.kron(- np.ones((ns, 1)), np.identity(nt))))

def construct_cost(Xs, ys, Xt, yt):
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

def construct_B(T, u, v, c, tol=1e-6):
    ns, nt = T.shape
    DJ = DisjointSet(range(ns + nt))
    B = []

    # Vectorized first loop - process elements where T > tol
    large_T_indices = np.where(T > tol)
    for i, j in zip(large_T_indices[0], large_T_indices[1]):
        DJ.merge(i, j + ns)
        B.append(i * nt + j)
    
    # Early exit if we already have enough elements
    if len(B) >= ns + nt - 1:
        return sorted(B[:ns + nt - 1])
    
    # Vectorized computation of reduced costs
    rc = c - u[:, np.newaxis] - v[np.newaxis, :]
    
    # Find candidates where |rc| < tol
    candidate_indices = np.where(np.abs(rc) < tol)
    
    # Process candidates in order
    for i, j in zip(candidate_indices[0], candidate_indices[1]):
        if len(B) >= ns + nt - 1:
            break
        if not DJ.connected(i, j + ns):
            DJ.merge(i, j + ns)
            B.append(i * nt + j)
    
    return sorted(B)

class OTDA():
    """
    Optimal Transport-based Domain Adaptation (OTDA).
    """
    def __init__(self, Ds, Dt):
        self.Xs = Ds[:,:-1]
        self.Xt = Dt[:,:-1]
        self.ys = Ds[:,-1:]
        self.yt = Dt[:,-1:]

        self.ns, self.nt = self.Xs.shape[0], self.Xt.shape[0]
        self.c_, self.c = construct_cost(self.Xs, self.ys, self.Xt, self.yt)
        self.H = construct_H(self.ns, self.nt)
        self.h = construct_h(self.ns, self.nt)
    
    def fit(self):
        row_mass = np.ones(self.ns) / self.ns
        col_mass = np.ones(self.nt) / self.nt
        T, log = ot.emd(a=row_mass, b=col_mass, M=self.c.reshape(self.ns, self.nt), log=True)
        B = np.where(T.reshape(-1) != 0)[0].tolist()

        if len(B) != self.ns+self.nt-1:
            B = construct_B(T, log['u'], log['v'], self.c.reshape(self.ns, self.nt))

        self.T = T
        self.B = B
        
        self.v = - np.linalg.inv(self.H[:,self.B].T) @ self.c[self.B,:]
        self.u = self.c + self.H.T @ self.v
        return self.T, self.B

    def check_KKT(self):
        t = self.T.reshape(-1, 1)
        sta = self.c - self.u + self.H.T @ self.v
        prec = 1e-8
        if np.any((sta < -prec) | (sta > prec)):
            print(sta[np.where((sta < -prec) | (sta > prec))[0],:])
            raise ValueError("Stationarity Condition Failed!")

        ut = self.u * t
        if np.any((ut < -prec) | (ut > prec)):
            print(ut[np.where((ut < -prec) | (ut > prec))[0],:])
            raise ValueError("Complementary Slackness Failed!")

        if not np.all(t >= -prec):
            print(t[np.where(t < -prec)[0],:])
            raise ValueError("Primal Feasibility Failed!")

        Ht_h = self.H @ t - self.h
        if np.any((Ht_h < -prec) | (Ht_h > prec)):
            print(Ht_h[np.where((Ht_h < -prec) | (Ht_h > prec))[0],:])
            raise ValueError("Primal Feasibility Failed!")

        if not np.all(self.u >= -prec):
            print(self.u[np.where(self.u < -prec)[0],:])
            raise ValueError("Dual Feasibility Failed!")

    def si(self, a, b):
        Bc = list(set(range(self.ns*self.nt))-set(self.B))

        Theta = construct_Theta(self.ns, self.nt)
        Theta_a = Theta.dot(a)
        Theta_b = Theta.dot(b)

        p_tilde = self.c_ + Theta_a * Theta_a
        q_tilde = 2 * Theta_a * Theta_b
        r_tilde = Theta_b * Theta_b

        HB_invHBc = np.linalg.inv(self.H[:, self.B]).dot(self.H[:, Bc])

        p = (p_tilde[Bc, :].T - p_tilde[self.B, :].T.dot(HB_invHBc)).T
        q = (q_tilde[Bc, :].T - q_tilde[self.B, :].T.dot(HB_invHBc)).T
        r = (r_tilde[Bc, :].T - r_tilde[self.B, :].T.dot(HB_invHBc)).T

        flag = False
        list_intervals = []

        for i in range(p.shape[0]):
            fa = - r[i][0]
            sa = - q[i][0]
            ta = - p[i][0]

            temp = solve_quadratic_inequality(fa, sa, ta)
            if flag == False:
                flag = True
                list_intervals = temp
            else:
                list_intervals = intersect(list_intervals, temp)
        return list_intervals