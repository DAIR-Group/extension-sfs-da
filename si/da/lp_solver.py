import numpy as np
from scipy.optimize import linprog
import ot

def solve(ns, nt, c, A_eq, b_eq):
    row_mass = np.ones(ns) / ns
    col_mass = np.ones(nt) / nt
    T = ot.emd(a=row_mass, b=col_mass, M=c.reshape(ns, nt))
    B = np.where(T.reshape(-1) != 0)[0]

    if B.shape[0] != ns+nt-1:
        n = c.shape[0]
        res = linprog(c, A_ub = -np.identity(n), b_ub = np.zeros((n, 1)), A_eq = A_eq, b_eq = b_eq,
                      method = 'simplex', options = {'maxiter': 100000})
        T = res.x.reshape(ns, nt)
        B = res.basis

    B = B.tolist()
    return T, B