import numpy as np
from scipy.optimize import linprog

def solve(c, A_eq, b_eq, method):
    n = c.shape[0]
    res = linprog(c, A_ub = - np.identity(n), b_ub = np.zeros((n, 1)), A_eq = A_eq, b_eq = b_eq, method = method, options = {'maxiter': 100000})
    B = res.basis
    Bc = list(set(range(n))-set(B))
    return res.x, B, Bc