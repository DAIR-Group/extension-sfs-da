import numpy as np
from scipy.optimize import linprog

def solve(c, A_eq, b_eq):
    n = c.shape[0]
    res = linprog(c, A_ub = - np.identity(n), b_ub = np.zeros((n, 1)), A_eq = A_eq, b_eq = b_eq, method = 'simplex', options = {'maxiter': 20000})

    B = res.basis
    Bc = np.arange(n)

    for i in B:
        Bc = np.delete(Bc, np.where(Bc == i)[0][0])

    return res.x, B, Bc