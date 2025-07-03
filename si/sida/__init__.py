import numpy as np
from .. import util
from . import ot

def fit(ns, nt, X, y, a, b, c_, H, B, Bc):
    Theta = ot.construct_Theta(ns, nt)    
    Theta_a = Theta.dot(a)
    Theta_b = Theta.dot(b)

    p_tilde = c_ + Theta_a * Theta_a
    q_tilde = 2 * Theta_a * Theta_b
    r_tilde = Theta_b * Theta_b

    H_B_invH_Bc = np.linalg.inv(H[:, B]).dot(H[:, Bc])

    p = (p_tilde[Bc, :].T - p_tilde[B, :].T.dot(H_B_invH_Bc)).T
    q = (q_tilde[Bc, :].T - q_tilde[B, :].T.dot(H_B_invH_Bc)).T
    r = (r_tilde[Bc, :].T - r_tilde[B, :].T.dot(H_B_invH_Bc)).T

    flag = False
    list_intervals = []

    for i in range(p.shape[0]):
        fa = - r[i][0]
        sa = - q[i][0]
        ta = - p[i][0]

        temp = util.solve_quadratic_inequality(fa, sa, ta)
        
        if flag == False:
            flag = True
            list_intervals = temp
        else:
            list_intervals = util.intersect(list_intervals, temp)

    return list_intervals