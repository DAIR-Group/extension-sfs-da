import numpy as np
from .. import util

def check_KKT(eps, u, v, A, delta, B1, B2):
    # Check KKT conditions
    # print("Checking KKT conditions...")
    sta = A @ eps + delta + B1.T @ u + B2.T @ v
    prec = 1e-8
    if np.any((sta < -prec) | (sta > prec)):
        print(sta[np.where((sta < -prec) | (sta > prec))[0],:])
        raise ValueError("Stationarity Condition Failed!")

    B1eps = B1 @ eps
    B2eps = B2 @ eps
    uB1eps = u * B1eps

    if np.any((uB1eps < -prec) | (uB1eps > prec)):
        print(uB1eps[np.where((uB1eps < -prec) | (uB1eps > prec))[0],:])
        raise ValueError("Complementary Slackness Failed!")

    if not np.all(B1eps <= prec):
        print(B1eps[np.where(B1eps > -prec)[0],:])
        raise ValueError("Primal Feasibility Failed!")

    if np.any((B2eps < -prec) | (B2eps > prec)):
        print(B2eps[np.where((B2eps < -prec) | (B2eps > prec))[0],:])
        raise ValueError("Primal Feasibility Failed!")

    if not np.all(u >= -prec):
        print(u[np.where(u < -prec)[0],:])
        raise ValueError("Dual Feasibility Failed!")
    # print("Finished checking KKT conditions.")

def fit(X, y, a, b, Lambda, u, v, A, B1, B2):
    p = y.shape[0]
    m = p-1
    # y = a + bz
    # [-delta]T
    # [   0  ]   =  g0 + g1z
    g0 = -np.vstack((-X.T @ a, Lambda * np.ones((2*m, 1))))
    g1 = -np.vstack((-X.T @ b, np.zeros((2*m, 1))))
    I = np.where(u > 0)[0].tolist()
    Ic = [i for i in range(len(u)) if i not in I]
    B1I = np.copy(B1[I, :])
    B1Ic = np.copy(B1[Ic, :])
    mat1 = np.vstack((np.hstack((B1Ic, np.zeros((len(Ic), len(I))))),       
                      np.hstack((np.zeros((len(I), p+2*m)), -np.identity(len(I))))))

    mat2 = np.vstack((np.hstack((A, B1I.T, B2.T)), np.hstack((B1I, np.zeros((len(I), len(I))), np.zeros((len(I), m)))), np.hstack((B2, np.zeros((m, len(I)+m))))))
    mat2 = np.linalg.inv(mat2)
    red = np.hstack((np.eye(p+2*m+len(I)), np.zeros((p+2*m+len(I), m))))

    vec1 = np.vstack((g0, np.zeros((len(I)+m, 1))))
    vec2 = np.vstack((g1, np.zeros((len(I)+m, 1))))

    temp = mat1 @ red @ mat2
    p = temp @ vec1
    q = temp @ vec2

    # Solve the inequalities: p + qz <= 0
    return util.solve_linear_inequalities(p, q)