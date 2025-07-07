import numpy as np
from . import da
from . import qp
from . import util

def divide_and_conquer(ns, nt, trans_mat, a, b, Lambda, zmin, zmax, method):
    n = ns+nt
    list_intervals = []
    list_M = []
    zuv = zmin
    while zuv < zmax:
        y_zuv = a+b*zuv
        ys, yt = y_zuv[:ns,:], y_zuv[ns:,:]
        c_, cost = da.ot.construct_cost(ys, yt)
        H = da.ot.construct_H(ns, nt)
        h = da.ot.construct_h(ns, nt)
        Tu, Bu, Bcu = da.ot.fit(ys, yt, cost, H, h, method)
        Omega_u = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * Tu, np.identity(nt)))))
        interval_u = da.fit(ns, nt, np.eye(n), y_zuv, a, b, c_, H, Bu, Bcu)

        # Select the interval containing the data point that we are currently considering.
        for i in interval_u:
            if i[0] <= zuv <= i[1]:
                interval_u = [i]
                break
        a_tilde, b_tilde = Omega_u @ a, Omega_u @ b 
        # Xtilde_u = Omega_u.dot(X)

        while zuv < interval_u[0][1]:
            y_zuv = a+b*zuv
            y_tilde_u_zuv = Omega_u.dot(y_zuv)
            sorted_y_tilde_u_zuv = trans_mat @ y_tilde_u_zuv    
            D, A, delta, B1, B2 = qp.fused_lasso.util(np.eye(n), sorted_y_tilde_u_zuv, Lambda)
            eps, u, v = qp.fused_lasso.fit(A, delta, B1, B2)
            beta = eps[0:n]
            M_v = qp.fused_lasso.find_change_points(beta, D)
            M_v = [0] + M_v + [n-1]
            interval_v = qp.fit(np.eye(n), sorted_y_tilde_u_zuv, trans_mat @ a_tilde, trans_mat @ b_tilde, Lambda, u, v, A, B1, B2)            
            interval_uv = util.intersect(interval_u, interval_v)
            with open("./debug.txt", "a") as f:
                f.write(f'{interval_uv}\t\t{zuv}\t\t{M_v}\n')
            list_intervals += interval_uv
            list_M += [M_v]
            zuv = interval_uv[0][1] + 1e-4

    return list_intervals, list_M

def fit(etaj, ns, nt, ys, yt, Sigma_s, Sigma_t, trans_mat, Lambda, M_obs, zmin=-20, zmax=20, method='highs'):
    y = np.vstack((ys, yt))
    a, b = util.compute_a_b(y, etaj)
    list_intervals, list_M = divide_and_conquer(ys.shape[0], yt.shape[0], trans_mat, a, b, Lambda, zmin, zmax, method)

    Z = []
    for i in range(len(list_intervals)):
        if np.array_equal(list_M[i], M_obs):
            Z.append(list_intervals[i])
        
    Sigma = np.vstack((np.hstack((Sigma_s , np.zeros((ns, nt)))),
                        np.hstack((np.zeros((nt, ns)), Sigma_t))))
    etajTy = np.dot(etaj.T, y)[0][0]
    etajTSigmaetaj = (etaj.T @ Sigma @ etaj)[0][0]
    tn_sigma = np.sqrt(etajTSigmaetaj)

    p_value = util.p_value(list_intervals, etajTy, tn_sigma)
    return p_value