from .da import OTDA
from .cv import HoldOutCV
from .qp import VanillaLasso, ElasticNet, NNLS, FusedLasso
from .utils import intersect
import numpy as np

def divide_and_conquer(a, b, regr_ins, da_ins, zmin, zmax, unit, cp_mat):
    regr_class = type(regr_ins)
    hyperparams = regr_ins.get_hyperparams()
    da_class = type(da_ins)
    ns, nt = da_ins.ns, da_ins.nt
    Xs, Xt = da_ins.Xs, da_ins.Xt
    X = np.vstack((Xs, Xt))
    
    list_intervals = []
    list_M = []
    zuv = zmin
    while zuv < zmax:
        y_zuv = a+b*zuv
        ys, yt = y_zuv[:ns,:], y_zuv[ns:,:]
        da_model = da_class(np.hstack((Xs, ys)), np.hstack((Xt, yt)))
        Tu, _ = da_model.fit()
        interval_da = da_model.si(a, b)

        # Select the interval containing the data point that we are currently considering.
        for i in interval_da:
            if i[0] <= zuv <= i[1]:
                interval_da = [i]
                break

        Omega_u = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * Tu, np.identity(nt)))))
        a_tilde, b_tilde = Omega_u @ a, Omega_u @ b 
        if cp_mat is not None:
            a_tilde, b_tilde = cp_mat @ a_tilde, cp_mat @ b_tilde  
        else:
            X_tilde = Omega_u @ X

        while zuv < interval_da[0][1]:
            y_zuv = a+b*zuv
            y_tilde_u_zuv = Omega_u.dot(y_zuv)
            
            if cp_mat is not None:
                y_tilde_u_zuv = cp_mat @ y_tilde_u_zuv  
                regr_model = regr_class(y_tilde_u_zuv, **hyperparams)
            else:
                regr_model = regr_class(X_tilde, y_tilde_u_zuv, **hyperparams)
            M_v = regr_model.fit()
            if unit is not None:
                M_v = list(dict.fromkeys(i // (unit+1) for i in M_v[1:-1]))
                M_v = [0] + M_v + [nt-1]
            if regr_model.is_empty():
                zuv += 5e-4
                continue
         
            interval_regr = regr_model.si(a_tilde, b_tilde)            
            interval_uv = intersect(interval_da, interval_regr)
            # with open("./debug.txt", "a") as f:
            #     f.write(f'{interval_uv}\t\t{zuv}\t\t{interval_da}\t\t{interval_regr}\t\t{M_v}\n')
            list_intervals += interval_uv
            list_M += [M_v]
            zuv = interval_uv[0][1] + 5e-5
    return list_intervals, list_M

def is_continuous_sublist(a, b):
    a = np.array(a)
    b = np.array(b)
    if len(a) == 0:
        return True
    if len(a) > len(b):
        return False

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(b, len(a))
    return np.any(np.all(windows == a, axis=1))

def fit(a, b, regr_ins, da_ins, zmin=-20, zmax=20, min_condition=None, unit=None, cp_mat=None):
    list_intervals, list_M = divide_and_conquer(a, b, regr_ins, da_ins, zmin, zmax, unit, cp_mat)
    Z = []
    if unit is not None:
        # M_obs = list(dict.fromkeys(i // (unit+1) for i in regr_ins.active_set[1:-1]))
        # M_obs = [0] + M_obs + [da_ins.nt-1]
        for i in range(len(list_intervals)):
            if is_continuous_sublist(min_condition, list_M[i]):
                Z.append(list_intervals[i])
        return Z
    else:
        M_obs = regr_ins.active_set
    for i in range(len(list_intervals)):
        if np.array_equal(list_M[i], M_obs):
            Z.append(list_intervals[i])
    return Z