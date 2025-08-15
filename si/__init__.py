from .da import OTDA
from .cv import HoldOutCV
from .qp import VanillaLasso, ElasticNet, NNLS, FusedLasso
from .utils import intersect
import numpy as np

def divide_and_conquer(a, b, regr_ins, cv_ins, da_ins, zmin, zmax, unit, cp_mat):
    if cv_ins is None:
        regr_class = type(regr_ins)
        hyperparams = regr_ins.get_hyperparams()
        da_class = type(da_ins)
        ns, nt = da_ins.ns, da_ins.nt
        Xs, Xt = da_ins.Xs, da_ins.Xt
        X = np.vstack((Xs, Xt))
        
        list_intervals = []
        list_M = []
        z = zmin
        while z < zmax:
            yz = a + b * z
            ys, yt = yz[0:ns, :], yz[ns:, :]
            da_model = da_class(np.hstack((Xs, ys)), np.hstack((Xt, yt)))
            Tz, _ = da_model.fit()
            # da_model.check_KKT()
            interval_da = da_model.si(a, b)

            # Select the interval containing the data point that we are currently considering.
            for i in interval_da:
                if i[0] <= z <= i[1]:
                    interval_da = [i]
                    break

            Omega_z = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * Tz, np.identity(nt)))))
            a_tilde, b_tilde = Omega_z @ a, Omega_z @ b 
            
            if cp_mat is not None:
                a_tilde, b_tilde = cp_mat @ a_tilde, cp_mat @ b_tilde  
            else:
                Xz_tilde = Omega_z @ X
            
            while z < interval_da[0][1]:
                yz = a + b * z
                yz_tilde = Omega_z @ yz
                
                if cp_mat is not None:
                    yz_tilde = cp_mat @ yz_tilde  
                    regr_model = regr_class(yz_tilde, **hyperparams)
                else:
                    regr_model = regr_class(Xz_tilde, yz_tilde, **hyperparams)
                M_v = regr_model.fit()
                # regr_model.check_KKT()
                
                if unit is not None:
                    M_v = list(dict.fromkeys(i // (unit+1) for i in M_v[1:-1]))
                    M_v = [0] + M_v + [nt-1]
                
                if regr_model.is_empty():
                    z += 1e-4
                    continue
            
                interval_regr = regr_model.si(a_tilde, b_tilde)            
                interval_z = intersect(interval_da, interval_regr)
                # with open("./debug.txt", "a") as f:
                    # f.write(f'{interval_uv}\t\t{zuv}\t\t{interval_da}\t\t{interval_regr}\t\t{M_v}\n')
                list_intervals += interval_z
                list_M += [M_v]
                z = interval_z[0][1] + 1e-5
        
        return list_intervals, list_M
    else:
        regr_class = type(regr_ins)
        hyperparams = regr_ins.get_hyperparams()
        cv_class = type(cv_ins)
        da_class = type(da_ins)
        ns, nt = da_ins.ns, da_ins.nt
        Xs, Xt = da_ins.Xs, da_ins.Xt
        X = np.vstack((Xs, Xt))
        
        list_intervals = []
        list_M = []
        z = zmin

        while z < zmax:
            yz = a + b * z
            ys, yt = yz[0:ns, :], yz[ns:, :]
            da_model = da_class(np.hstack((Xs, ys)), np.hstack((Xt, yt)))
            Tz, _ = da_model.fit()
            # da_model.check_KKT()
            interval_da = da_model.si(a, b)

            # Select the interval containing the data point that we are currently considering.
            for i in interval_da:
                if i[0] <= z <= i[1]:
                    interval_da = [i]
                    break

            Omega_z = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * Tz, np.identity(nt)))))
            a_tilde, b_tilde = Omega_z @ a, Omega_z @ b 
            Xz_tilde = Omega_z @ X

            while z < interval_da[0][1]:
                yz = a + b * z
                yz_tilde = Omega_z @ yz
                
                cv_model = cv_class(train_indices=cv_ins.train_indices, val_indices=cv_ins.val_indices)
                best_Lambda, _ = cv_model.fit(Xz_tilde, yz_tilde, regr_class, cv_ins.list_lambda)
                interval_cv = cv_model.si(a_tilde, b_tilde)

                for i in interval_cv:
                    if i[0] <= z <= i[1]:
                        interval_cv = [i]
                        break

                interval_cv = intersect(interval_da, interval_cv)
                hyperparams['Lambda'] = best_Lambda

                while z < interval_cv[0][1]:
                    yz = a + b * z
                    yz_tilde = Omega_z @ yz

                    regr_model = regr_class(Xz_tilde, yz_tilde, **hyperparams)
                    M_v = regr_model.fit()
                    # regr_model.check_KKT()
                    
                    if regr_model.is_empty():   
                        z += 1e-4
                        continue
                
                    interval_regr = regr_model.si(a_tilde, b_tilde)
                    interval_z = intersect(interval_cv, interval_regr)
                    list_intervals += interval_z
                    list_M += [M_v]
                    z = interval_z[0][1] + 1e-5
                    # with open("./debug.txt", "a") as f:
                        # f.write(f'{z}\t\t{interval_z}\n')
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

def fit(a, b, regr_ins, cv_ins=None, da_ins=None, zmin=-20, zmax=20, 
        min_condition=None, unit=None, cp_mat=None):
    
    list_intervals, list_M = divide_and_conquer(a, b, regr_ins, cv_ins, da_ins, zmin, zmax, unit, cp_mat)
    
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