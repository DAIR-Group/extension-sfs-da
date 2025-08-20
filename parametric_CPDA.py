import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import si
from si import utils, OTDA, FusedLasso
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm
import scipy.stats
import json
import re
import time
from multiprocessing import Pool

def get_next_id(base_dir="exp"):
    """
    Determine the next available experiment ID by checking existing directories.
    """
    if not os.path.exists(base_dir):
        return 1

    existing = [
        int(match.group(1)) for name in os.listdir(base_dir)
        if (match := re.match(r"exp_(\d+)", name)) and os.path.isdir(os.path.join(base_dir, name))
    ]
    return max(existing, default=0) + 1

def create_experiment_folder(base_dir="exp", config_data=None):
    """
    Automatically create the next experiment folder with custom config.

    Parameters:
        base_dir (str): The base directory for experiments.
        config_data (dict): JSON serializable data for config.json.
    """
    os.makedirs(base_dir, exist_ok=True)
    exp_id = get_next_id(base_dir)
    exp_dir = os.path.join(base_dir, f"exp_{exp_id}")
    os.makedirs(exp_dir, exist_ok=True)

    # Write config.json
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data or {}, f, indent=4)

    print(f"Created experiment folder: {exp_dir}")
    return exp_dir

nt = 10
unit = 10
Lambda = 10
delta = 0
model_name = "OT-FusedLasso"

def run(args):
    k = args[0] 
    folder_path = args[1]
    try:
        # Generate target data
        np.random.seed(k)
        nt = 10
        unit = 10
        ns = (nt-1) * unit 
        
        list_change_points_t = [1, 3, 5, 7, 9]
        delta_t = delta
        yt, mu_t, Sigma_t = FusedLasso.gen_data(nt, delta_t, list_change_points_t)
        
        list_change_points_s = [i*unit for i in list_change_points_t]
        delta_s = 2
        ys, mu_s, Sigma_s = FusedLasso.gen_data(ns, delta_s, list_change_points_s)

        y = np.vstack((ys, yt))
        mu = np.vstack((mu_s, mu_t))
        Sigma = np.vstack((np.hstack((Sigma_s , np.zeros((ns, nt)))),
                            np.hstack((np.zeros((nt, ns)), Sigma_t))))

        da_model = OTDA(ys, yt)
        T, _ = da_model.fit()
        # da_model.check_KKT()

        # Adapt the data
        Omega = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt)))))
        y_tilde = Omega @ y

        n = ns+nt
        trans_mat = np.zeros((n, n))

        for i in range(nt):
            special_row_idx = i * (unit + 1)
            trans_mat[special_row_idx, ns + i] = 1
            if i < nt - 1:
                row_start = special_row_idx + 1
                row_end = row_start + unit
                col_start = i * unit
                col_end = col_start + unit
                trans_mat[row_start:row_end, col_start:col_end] = np.eye(unit)

        sorted_y = trans_mat @ y_tilde

        hyperparams = {'Lambda': Lambda}
        cp_model = FusedLasso(sorted_y, **hyperparams)
        M = cp_model.fit()
        # cp_model.check_KKT()
        
        if cp_model.is_empty():
            return None
        
        Mt = list(dict.fromkeys(i // (unit + 1) for i in M[1:-1]))
        Mt = [0] + Mt + [nt-1] # Add boundaries to change points

        # Hypothesis Testing
        # Test statistic
        j = np.random.randint(1, len(Mt)-1, 1)[0]
        cp_selected = Mt[j]
        # print("Selected Change Point:", cp_selected)

        # For FPR tests, we will use the false detected change points
        if delta != 0 and  cp_selected not in list_change_points_t:
            return None
        
        pre_cp = Mt[j-1]
        next_cp = Mt[j+1]
        prev_len = cp_selected - pre_cp
        next_len = next_cp - cp_selected
        etaj = np.zeros(n)        
        etaj[(pre_cp+ns):(cp_selected+ns)] = np.ones(prev_len)/prev_len
        etaj[(cp_selected+ns):(next_cp+ns)] = -np.ones(next_len)/next_len
        etaj = etaj.reshape(-1,1)
        etajTy = np.dot(etaj.T, y)[0][0]
        etajTSigmaetaj = (etaj.T @ Sigma @ etaj)[0][0]
        tn_sigma = np.sqrt(etajTSigmaetaj)
        start = time.time()
        # Selective Inference
        a, b = utils.compute_a_b(y, etaj)      
        intervals = si.fit(a, b, cp_model, da_ins=da_model, zmin=-20*tn_sigma, zmax=20*tn_sigma, 
                           unit=unit, cp_mat=trans_mat)
        p_value = utils.p_value(intervals, etajTy, tn_sigma)
        with open(folder_path + '/p_values.txt', 'a') as f:
            f.write(f"{p_value}\n")
        with open(folder_path + '/times.txt', 'a') as f:
            f.write(f"{time.time()-start}\n")
        return p_value
    except Exception as e:
        print(f"\nError in run({k}): {e}")
        return None

if __name__ == "__main__":
    folder_path = create_experiment_folder(
        config_data={"nt": nt, "unit": unit, "Lambda": Lambda, "delta": delta, 
                     "method": "parametric", "model": model_name}
    )

    max_iter = 120
    alpha = 0.05
    cnt = 0

    args = [[i, folder_path] for i in range(max_iter)]
    list_p_values = []
    with Pool() as pool:
        list_result = list(tqdm(pool.imap_unordered(run, args), total=max_iter, desc="Iter"))

    for p_value in list_result:
        if p_value is None:
            continue
        list_p_values.append(p_value)
        if p_value <= alpha:
            cnt += 1

    FPR = cnt / len(list_p_values)
    print("FPR/TPR:", FPR)
    ks_test = scipy.stats.kstest(list_p_values, "uniform")[1]
    print(f'KS-Test: {ks_test}')

    with open(folder_path+'/metrics.txt', 'w') as f:
        f.write(f"FPR/TPR: \t{FPR}\nKS-Test: \t{ks_test}")

    plt.hist(list_p_values)
    plt.savefig(folder_path + '/p_value_hist.pdf')
    plt.close()