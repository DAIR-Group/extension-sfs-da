import si
from si import utils
from si import OTDA, VanillaLasso
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from multiprocessing import Pool
import statsmodels.api as sm
import scipy.stats 
import json
import re

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

ns, nt, p = 100, 10, 5
Lambda = 10
Gamma = 1
true_beta = 0
true_beta_t = np.full((p, 1), true_beta)
model_name = "OT-VanillaLasso"

def run(args):
    k = args[0]
    folder_path = args[1]
    try:
        # Generate target data
        np.random.seed(k)
        Xs, ys, mu_s, Sigma_s = VanillaLasso.gen_data(ns, p, true_beta_s)
        true_beta_s = np.full((p, 1), 2)
        Xt, yt, mu_t, Sigma_t = VanillaLasso.gen_data(nt, p, true_beta_t)

        X = np.vstack((Xs, Xt))
        y = np.vstack((ys, yt))
        mu = np.vstack((mu_s, mu_t))
        Sigma = np.vstack((np.hstack((Sigma_s , np.zeros((ns, nt)))),
                            np.hstack((np.zeros((nt, ns)), Sigma_t))))

        da_model = OTDA(np.hstack((Xs, ys)), np.hstack((Xt, yt)))
        T, _ = da_model.fit()
        # da_model.check_KKT()

        # Adapt the data
        Omega = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt)))))
        X_tilde = Omega @ X
        y_tilde = Omega @ y

        hyperparams = {'Lambda': Lambda, 'Gamma': Gamma}
        fs_model = VanillaLasso(X_tilde, y_tilde, **hyperparams)
        M = fs_model.fit()
        # fs_model.check_KKT()

        if fs_model.is_empty():
            return None
                
        # Hypothesis Testing
        # Test statistic
        j = np.random.randint(0, len(M), 1)[0]
        ej = np.zeros((len(M), 1))
        ej[j] = 1
        XM = X[:, M]
        etaj = XM @ np.linalg.inv(XM.T @ XM) @ ej
        etajTy = np.dot(etaj.T, y)[0][0]
        etajTSigmaetaj = (etaj.T @ Sigma @ etaj)[0][0]
        tn_sigma = np.sqrt(etajTSigmaetaj)

        # Selective Inference
        a, b = utils.compute_a_b(y, etaj)
        intervals = si.fit(a, b, fs_model, da_model, zmin=-20*tn_sigma, zmax=20*tn_sigma)
        p_value = utils.p_value(intervals, etajTy, tn_sigma)
        with open(folder_path + '/p_values.txt', 'a') as f:
            f.write(f"{p_value}\n")
        return p_value
    except Exception as e:
        print(f"\nError in run({k}): {e}")
        return None

if __name__ == "__main__":
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    folder_path = create_experiment_folder(
        config_data={"ns": ns, "nt": nt, "p": p, "Lambda": Lambda, "Gamma": Gamma, "true_beta": true_beta, 
                     "method": "parametric", "model": model_name}
    )

    max_iter = 1000
    alpha = 0.05
    cnt = 0

    args = [[i, folder_path] for i in range(max_iter)]
    list_p_values = []
    with Pool(processes=10) as pool:
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
    # plt.show()
    plt.close()