import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

from si import utils, OTDA, KFoldCV, VanillaLasso
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import statsmodels.api as sm
import scipy.stats

def run(k):
    try:
        # Generate target data
        np.random.seed(k)
        ns, nt, p = 100, 20, 5

        true_beta = np.full((p, 1), 0)
        list_lambda = [2 ** x for x in range(-5, 6)]

        Xs, ys, mu_s, Sigma_s = VanillaLasso.gen_data(ns, p, true_beta)
        Xt, yt, mu_t, Sigma_t = VanillaLasso.gen_data(nt, p, true_beta)

        X = np.vstack((Xs, Xt))
        y = np.vstack((ys, yt))
        mu = np.vstack((mu_s, mu_t))
        Sigma = np.vstack((np.hstack((Sigma_s , np.zeros((ns, nt)))),
                            np.hstack((np.zeros((nt, ns)), Sigma_t))))

        # Data Splitting
        indices = np.arange(nt)
        np.random.shuffle(indices)
        n_test = int(nt * 0.3)  # 30% test set
        split_index = nt - n_test # 70% train
        Xt_train, yt_train = Xt[indices[:split_index]], yt[indices[:split_index]]
        Xt_test, _ = Xt[indices[split_index:]], yt[indices[split_index:]]

        da_model = OTDA(np.hstack((Xs, ys)), np.hstack((Xt_train, yt_train)))
        T, _ = da_model.fit()
        # da_model.check_KKT()

        # Adapt the data
        X_train = np.vstack((Xs, Xt_train))
        y_train = np.vstack((ys, yt_train))
        Omega = np.hstack((np.zeros((ns + nt - n_test, ns)), np.vstack((ns * T, np.identity(nt-n_test)))))
        X_tilde = Omega @ X_train
        y_tilde = Omega @ y_train
        
        # Tuning Lambda
        cv = KFoldCV(random_state=k)
        cv.split(ns+nt-n_test)
        best_Lambda, _ = cv.fit(X_tilde, y_tilde, VanillaLasso, list_lambda)
        Gamma = 1
        hyperparams = {'Lambda': best_Lambda, 'Gamma': Gamma}
        fs_model = VanillaLasso(X_tilde, y_tilde, **hyperparams)
        M = fs_model.fit()
        # fs_model.check_KKT()
        
        if fs_model.is_empty():
            return None
                
        # Hypothesis Testing
        # Test statistic on the test set
        j = np.random.randint(0, len(M), 1)[0]
        ej = np.zeros((len(M), 1))
        ej[j][0] = 1
        XtM = Xt_test[:, M]
        Delta = np.hstack((np.zeros((n_test, ns+nt-n_test)), np.eye(n_test)))
        etaj = Delta.T @ XtM @ np.linalg.inv(XtM.T @ XtM) @ ej
        etajTy = np.dot(etaj.T, y)[0][0]
        etajTSigmaetaj = (etaj.T @ Sigma @ etaj)[0][0]
        tn_sigma = np.sqrt(etajTSigmaetaj)

        cdf = scipy.stats.norm.cdf(etajTy, loc=0, scale=tn_sigma)
        p_value = 2 * min(1 - cdf, cdf)
        return p_value

    except Exception as e:
        print(f"\nError in run({k}): {e}")
        return None

if __name__ == "__main__":
    max_iter = 120
    alpha = 0.05
    cnt = 0

    list_p_values = []
    with Pool() as pool:
        list_result = list(tqdm(pool.imap_unordered(run, range(max_iter)), total=max_iter, desc="Iter"))

    for p_value in list_result:
        if p_value is None:
            continue
        list_p_values.append(p_value)
        if p_value <= alpha:
            cnt += 1

    plt.hist(list_p_values)
    # plt.savefig('./results/p_value_hist.pdf')
    plt.show()
    plt.close()

    plt.rcParams.update({'font.size': 16})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_values))(grid), 'r-', linewidth=5, label='p-value')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('./results/uniform_pivot.pdf')
    plt.show()
    plt.close()

    print("FPR:", cnt / len(list_p_values))
    print(f'KS-Test result: {scipy.stats.kstest(list_p_values, "uniform")[1]}')