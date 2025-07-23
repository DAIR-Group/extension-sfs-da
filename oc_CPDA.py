from si import utils
from si import OTDA, FusedLasso
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from multiprocessing import Pool
import statsmodels.api as sm
import scipy.stats

def run(k):
    try:
        # Generate target data
        np.random.seed(k)
        nt = 20
        unit = 5
        ns = (nt-1) * unit
        Lambda = 1
        delta_s, delta_t = 2, 4
        
        # list_change_points_t = []
        list_change_points_t = list(np.arange(2, nt, 2))
        yt, mu_t, Sigma_t = FusedLasso.gen_data(nt, delta_t, list_change_points_t)
        
        list_change_points_s = [i*unit for i in list_change_points_t]
        ys, mu_s, Sigma_s = FusedLasso.gen_data(ns, delta_s, list_change_points_s)

        y = np.vstack((ys, yt))
        mu = np.vstack((mu_s, mu_t))
        Sigma = np.vstack((np.hstack((Sigma_s , np.zeros((ns, nt)))),
                            np.hstack((np.zeros((nt, ns)), Sigma_t))))

        da_model = OTDA(ys, yt)
        T, _ = da_model.fit()
        da_model.check_KKT()

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

        list_change_points = [i*(unit+1) for i in list_change_points_t]
        sorted_y = trans_mat @ y_tilde

        cp_model = FusedLasso(sorted_y, Lambda)
        M = cp_model.fit()
        cp_model.check_KKT()
        
        if len(M)==2:
            return None
                
        # Hypothesis Testing
        # Test statistic
        j = np.random.randint(1, len(M)-1, 1)[0]
        cp_selected = M[j]
        # print("Selected Change Point:", cp_selected)

        # For FPR tests, we will use the false detected change points
        # if cp_selected in list_change_points:
        #     return None

        # For TPR tests        
        if cp_selected not in list_change_points:
            return None
        
        pre_cp = M[j-1]
        next_cp = M[j+1]
        prev_len = cp_selected - pre_cp
        next_len = next_cp - cp_selected

        etaj = np.zeros(n)
        etaj[pre_cp:cp_selected] = np.ones(prev_len)/prev_len
        etaj[cp_selected:next_cp] = - np.ones(next_len)/next_len
        etaj = etaj.reshape(-1,1)
        etajTy = np.dot(etaj.T, y)[0][0]
        etajTSigmaetaj = (etaj.T @ Sigma @ etaj)[0][0]
        tn_sigma = np.sqrt(etajTSigmaetaj)

        # Selective Inference
        a, b = utils.compute_a_b(y, etaj)
        intervals_da = da_model.si(a, b)

        a_tilde, b_tilde = Omega @ a, Omega @ b
        a_sorted, b_sorted = trans_mat @ a_tilde, trans_mat @ b_tilde
        intervals_cp = cp_model.si(a_sorted, b_sorted)

        intervals = utils.intersect(intervals_da, intervals_cp)
        p_value = utils.p_value(intervals, etajTy, tn_sigma)
        # with open('./results/p_values.txt', 'a') as f:
        #     f.write(f"{p_value}\n")
        return p_value
    except Exception as e:
        print(f"\nError in run({k}): {e}")
        return None

if __name__ == "__main__":
    # run(0)
    # max_iter = 120
    # for i in tqdm(range(max_iter), total=max_iter, desc="Processing"):
    #     run(i)
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    max_iter = 10000
    alpha = 0.05
    cnt = 0

    list_p_values = []
    with Pool() as pool:
        list_result = list(tqdm(pool.imap(run, range(max_iter)), total=max_iter, desc="Iter"))

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