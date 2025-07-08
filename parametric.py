import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import si.qp.fused_lasso as fused_lasso
import si.da.otda as otda
import os
import si
from tqdm import tqdm
from multiprocessing import Pool

def run(k):
    try:
        # Generate target data
        np.random.seed(k)
        nt = 20
        delta_t = 4
        list_change_points = []
        yt, _, Sigma_t = fused_lasso.gen_data(nt, delta_t, list_change_points)

        # Generate source data
        unit = 5
        ns = (nt-1) * unit
        scale = 2
        delta_s = scale * delta_t
        list_change_points = [i*(unit+1) for i in list_change_points]
        ys, _, Sigma_s = fused_lasso.gen_data(ns, delta_s, list_change_points)

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
        y = np.vstack((ys, yt))
        
        # Domain Adaptation
        c_, cost = otda.construct_cost(ys, yt)
        H = otda.construct_H(ns, nt)
        h = otda.construct_h(ns, nt)
        T, B = otda.fit(ys, yt, cost, H, h)

        Omega = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt)))))
        y_tilde = Omega @ y

        sorted_y_tilde = trans_mat @ y_tilde
        Lambda = 2
        # Construct the quadratic program
        D, A, delta, B1, B2 = fused_lasso.util(np.eye(n), sorted_y_tilde, Lambda)
        eps, u, v = fused_lasso.fit(A, delta, B1, B2)
        beta = eps[0:n]

        # Find change points
        M = fused_lasso.find_change_points(beta, D)
        if len(M)==0:
            return None
        # print("Estimated Change Points:", M)
        M = [0] + M + [n-1]  # Add boundaries to change points

        # Hypothesis Testing
        # Test statistic
        j = np.random.randint(1, len(M)-1, 1)[0]
        cp_selected = M[j]
        # print("Selected Change Point:", cp_selected)

        # For FPR tests, we will use the false detected change points
        if cp_selected in list_change_points:
            return None 
        
        pre_cp = M[j-1]
        next_cp = M[j+1]
        prev_len = cp_selected - pre_cp
        next_len = next_cp - cp_selected
        etaj = np.zeros(n)
        etaj[pre_cp:cp_selected] = np.ones(prev_len)/prev_len
        etaj[cp_selected:next_cp] = - np.ones(next_len)/next_len
        etaj = etaj.reshape(-1,1)

        p_value = si.fit(etaj, ns, nt, ys, yt, Sigma_s, Sigma_t, trans_mat, Lambda, M, zmin=-20, zmax=20)
        with open('./results/parametric/p_values.txt', 'a') as f:
            f.write(f"{p_value}\n")
        return p_value
    
    except Exception as e:
        print(f"\nError in run({k}): {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # run(0)
    os.environ["MKL_NUM_THREADS"] = "1" 
    os.environ["NUMEXPR_NUM_THREADS"] = "1" 
    os.environ["OMP_NUM_THREADS"] = "1" 
    
    max_iter = 120
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
    plt.savefig('./results/parametric/p_value_hist.pdf')
    plt.show()
    plt.close()

    plt.rcParams.update({'font.size': 16})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_values))(grid), 'r-', linewidth=5, label='p-value')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/parametric/uniform_pivot.pdf')
    plt.show()
    plt.close()

    print("FPR:", cnt / len(list_p_values))
    print(f'KS-Test result: {scipy.stats.kstest(list_p_values, "uniform")[1]}')