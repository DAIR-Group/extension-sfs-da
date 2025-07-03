import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import si.siqp.fused_lasso as fused_lasso
import si.siqp as siqp
import si.sida.ot as ot
import si.sida as sida
import si.util as util

def run():
    # np.random.seed(42)
    # Generate target data
    nt = 20
    delta_t = 4
    list_change_points = [10, 15]
    yt, true_yt, Sigma_t = fused_lasso.gen_data(nt, delta_t, list_change_points)

    # plt.figure(figsize=(10, 5))
    # plt.plot(yt, label="Target Data", linestyle="dotted", marker="o", alpha=0.5)
    # plt.plot(true_yt, label="Ground Truth", linestyle="dotted", marker="o", alpha=0.5)
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.show()
    # plt.close()

    # Generate source data
    unit = 3
    ns = (nt-1) * unit
    scale = 2
    delta_s = scale * delta_t
    list_change_points = [i*(unit+1) for i in list_change_points]
    ys, true_ys, Sigma_s = fused_lasso.gen_data(ns, delta_s, list_change_points)

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
    true_y = np.vstack((true_ys, true_yt))
    Sigma = np.vstack((np.hstack((Sigma_s , np.zeros((ns, nt)))),
                        np.hstack((np.zeros((nt, ns)), Sigma_t))))
    # Domain Adaptation
    c_, cost = ot.construct_cost(ys, yt)
    H = ot.construct_H(ns, nt)
    h = ot.construct_h(ns, nt)
    T, B, Bc = ot.fit(ys, yt, cost, H, h)
    Omega = np.hstack((np.zeros((ns + nt, ns)), np.vstack((ns * T, np.identity(nt)))))
    y_tilde = Omega @ y
    true_y_tilde = Omega @ true_y

    sorted_y_tilde = trans_mat @ y_tilde
    # sorted_true_y_tilde = trans_mat @ true_y_tilde

    # plt.figure(figsize=(10, 5))
    # plt.plot(sorted_y_tilde, label="Transformed Data", linestyle="dotted", marker="o", alpha=0.5)
    # plt.plot(sorted_true_y_tilde, label="Ground Truth", linestyle="dotted", marker="o", alpha=0.5)
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.show()
    # plt.close()

    Lambda = 2
    # Construct the quadratic program
    D, A, delta, B1, B2 = fused_lasso.util(np.eye(n), sorted_y_tilde, Lambda)
    eps, u, v = fused_lasso.fit(A, delta, B1, B2)
    # siqp.check_KKT(eps, u, v, A, delta, B1, B2)
    beta = eps[0:n]

    # plt.figure(figsize=(10, 5))
    # plt.plot(sorted_y_tilde, label="Transformed Data", linestyle="dotted", marker="o", alpha=0.5)
    # plt.plot(sorted_true_y_tilde, label="Ground Truth", linestyle="dotted", marker="o", alpha=0.5)
    # plt.plot(beta, label="Fused Lasso Estimate", color="red", linewidth=2)
    # plt.xlabel("Index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.show()
    # plt.close()

    # Find change points
    M = fused_lasso.find_change_points(beta, D)
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
    etajTy = np.dot(etaj.T, y)[0][0]
    etajTSigmaetaj = (etaj.T @ Sigma @ etaj)[0][0]
    tn_sigma = np.sqrt(etajTSigmaetaj)

    # Selective inference
    a, b = util.compute_a_b(y, etaj)
    intervals_1 = sida.fit(ns, nt, np.eye(n), y, a, b, c_, H, B, Bc)
    a_tilde = Omega@a
    b_tilde = Omega@b
    interval_2 = siqp.fit(np.eye(n), sorted_y_tilde, trans_mat @ a_tilde, trans_mat @ b_tilde, Lambda, u, v, A, B1, B2)
    intervals = util.intersect(intervals_1, interval_2)
    p_value = util.p_value(intervals, etajTy, tn_sigma)
    return p_value

if __name__ == "__main__":
    # run()
    max_iter = 1000
    alpha = 0.05
    list_p_values = []
    cnt = 0
    for i in range(max_iter):
        print(f"Iteration {i+1}/{max_iter}")
        np.random.seed(i)
        p_value = run()
        if p_value is None:
            continue
        list_p_values.append(p_value)
        if p_value <= alpha:
            cnt += 1

    plt.hist(list_p_values)
    plt.show()
    plt.close()

    plt.rcParams.update({'font.size': 16})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_values))(grid), 'r-', linewidth=5, label='p-value')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('./results/uniform_pivot.png', dpi=100)
    plt.show()
    plt.close()

    print("FPR:", cnt / max_iter)
    print(f'KS-Test result: {scipy.stats.kstest(list_p_values, "uniform")[1]}')