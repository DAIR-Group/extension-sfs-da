import numpy as np
from mpmath import mp
mp.dps = 1000

def compute_a_b(y, etaj):    
    sq_norm = (np.linalg.norm(etaj))**2
    e1 = np.identity(y.shape[0]) - (np.dot(etaj, etaj.T))/sq_norm
    a = np.dot(e1, y)
    b = etaj/sq_norm
    return a, b

def solve_linear_inequalities(p, q):
    # Solve system of inequalities has the form of p+qz <= 0
    left = -q
    right = p
    Z_minus = -np.inf
    Z_plus = np.inf

    for i in range(left.shape[0]):
        l = left[i][0]
        r = right[i][0]
        if - 1e-10 <= l <= 1e-10:
            l = 0.0
        if l == 0.0:
            continue
        else:
            temp = r / l
            if l > 0.0:
                Z_minus = max(temp, Z_minus)
            else:
                Z_plus = min(temp, Z_plus)
    return [[Z_minus, Z_plus]]

def solve_quadratic_inequality(a, b, c):
    # if -1e-10 <= a <= 1e-10:
    #     a = 0.0

    # if -1e-10 <= b <= 1e-10:
    #     b = 0.0

    # if -1e-10 <= c <= 1e-10:
    #     c = 0.0

    # Solve quadratic inequality has the form of ax^2 + bx + c <= 0
    if a == 0.0:
        if b != 0.0:
            root = -c/b

            if b > 0.0:
                return [[-np.inf, root]]
            else:
                return [[root, np.inf]]
        
        else:
            if c > 0.0:
                # print("Error no roots: c > 0")
                print("Case 1:", a, b, c)
                return None
            else:
                return [[-np.inf, np.inf]]
            
    delta = b**2 - 4 * a * c

    # if -1e-10 <= delta <= 1e-10:
    #     delta = 0.0

    if delta < 0.0:
        if a > 0.0:
            print("Error no roots: a > 0")
            # print("Case 2:", a, b, c, delta)
            return None
        else: 
            return [[-np.inf, np.inf]]
    else:
        sqrt_delta = np.sqrt(delta)

        if b > 0:
            root1 = (-b-sqrt_delta)/(2*a)
        else:
            root1 = (-b+sqrt_delta)/(2*a)

        root2 = c / (a * root1)

        roots = sorted([root1, root2])
        
        if a > 0:
            return [roots]
        else:
            return [[-np.inf, roots[0]], [roots[1], np.inf]]
        
def intersect(list_intervals_1, list_intervals_2):
    results = []
    for interval_1 in list_intervals_1:
        for interval_2 in list_intervals_2:
            if interval_1[1] < interval_2[0]:
                break
            if interval_1[0] > interval_2[1]:
                continue
            results.append([max(interval_1[0], interval_2[0]), min(interval_1[1], interval_2[1])])
    return results

def pivot(list_itvs, etajTy, tn_mu, tn_sigma):
    if len(list_itvs) == 0:
        return None
    list_tn_cdfs = []
    for interval in list_itvs:
        temp = mp.ncdf((interval[1] - tn_mu) / tn_sigma) - mp.ncdf((interval[0] - tn_mu) / tn_sigma)
        list_tn_cdfs.append(temp)
    numerator = 0
    for i in range(len(list_itvs)):
        interval = list_itvs[i]
        if etajTy > interval[1]:
            numerator += list_tn_cdfs[i]
        else:
            numerator += mp.ncdf((etajTy - tn_mu) / tn_sigma) - mp.ncdf((interval[0] - tn_mu) / tn_sigma)
            break
    denominator = sum(list_tn_cdfs)
    if denominator == 0.0:
        print('Numerical error')
        return None
    
    tn_cdf = float(numerator / denominator)
    return tn_cdf

def p_value(list_itvs, etajTy, tn_sigma):
    value = pivot(list_itvs, etajTy, 0, tn_sigma)
    return 2 * min(1 - value, value)