import numpy as np
from . import qp_solver

def gen_data(n, p, true_beta):
    X = np.random.normal(loc = 0, scale = 1, size = (n, p))

    mu = X.dot(true_beta)

    Y = mu + np.random.normal(loc = 0, scale = 1, size = (n, 1))

    Sigma = np.identity(n)

    return X, Y, Sigma

def construct_D(p):
    return np.eye(p)

def construct_A(X):
    p = X.shape[1]
    m = p-1
    XTX = X.T.dot(X)
    A = np.zeros((p+2*m, p+2*m))
    A[:p, :p] = np.copy(XTX)
    return A

def construct_delta(X, Y, Lambda):
    p = X.shape[1]
    m = p-1

    delta1 = Lambda * np.vstack((np.zeros((p, 1)), np.ones((2*m, 1))))
    XTY = X.T.dot(Y)
    delta2 = np.vstack((XTY, np.zeros((2*m, 1))))
    delta = delta1 - delta2
    return delta

def construct_B1(D):
    m = D.shape[0]
    p = D.shape[1]

    # row_1 = np.hstack((D, -np.identity(m), np.identity(m)))
    # row_2 = np.hstack((-D, np.identity(m), -np.identity(m)))
    row_3 = np.hstack((np.zeros((m, p)), -np.eye(m), np.zeros((m, m))))
    row_4 = np.hstack((np.zeros((m, p)), np.zeros((m, m)), -np.eye(m)))
    return np.vstack((row_3, row_4))

def construct_B2(D):
    m = D.shape[0]

    row_1 = np.hstack((D, -np.identity(m), np.identity(m)))
    return row_1
    # row_2 = np.hstack((-D, np.identity(m), -np.identity(m)))
    # return np.vstack((row_1, row_2))

def util(X, Y, Lambda):
    D = construct_D(Y.shape[0])
    A = construct_A(X)
    delta = construct_delta(X, Y, Lambda)
    B1 = construct_B1(D)
    B2 = construct_B2(D)
    return D, A, delta, B1, B2

def fit(A, delta, B1, B2):
    eps, u, v = qp_solver.solve(A, delta, B1, B2)
    return eps, u, v

def find_change_points(beta, D):
    # Find change points from the solution
    change_points = np.where(np.round(D @ beta, 9) != 0)[0] + 1
    return change_points

def check_KKT(eps, u, v, A, delta, B1, B2):
    # Check KKT conditions
    print("Checking KKT conditions...")
    sta = A @ eps + delta + B1.T @ u + B2.T @ v
    if not np.all(np.round(sta, 9) == 0):
        print("\tStationarity Condition Failed!")

    B1eps = B1 @ eps
    B2eps = B2 @ eps
    uB1eps = u * B1eps
    if not np.all(np.round(uB1eps, 9) == 0):
        print("\tComplementary Slackness Failed!")

    if not np.all(np.round(B1eps, 9) <= 0) and not np.all(np.round(B2eps, 9) == 0):
        print("\tPrimal Feasibility Failed!")

    if not np.all(np.round(u, 9) >= 0):
        print("\tDual Feasibility Failed!")
    print("Finished checking KKT conditions.")    