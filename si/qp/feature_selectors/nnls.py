from .. import core
import numpy as np
import cvxpy as cp
from ...utils import solve_linear_inequalities

class NNLS(core.FeatureSelectorBase):
    """
    Non-Negative Least Squares feature selector.
    """
    def __init__(self, X, y, Lambda=None, Gamma=None):
        self.X = X
        self.y = y

        self.D = np.eye(X.shape[1])
        self.m, self.p = self.D.shape
        XTX = X.T.dot(X)

        self.A = np.zeros((self.p+2*self.m, self.p+2*self.m))
        self.A[:self.p, :self.p] = np.copy(XTX)

        XTY = X.T.dot(y)
        self.Delta = np.vstack((XTY, np.zeros((2*self.m, 1))))
        self.P = np.hstack((-np.eye(self.m), np.zeros((self.m,self.m)), np.zeros((self.m,self.m))))
    
    def solve(self):
        '''
        Solve the quadratic programming problem
        '''
        x = cp.Variable(self.A.shape[0])
        objective = cp.Minimize(0.5 * cp.quad_form(x, self.A)  + self.Delta.T @ x)
        constraints = [self.P @ x <= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, verbose=False)
        self.eps = x.value.reshape(-1,1)
        self.u = prob.constraints[0].dual_value.reshape(-1,1)
        return self.eps, self.u
    
    def check_KKT(self):
        ''' 
        Check KKT conditions
        '''
        sta = self.A @ self.eps + self.Delta + self.P.T @ self.u
        prec = 1e-9
        if np.any((sta < -prec) | (sta > prec)):
            print(sta[np.where((sta < -prec) | (sta > prec))[0],:])
            raise ValueError("Stationarity Condition Failed!")

        Peps = self.P @ self.eps
        uPeps = self.u * Peps

        if np.any((uPeps < -prec) | (uPeps > prec)):
            print(uPeps[np.where((uPeps < -prec) | (uPeps > prec))[0],:])
            raise ValueError("Complementary Slackness Failed!")

        if not np.all(Peps <= prec):
            print(Peps[np.where(Peps > prec)[0],:])
            raise ValueError("Primal Feasibility Failed!")

        if not np.all(self.u >= -prec):
            print(self.u[np.where(self.u < -prec)[0],:])
            raise ValueError("Dual Feasibility Failed!")
        
    def si(self, a, b):
        '''
        Selective Inference
        '''
        # y = a + bz
        # [-delta] =  g0 + g1z
        g0 = -np.vstack((-self.X.T @ a, np.ones((2*self.m, 1))))
        g1 = -np.vstack((-self.X.T @ b, np.zeros((2*self.m, 1))))

        I = np.where(self.u > 0)[0].tolist()
        Ic = [i for i in range(len(self.u)) if i not in I]
        PI = np.copy(self.P[I, :])
        PIc = np.copy(self.P[Ic, :])
        mat1 = np.vstack((np.hstack((PIc, np.zeros((len(Ic), len(I))))),       
                        np.hstack((np.zeros((len(I), self.p+2*self.m)), -np.identity(len(I))))))

        mat2 = np.vstack((np.hstack((self.A, PI.T)), 
                          np.hstack((PI, np.zeros((len(I), len(I)))))))
        mat2 = np.linalg.inv(mat2)

        vec1 = np.vstack((g0, np.zeros((len(I), 1))))
        vec2 = np.vstack((g1, np.zeros((len(I), 1))))

        temp = mat1 @ mat2
        p = temp @ vec1
        q = temp @ vec2

        # Solve the inequalities: p + qz <= 0
        return solve_linear_inequalities(p, q)