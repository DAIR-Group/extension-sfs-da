from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp
from ...utils import solve_linear_inequalities
class QuadraticProgramming(ABC):
    def __init__(self):
        '''
        Initialize the Quadratic Programming model
        '''
        self.A = None
        self.Delta = None
        self.P = None
        self.Q = None
        self.u = None
        self.v = None

    def solve(self):
        '''
        Solve the quadratic programming problem
        '''
        x = cp.Variable(self.A.shape[0])
        objective = cp.Minimize(0.5 * cp.quad_form(x, self.A) + self.Delta.T @ x)
        constraints = [self.P @ x <= 0]

        if self.Q is not None:
            constraints.append(self.Q @ x == 0)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-10, eps_rel=1e-10, max_iter=1000000, polish=True ,verbose=False)
        self.eps = x.value.reshape(-1,1)
        self.u = prob.constraints[0].dual_value.reshape(-1,1)

        if self.Q is not None:
            self.v = prob.constraints[1].dual_value.reshape(-1,1)

        return self.eps, self.u, self.v

    def check_KKT(self):
        ''' 
        Check KKT conditions
        '''
        if self.Q is not None:
            sta = self.A @ self.eps + self.Delta + self.P.T @ self.u + self.Q.T @ self.v
            prec = 1e-6
            if np.any((sta < -prec) | (sta > prec)):
                print(sta[np.where((sta < -prec) | (sta > prec))[0],:])
                raise ValueError("Stationarity Condition Failed!")

            Peps = self.P @ self.eps
            Qeps = self.Q @ self.eps
            uPeps = self.u * Peps

            if np.any((uPeps < -prec) | (uPeps > prec)):
                print(uPeps[np.where((uPeps < -prec) | (uPeps > prec))[0],:])
                raise ValueError("Complementary Slackness Failed!")

            if not np.all(Peps <= prec):
                print(Peps[np.where(Peps > prec)[0],:])
                raise ValueError("Primal Feasibility Failed!")

            if np.any((Qeps < -prec) | (Qeps > prec)):
                print(Qeps[np.where((Qeps < -prec) | (Qeps > prec))[0],:])
                raise ValueError("Primal Feasibility Failed!")

            if not np.all(self.u >= -prec):
                print(self.u[np.where(self.u < -prec)[0],:])
                raise ValueError("Dual Feasibility Failed!")
        else:
            
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
        if self.Q is not None: # Vanilla Lasso, Elastic Net, Fused Lasso
            # y = a + bz
            # [-delta] = g0 + g1z
            g0 = -np.vstack((-self.X.T @ a, self.Lambda * np.ones((2*self.m, 1))))
            g1 = -np.vstack((-self.X.T @ b, np.zeros((2*self.m, 1))))

            I = np.where(self.u > 1e-6)[0].tolist()
            Ic = [i for i in range(len(self.u)) if i not in I]
            PI = np.copy(self.P[I, :])
            PIc = np.copy(self.P[Ic, :])

            mat1 = np.vstack((np.hstack((PIc, np.zeros((len(Ic), len(I))))),       
                            np.hstack((np.zeros((len(I), self.p+2*self.m)), -np.identity(len(I))))))

            self.mat2 = np.vstack((np.hstack((self.A, PI.T, self.Q.T)), 
                            np.hstack((PI, np.zeros((len(I), len(I))), np.zeros((len(I), self.m)))), 
                            np.hstack((self.Q, np.zeros((self.m, len(I)+self.m))))))
            
            self.mat2 = np.linalg.inv(self.mat2)
            red = np.hstack((np.eye(self.p+2*self.m+len(I)), np.zeros((self.p+2*self.m+len(I), self.m))))

            self.vec1 = np.vstack((g0, np.zeros((len(I)+self.m, 1))))
            self.vec2 = np.vstack((g1, np.zeros((len(I)+self.m, 1))))

            temp = mat1 @ red @ self.mat2
            p = temp @ self.vec1
            q = temp @ self.vec2

            # Solve the inequalities: p + qz <= 0
            return solve_linear_inequalities(p, q)
        else: # Non-negative Least Squares
            g0 = -np.vstack((-self.X.T @ a))
            g1 = -np.vstack((-self.X.T @ b))

            I = np.where(self.u > 1e-6)[0].tolist()
            Ic = [i for i in range(len(self.u)) if i not in I]
            PI = np.copy(self.P[I, :])
            PIc = np.copy(self.P[Ic, :])
            mat1 = np.vstack((np.hstack((PIc, np.zeros((len(Ic), len(I))))),       
                            np.hstack((np.zeros((len(I), self.P.shape[1])), -np.identity(len(I))))))

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

class FeatureSelectorBase(QuadraticProgramming):
    def gen_data(n, p, true_beta):
        '''
        Generate synthetic data for feature selection
        '''
        X = np.random.normal(loc=0, scale=1, size=(n,p))
        mu = X @ true_beta
        y = mu + np.random.normal(loc=0, scale=1, size=(n,1))
        Sigma = np.identity(n)
        return X, y, mu, Sigma

    def fit(self):
        self.solve()
        self.beta = np.copy(self.eps[0:self.p,:])
        self.active_set = np.where(np.round(self.beta, 9) != 0)[0].tolist()
        return self.active_set
    
    def eval(self, X_val, y_val):
        '''
        Evaluate the model on validation data
        '''    
        y_pred = X_val @ self.beta
        residuals = y_val - y_pred
        mse = 1/2 * np.mean(residuals**2)
        return mse
    
    def is_empty(self):
        return len(self.active_set)==0
    
class ChangePointDetectorBase(QuadraticProgramming):
    def gen_data(n, delta, list_change_points):
        true_y = np.zeros(n)
    
        if len(list_change_points)==1:
            true_y[list_change_points[0]:] += delta
        elif len(list_change_points)>1:
            segments = [(start, end) for start, end in zip(list_change_points[:-1], list_change_points[1:])]
            sign = 1
            for segment in segments:
                start = segment[0]
                end = segment[1]
                true_y[start:end] += sign * delta
                sign = 1 - sign
        
        y = true_y + np.random.normal(0, 1, n)
        return y.reshape(-1,1), true_y.reshape(-1,1), np.eye(n)

    def fit(self):
        self.solve()
        self.beta = np.copy(self.eps[0:self.p])
        temp = self.D @ self.beta
        self.active_set = (np.where(np.round(temp, 9) != 0)[0] + 1).tolist()
        self.active_set = [0] + self.active_set + [self.p - 1]  # Add boundaries to change points
        return self.active_set
    
    def is_empty(self):
        return len(self.active_set)==2