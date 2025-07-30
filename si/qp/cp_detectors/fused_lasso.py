from .. import core
from ...utils import solve_linear_inequalities
import numpy as np

class FusedLasso(core.ChangePointDetectorBase):
    """
    Fused Lasso change point detector.
    """
    def __init__(self, y, Lambda):
        super().__init__

        self.Lambda = Lambda
        self.X = np.eye(y.shape[0])
        self.y = y

        self.D = (np.diag([-1] * y.shape[0], k=0) + np.diag([1] * (y.shape[0] - 1), k=1))[:-1]
        self.m, self.p = self.D.shape
        XTX = self.X.T.dot(self.X)

        self.A = np.zeros((self.p+2*self.m, self.p+2*self.m))
        self.A[:self.p, :self.p] = XTX

        delta1 = Lambda * np.vstack((np.zeros((self.p, 1)), np.ones((2*self.m, 1))))
        XTY = self.X.T.dot(y)
        delta2 = np.vstack((XTY, np.zeros((2*self.m, 1))))
        self.Delta = delta1 - delta2

        row_1 = np.hstack((np.zeros((self.m, self.p)), -np.eye(self.m), np.zeros((self.m, self.m))))
        row_2 = np.hstack((np.zeros((self.m, self.p)), np.zeros((self.m, self.m)), -np.eye(self.m)))
        self.P = np.vstack((row_1, row_2))
        self.Q = np.hstack((np.copy(self.D), -np.identity(self.m), np.identity(self.m)))