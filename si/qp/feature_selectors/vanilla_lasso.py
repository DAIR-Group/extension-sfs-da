from .. import core
from ...utils import solve_linear_inequalities
import numpy as np

class VanillaLasso(core.FeatureSelectorBase):
    """
    Vanilla Lasso feature selector.
    """
    def __init__(self, X, y, Lambda):
        super().__init__()

        self.Lambda = Lambda
        self.X = X
        self.y = y

        self.D = np.eye(X.shape[1])
        self.m, self.p = self.D.shape
        XTX = self.X.T.dot(self.X)

        self.A = np.zeros((self.p+2*self.m, self.p+2*self.m))
        self.A[:self.p, :self.p] = np.copy(XTX)

        delta1 = Lambda * np.vstack((np.zeros((self.p, 1)), np.ones((2*self.m, 1))))
        XTY = self.X.T.dot(y)
        delta2 = np.vstack((XTY, np.zeros((2*self.m, 1))))
        self.Delta = delta1 - delta2

        row_1 = np.hstack((np.zeros((self.m, self.p)), -np.eye(self.m), np.zeros((self.m, self.m))))
        row_2 = np.hstack((np.zeros((self.m, self.p)), np.zeros((self.m, self.m)), -np.eye(self.m)))
        self.P = np.vstack((row_1, row_2))
        self.Q = np.hstack((np.copy(self.D), -np.identity(self.m), np.identity(self.m)))