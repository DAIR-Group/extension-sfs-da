from .. import core
from ...utils import solve_linear_inequalities
import numpy as np

class ElasticNet(core.FeatureSelectorBase):
    """
    Elastic Net feature selector.
    """
    def __init__(self, X, y, **kwargs):
        super().__init__()

        self.X = X
        self.y = y

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.D = np.eye(X.shape[1])
        
        self.m, self.p = self.D.shape 
        XTX_Gamma = X.T.dot(X) + self.Gamma * np.eye(self.p)

        self.A = np.zeros((self.p+2*self.m, self.p+2*self.m))
        self.A[:self.p, :self.p] = XTX_Gamma

        delta1 = self.Lambda * np.vstack((np.zeros((self.p, 1)), np.ones((2*self.m, 1))))
        XTY = X.T.dot(y)
        delta2 = np.vstack((XTY, np.zeros((2*self.m, 1))))
        self.Delta = delta1 - delta2

        row_1 = np.hstack((np.zeros((self.m, self.p)), -np.eye(self.m), np.zeros((self.m, self.m))))
        row_2 = np.hstack((np.zeros((self.m, self.p)), np.zeros((self.m, self.m)), -np.eye(self.m)))
        self.P = np.vstack((row_1, row_2))

        self.Q = np.hstack((np.copy(self.D), -np.identity(self.m), np.identity(self.m)))

    def get_hyperparams(self):
        return {'Lambda': self.Lambda, 'Gamma': self.Gamma}