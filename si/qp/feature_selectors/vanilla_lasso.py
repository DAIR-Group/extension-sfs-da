from .. import core
import numpy as np

class VanillaLasso(core.FeatureSelectorBase):
    """
    Vanilla Lasso feature selector.
    """
    def __init__(self, X, y, Lambda, Gamma=None):
        self.Lambda = Lambda
        self.X = X
        self.y = y
        
        D = np.eye(X.shape[1])
        
        m, p = D.shape 
        XTX = X.T.dot(X)
        
        A = np.zeros((p+2*m, p+2*m))
        A[:p, :p] = np.copy(XTX)

        delta1 = Lambda * np.vstack((np.zeros((p, 1)), np.ones((2*m, 1))))
        XTY = X.T.dot(y)
        delta2 = np.vstack((XTY, np.zeros((2*m, 1))))
        delta = delta1 - delta2

        row_1 = np.hstack((np.zeros((m, p)), -np.eye(m), np.zeros((m, m))))
        row_2 = np.hstack((np.zeros((m, p)), np.zeros((m, m)), -np.eye(m)))
        P = np.vstack((row_1, row_2))

        Q = np.hstack((np.copy(D), -np.identity(m), np.identity(m)))
        
        super().__init__(A, delta, P, Q, D)