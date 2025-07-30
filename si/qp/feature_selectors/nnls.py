from .. import core
from ...utils import solve_linear_inequalities
import numpy as np

class NNLS(core.FeatureSelectorBase):
    """
    Non-Negative Least Squares feature selector.
    """
    def __init__(self, X, y):
        super().__init__()

        self.X = X
        self.y = y

        self.p = self.X.shape[1]
        self.A = X.T.dot(X)
        self.Delta = -X.T.dot(y)
        self.P = -np.eye(self.p)