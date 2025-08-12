import numpy as np
from ..utils import solve_quadratic_inequality, intersect

class HoldOutCV():
    def __init__(self, val_size=0.3, train_indices=None, val_indices=None, random_state=None):
        self.val_size = val_size
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.random_state = random_state

    def split(self, n):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = n
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        split_index = int(n_samples * (1 - self.val_size))
        self.train_indices = indices[:split_index]
        self.val_indices = indices[split_index:]

        if len(self.train_indices) == 0 or len(self.val_indices) == 0:
            raise ValueError("Not enough samples to split into training and validation sets.")
        
        return self.train_indices, self.val_indices
    
    def fit(self, X, y, model_class, list_lambda):
        self.X_train, self.y_train = X[self.train_indices, :], y[self.train_indices, :]
        self.X_val, self.y_val = X[self.val_indices, :], y[self.val_indices, :]

        best_score = np.inf
        self.best_model = None
        self.list_lambda = list_lambda
        self.list_models = []
        
        for lam in self.list_lambda:
            model = model_class(self.X_train, self.y_train, Lambda=lam)
            model.fit()
            val_score = model.eval(self.X_val, self.y_val)
            if val_score < best_score:
                best_score = val_score
                self.best_model = model

            self.list_models.append(model)
        return self.best_model.Lambda, best_score

    def si(self, a, b):
        """
        Selective Inference
        """
        a_train, b_train = a[self.train_indices,:], b[self.train_indices,:]
        a_val, b_val = a[self.val_indices,:], b[self.val_indices,:]
        
        flag = False
        intervals_1 = []
        for model in self.list_models:
            temp = model.si(a_train, b_train)
            
            if not flag:
                flag = True
                intervals_1 = temp
            else:
                intervals_1 = intersect(intervals_1, temp)

            # print(model.Lambda, temp, intervals_1)

        flag = False
        intervals_2 = []
        p = self.best_model.p
        l0_cv = (self.best_model.mat2 @ self.best_model.vec1)[0:p,:]
        l1_cv = (self.best_model.mat2 @ self.best_model.vec2)[0:p,:]
        
        X = self.X_val
        Xl0_cv = X @ l0_cv
        Xl1_cv = X @ l1_cv

        left_a = Xl1_cv.T @ (Xl1_cv - 2 * b_val)
        left_b = 2 * (Xl0_cv.T @ Xl1_cv - Xl0_cv.T @ b_val - Xl1_cv.T @ a_val)
        left_c = Xl0_cv.T @ (Xl0_cv - 2 * a_val)

        for model in self.list_models:
            if model.Lambda == self.best_model.Lambda:
                continue

            l0 = (model.mat2 @ model.vec1)[0:p,:]
            l1 = (model.mat2 @ model.vec2)[0:p,:]
            Xl0 = X @ l0
            Xl1 = X @ l1

            right_a = Xl1.T @ (Xl1 - 2 * b_val)
            right_b = 2 * (Xl0.T @ Xl1 - Xl0.T @ b_val - Xl1.T @ a_val)
            right_c = Xl0.T @ (Xl0 - 2 * a_val)

            fa = (left_a - right_a)[0,0]
            sa = (left_b - right_b)[0,0]
            ta = (left_c - right_c)[0,0]

            temp = solve_quadratic_inequality(fa, sa, ta)
            if not flag:
                flag = True
                intervals_2 = temp
            else:
                intervals_2 = intersect(intervals_2, temp)

        return intersect(intervals_1, intervals_2)