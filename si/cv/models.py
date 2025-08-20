import numpy as np
from ..utils import solve_quadratic_inequality, intersect

class HoldOutCV():
    def __init__(self, val_size=0.3, train_val_pairs=None, random_state=None):
        self.val_size = val_size
        self.train_val_pairs = train_val_pairs if train_val_pairs is not None else []
        self.train_indices, self.val_indices = train_val_pairs[0] if train_val_pairs else (None, None)
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
        self.train_val_pairs = [(self.train_indices, self.val_indices)]
        return self.train_val_pairs
    
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
    
class KFoldCV():
    def __init__(self, n_splits=5, train_val_pairs=None, random_state=None):
        self.n_splits = n_splits
        self.train_val_pairs = train_val_pairs if train_val_pairs is not None else []
        self.random_state = random_state

    def split(self, n):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = n
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        folds = np.array_split(indices, self.n_splits)
        for i in range(self.n_splits):
            val_idx = folds[i].tolist()
            train_idx = np.concatenate([folds[j] for j in range(self.n_splits) if j != i]).tolist()
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError("Not enough samples to split into training and validation sets.")
            self.train_val_pairs.append((train_idx, val_idx))
        return self.train_val_pairs
    
    def fit(self, X, y, model_class, list_lambda):
        self.X_train = []
        self.y_train = []
        self.X_val = []
        self.y_val = []

        self.list_lambda = list_lambda
        self.list_models = {lam: [] for lam in self.list_lambda}
        self.list_scores = {lam: 0 for lam in self.list_lambda}
        for train_idx, val_idx in self.train_val_pairs:
            X_train, y_train = X[train_idx, :], y[train_idx, :]
            X_val, y_val = X[val_idx, :], y[val_idx, :]

            for lam in self.list_lambda:
                model = model_class(X_train, y_train, Lambda=lam)
                model.fit()
                val_score = model.eval(X_val, y_val)
                self.list_scores[lam] += val_score
                self.list_models[lam].append(model)

            self.X_train.append(X_train)
            self.y_train.append(y_train)
            self.X_val.append(X_val)
            self.y_val.append(y_val)

        best_score = np.inf
        self.best_lam = None
        # Average the scores
        for lam in self.list_lambda:
            self.list_scores[lam] /= self.n_splits
            if self.list_scores[lam] < best_score:
                best_score = self.list_scores[lam]
                self.best_lam = lam

        return self.best_lam, best_score
    
    def si(self, a, b):
        """
        Selective Inference
        """
        # flag_1 = False
        # intervals_1 = [[-np.inf, np.inf]]
        # for lam in self.list_lambda:
        #     for (model, (train_idx, _)) in zip(self.list_models[lam], self.train_val_pairs):
        #         a_train, b_train = a[train_idx,:], b[train_idx,:]
        #         temp = model.si(a_train, b_train)
        #         if not flag:
        #             flag = True
        #             intervals_1 = temp
        #         else:
        #             intervals_1 = intersect(intervals_1, temp)

        # flag_2 = False
        # intervals_2 = [[-np.inf, np.inf]]
        # 

        intervals = [[-np.inf, np.inf]]            
        l_a = 0
        l_b = 0
        l_c = 0
        for i, (model, (train_idx, val_idx)) in enumerate(zip(self.list_models[self.best_lam], self.train_val_pairs)):
            a_train, b_train = a[train_idx,:], b[train_idx,:]
            temp = model.si(a_train, b_train)
            intervals = intersect(intervals, temp)
            
            a_val, b_val = a[val_idx,:], b[val_idx,:]
                
            p = model.p
            l0 = (model.mat2 @ model.vec1)[0:p,:]
            l1 = (model.mat2 @ model.vec2)[0:p,:]
            Xl0 = self.X_val[i] @ l0
            Xl1 = self.X_val[i] @ l1

            l_a += Xl1.T @ (Xl1 - 2 * b_val)
            l_b += 2 * (Xl0.T @ Xl1 - Xl0.T @ b_val - Xl1.T @ a_val)
            l_c += Xl0.T @ (Xl0 - 2 * a_val)

        l_a /= self.n_splits
        l_b /= self.n_splits
        l_c /= self.n_splits

        for lam in self.list_lambda:
            if lam == self.best_lam:
                continue

            r_a = 0
            r_b = 0
            r_c = 0
            for i, (model, (train_idx, val_idx)) in enumerate(zip(self.list_models[lam], self.train_val_pairs)):
                a_train, b_train = a[train_idx,:], b[train_idx,:]
                temp = model.si(a_train, b_train)
                intervals = intersect(intervals, temp)
                
                a_val, b_val = a[val_idx,:], b[val_idx,:]
                
                p = model.p
                l0 = (model.mat2 @ model.vec1)[0:p,:]
                l1 = (model.mat2 @ model.vec2)[0:p,:]
                Xl0 = self.X_val[i] @ l0
                Xl1 = self.X_val[i] @ l1

                r_a += Xl1.T @ (Xl1 - 2 * b_val)
                r_b += 2 * (Xl0.T @ Xl1 - Xl0.T @ b_val - Xl1.T @ a_val)
                r_c += Xl0.T @ (Xl0 - 2 * a_val)

            r_a /= self.n_splits
            r_b /= self.n_splits
            r_c /= self.n_splits

            fa = (l_a - r_a)[0,0]
            sa = (l_b - r_b)[0,0]
            ta = (l_c - r_c)[0,0]
            temp = solve_quadratic_inequality(fa, sa, ta)
            intervals = intersect(intervals, temp)

        return intervals