import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import cross_val_score


@DeprecationWarning
class BestFitSVC(svm.SVC):
    def __init__(self, c_list, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):
        assert c_list is not None and len(c_list) > 0
        super().__init__(0, kernel, degree, gamma, coef0, shrinking, probability, tol, cache_size, class_weight, verbose, max_iter, decision_function_shape,
                         random_state)
        self.C = None
        self.c_list = c_list

    def fit(self, X, y, **kwargs):
        assert len(self.c_list) > 0

        best_c = 0
        best_score = -np.inf
        for c in self.c_list:
            svc = svm.SVC(C=c, max_iter=10000, gamma='auto', verbose=self.verbose, shrinking=self.shrinking, coef0=self.coef0, cache_size=self.cache_size,
                          tol=self.tol, decision_function_shape=self.decision_function_shape, degree=self.degree, kernel=self.kernel,
                          class_weight=self.class_weight, probability=self.probability)

            current_score = np.mean(cross_val_score(svc, X, y, cv=3))

            if current_score > best_score:
                best_score = current_score
                best_c = c
        self.C = best_c
        super().fit(X, y)
        return self




