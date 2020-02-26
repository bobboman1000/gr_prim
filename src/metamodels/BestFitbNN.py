import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


@DeprecationWarning
class BestFitbNN(BaggingClassifier):
    def __init__(self, n_estimators_list, n_neighbours_list):
        self.n_neighbours_list = n_neighbours_list
        self.n_estimators_list = n_estimators_list
        self.n_estimators = None
        self.base_estimator = None
        assert n_estimators_list is not None and len(n_estimators_list) > 0
        super().__init__()

    def fit(self, X, y, **kwargs):
        assert len(self.n_estimators_list) > 0
        best_e = 1
        best_bnn = None

        best_score = -np.inf
        for e in self.n_estimators_list:
            for nn in self.n_estimators_list:
                bNN = BaggingClassifier(KNeighborsClassifier(n_neighbors=nn, metric='manhattan', algorithm='ball_tree'), n_estimators=e)
                current_score = np.mean(cross_val_score(bNN, X, y, cv=3))

                if current_score > best_score:
                    best_score = current_score
                    best_e = e
                    best_bnn = bNN
        assert best_bnn is not None
        self.n_estimators = best_e
        self.base_estimator = best_bnn
        super().fit(X,y)
        return self




