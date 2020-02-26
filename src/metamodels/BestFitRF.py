import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


@DeprecationWarning
class BestFitRF(RandomForestClassifier):
    def __init__(self, estimators_list):
        assert estimators_list is not None and len(estimators_list) > 0
        super().__init__()
        self.n_estimators = None
        self.estimators_list = estimators_list

    def fit(self, X, y, **kwargs):
        assert len(self.estimators_list) > 0
        best_e = 1

        best_score = -np.inf
        for e in self.estimators_list:
            rf = RandomForestClassifier(n_estimators=e)

            current_score = np.mean(cross_val_score(rf, X, y, cv=3))

            if current_score > best_score:
                best_score = current_score
                best_e = e

        self.n_estimators = best_e
        super().fit(X,y)
        return self




