import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestCV:
    def __init__(self, cv=5, iid=True):
        self.rf: RandomForestClassifier = None
        self.cv = cv
        self.iid = iid

    def fit(self, X, y, **kwargs):
        m = X.shape[1]

        params = {"n_estimators": [2, np.sqrt(m), m]}
        grid = GridSearchCV(RandomForestClassifier(), params, cv=self.cv, iid=self.iid).best_estimator_
        self.rf = grid.fit(X, y)
        return self

    def predict(self, X):
        return self.rf.predict(X)

    def predict_proba(self, X):
        return self.rf.predict_proba(X)






