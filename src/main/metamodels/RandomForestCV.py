import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomForestCV:
    def __init__(self, cv = 5):
        self.rf = None
        self.cv = cv

    def fit(self, X, y):
        m = X.shape[1]
        params = {"max_features": [min(2, m), "sqrt", m]}
        grid = GridSearchCV(RandomForestClassifier(), params, cv = self.cv)
        self.rf = grid.fit(X, y).best_estimator_
        return self

    def predict(self, X):
        return self.rf.predict(X)

    def predict_proba(self, X):
        return self.rf.predict_proba(X)[:, int(np.where(self.rf.classes_ == 1)[0])]


# generated data 

np.random.seed(seed=1)
dx = np.random.random((1000,4))
dy = ((dx > 0.3).sum(axis = 1) == 4) - 0

import time
model = RandomForestCV()
start = time.time()
model.fit(dx,dy)  
end = time.time()
print(end - start)   

sum(model.predict(dx) - dy)
sum(model.predict_proba(dx) - dy)

