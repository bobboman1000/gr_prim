import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RF:
    def __init__(self, params = {"max_features": [2, "sqrt", None]}, cv = 5):
        self.params_ = params
        self.cv_ = cv

    def fit(self, X, y):
        self.model_ = GridSearchCV(RandomForestClassifier(), self.params_, cv = self.cv_).fit(X, y).best_estimator_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]


# =============================================================================
# # TEST 
# 
# import matplotlib.pyplot as plt
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 500)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
# y = np.hstack((np.zeros(500), np.ones(500)))
# plt.scatter(x[:,0], x[:,1], c = y)
# 
# rf = RF()
# rf.fit(x, y)
# sum(abs(rf.predict(x) - y)) # sometimes this overfits
# sum(abs(rf.predict_proba(x) - y))
# 
# from src.main.generators.GaussianMixtures import GMMBIC
# gmm = GMMBIC()
# gmm.fit(x)
# df = gmm.sample(n_samples = 201)
# ynew = rf.predict(df)
# plt.scatter(df[:,0], df[:,1], c = ynew)
# =============================================================================

