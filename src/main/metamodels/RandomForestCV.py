import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import uniform, randint


class RF:
    def __init__(self, params = {"max_features": [2, "sqrt", None]}, cv = 5, seed = 2020):
        self.params_ = params
        self.cv_ = cv
        self.seed_ = seed

    def fit(self, X, y):
        self.model_ = GridSearchCV(RandomForestClassifier(random_state = self.seed_), self.params_, cv = self.cv_).fit(X, y).best_estimator_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]



class SVCCV:
    def __init__(self, params = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}, cv = 5, seed = 2020):
        self.params_ = params
        self.cv_ = cv
        self.seed_ = seed

    def fit(self, X, y):
        self.model_ = CalibratedClassifierCV(GridSearchCV(SVC(random_state = self.seed_), self.params_, cv = self.cv_)).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]
    
    

class GP:
    def __init__(self, seed = 2020):
        self.seed_ = seed

    def fit(self, X, y):
        self.model_ = GaussianProcessClassifier(kernel = 1.0 * RBF(1.0), random_state = self.seed_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]



class NB:
    def __init__(self):
        return None

    def fit(self, X, y):
        self.model_ = CalibratedClassifierCV(base_estimator = GaussianNB()).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]
    
   
    
class XGBCV:
    def __init__(self, params = {'n_estimators' : randint(10,990),
        'learning_rate' : uniform(0.0001,0.2),
        'gamma' : uniform(0,0.4),
        'max_depth' : [6],
        'subsample' : uniform(0.5,0.5)}, cv = 5, seed = 2020):
        
        self.params_ = params
        self.cv_ = cv
        self.seed_ = seed

    def fit(self, X, y):
        self.model_ = RandomizedSearchCV(XGBClassifier(nthread = 1, verbosity = 0, use_label_encoder = False), 
                                         self.params_, random_state = self.seed_,
                                         cv = self.cv_, n_iter = 50, n_jobs = 1).fit(X, y).best_estimator_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]

    
    
# TEST 

import matplotlib.pyplot as plt
mean = [0, 0]
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, 500)
mean = [5, 5]
x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
y = np.hstack((np.zeros(500), np.ones(500))).astype(int)
plt.scatter(x[:,0], x[:,1], c = y)

from src.main.generators.GaussianMixtures import GMMBIC
gmm = GMMBIC()
gmm.fit(x)
df = gmm.sample(n_samples = 201)

rf = RF()
rf.fit(x, y)
sum(abs(rf.predict(x) - y))
sum(abs(rf.predict_proba(x) - y))
ynew = rf.predict(df)
plt.scatter(df[:,0], df[:,1], c = ynew)

svc_cv = SVCCV()
svc_cv.fit(x, y)
sum(abs(svc_cv.predict(x) - y))
sum(abs(svc_cv.predict_proba(x) - y))
ynew = svc_cv.predict(df)
plt.scatter(df[:,0], df[:,1], c = ynew)

gp = GP()
gp.fit(x, y)
sum(abs(gp.predict(x) - y)) 
sum(abs(gp.predict_proba(x) - y))
ynew = gp.predict(df)
plt.scatter(df[:,0], df[:,1], c = ynew)

nb = NB()
nb.fit(x, y)
sum(abs(nb.predict(x) - y)) 
sum(abs(nb.predict_proba(x) - y))
ynew = nb.predict(df)
plt.scatter(df[:,0], df[:,1], c = ynew)


xgb = XGBCV()
xgb.fit(x, y)
sum(abs(xgb.predict(x) - y)) 
sum(abs(xgb.predict_proba(x) - y))
ynew = xgb.predict(df)
plt.scatter(df[:,0], df[:,1], c = ynew)

