from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import src.generators.GaussianMixtures as mix
from src.generators.KernelDensityCV import KernelDensityBW, bw_method_scott
from src.generators.Munge_R import MUNGE
from src.metamodels.NNProbababilityEstimation import *
from src.subgroup_discovery.PRIM import PRIM

generators = {
    "gaussian-mixtures": mix.GaussianMixture(),
    "kde": KernelDensityBW(bw_method_scott),
    "munge": MUNGE(local_var=3)
}

cv = 3

classRF_cv_params = { "n_estimators": [30, 90] }
cv_classRF = GridSearchCV(estimator=RandomForestClassifier(), param_grid=classRF_cv_params, cv=cv, iid=True)

bnn_cv_params = {
    "n_estimators": [30, 90],
    "base_estimator": [KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=5), KNeighborsClassifier(n_neighbors=20)],
}
cv_bnn = GridSearchCV(estimator=BaggingClassifier(), param_grid=bnn_cv_params, cv=cv, iid=True)

svc_cv_params = {"C": [1, 10, 100]}
cv_svc = GridSearchCV(estimator=SVC(gamma='auto'), param_grid=svc_cv_params, cv=cv, iid=True)


metamodels = {
    "classRF": cv_classRF,
    #"bNN": cv_bnn,
    "kriging": GaussianProcessClassifier(),
    "neural-net": MLPClassifier(),
    "SVC-calibrated": CalibratedClassifierCV(base_estimator=cv_svc),
    "nb-calibrated": CalibratedClassifierCV(base_estimator=GaussianNB()),
    "classRF-calibrated": CalibratedClassifierCV(base_estimator=cv_classRF),
}

discovery_algs = {
    "prim": PRIM(threshold=1, mass_min=20)
}
