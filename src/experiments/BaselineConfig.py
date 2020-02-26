from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.generators.RandomSamples import *
from src.metamodels.NNProbababilityEstimation import *
from src.subgroup_discovery.PRIM import PRIM

enable_probabilities = True
generator_samples = 50000
fragment_limit = 30

generators = {
    "random-uniform": UniformRandomSamplesGenerator(),
    "random-normal": NormalRandomSampleGenerator(),
    #"point-s": PointShift(local_variance=1),
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
