from sklearn.mixture import GaussianMixture
from sklearn.calibration import CalibratedClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.generators.Munge_R import MUNGE
from src.generators.PerfectGenerator import PerfectGenerator
from src.generators.RandomSamples import NormalRandomSampleGenerator, UniformRandomSamplesGenerator
from src.metamodels.NNProbababilityEstimation import *
from src.metamodels.RandomForestCV import RandomForestCV
from src.subgroup_discovery.PRIM import PRIM


cv = 5
kde_params = {'bandwidth': np.logspace(-1, 1, 20)}
kde_cv = GridSearchCV(KernelDensity(), kde_params)

gmm_paramas = {'n_components': list(range(30)), "covariance_type": ["full", "tied", "diag", "spherical"]}
gmm_cv = GridSearchCV(GaussianMixture(), gmm_paramas)

generators = {
    "perfect": PerfectGenerator(),
    "random-unif": UniformRandomSamplesGenerator(),
    "random-norm": NormalRandomSampleGenerator(),
    "gaussian-mixtures": gmm_cv,
    "kde": kde_cv,
    "munge1": MUNGE(local_var=1),
    "munge0.2": MUNGE(local_var=0.2)
}

# Metamodel Configuration
svc_cv_params = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}
cv_svc = GridSearchCV(estimator=SVC(), param_grid=svc_cv_params, cv=cv, iid=True)

metamodels = {
    "classRF": RandomForestCV(cv=cv),
    "kriging": GaussianProcessClassifier(),
    "neural-net": MLPClassifier(),
    "SVC-calibrated": CalibratedClassifierCV(base_estimator=cv_svc),
    "nb-calibrated": CalibratedClassifierCV(base_estimator=GaussianNB())
}

discovery_algs = {
    "prim": PRIM(threshold=1, mass_min=20)
}
