from src.generators.GaussianMixtures import GaussianMixtureCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from src.generators.KernelDensityCV import KernelDensityCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.generators.Munge_R import Munge
from src.generators.RandomSamples import NormalRandomSampleGenerator, UniformRandomSamplesGenerator
from src.metamodels.NNProbababilityEstimation import *
from src.metamodels.RandomForestCV import RandomForestCV
from src.subgroup_discovery.BI import BestInterval
from src.subgroup_discovery.PRIM import PRIM

cv = 5

generators = {
    "random-unif": UniformRandomSamplesGenerator(),
    "random-norm": NormalRandomSampleGenerator(),
    "gaussian-mixtures": GaussianMixtureCV(list(range(1,30))),
    "kde": KernelDensityCV(np.linspace(0.1, 1.0, 30)),
    "munge1": Munge(local_var=1),
    "munge0.2": Munge(local_var=0.2)
}

# Metamodel Configuration
svc_cv_params = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]}
cv_svc = GridSearchCV(estimator=SVC(), param_grid=svc_cv_params, cv=cv)

metamodels = {
    "classRF": RandomForestCV(cv=cv),
    "kriging": GaussianProcessClassifier(),
    #"neural-net": MLPClassifier(),
    "SVC-calibrated": CalibratedClassifierCV(base_estimator=cv_svc),
    "nb-calibrated": CalibratedClassifierCV(base_estimator=GaussianNB())
}

discovery_algs = {
    "prim": PRIM(threshold=1, mass_min=20),
    "best-interval": BestInterval()
}
