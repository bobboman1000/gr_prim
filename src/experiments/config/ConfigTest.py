from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

#from src.subgroup_discovery.BI import BestInterval
from src.subgroup_discovery.BI import BestInterval
from src.subgroup_discovery.dssd import DSSD, Config
import src.generators.GaussianMixtures as mix
from src.generators.KernelDensityCV import KernelDensityBW, bw_method_scott
from src.metamodels.NNProbababilityEstimation import *
from src.subgroup_discovery.PRIM import PRIM

generators = {
    "kde": KernelDensityBW(bw_method_scott)
}

cv = 3

classRF_cv_params = { "n_estimators": [30, 90] }
cv_classRF = GridSearchCV(estimator=RandomForestClassifier(), param_grid=classRF_cv_params, cv=cv, iid=True)

metamodels = {
    "classRF": cv_classRF,
}



discovery_algs = {
    "prim": PRIM(threshold=1, mass_min=20),
    "best-interval": BestInterval(),
    "best-interval-b5": BestInterval(beam_size=5),
    "dssd": DSSD("dssd")
}
