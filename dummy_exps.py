import warnings

from sklearn.exceptions import ConvergenceWarning

import src.experiments.ExperimentManager as u
from src.experiments.config.DatasetsConfig import large_datasets
from src.experiments.config.Config import metamodels, discovery_algs, fragment_limit

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

exp_man = u.ExperimentManager()

exp_man.add_dummies(large_datasets, metamodels, discovery_algs, fragment_limit=fragment_limit)
res = exp_man.run_thread_per_dataset()
exp_man.export_experiments("dummies")