import warnings

from sklearn.exceptions import ConvergenceWarning

import src.experiments.Util as u
from src.experiments.DatasetsConfig import large_datasets
from src.experiments.Config import generators, metamodels, discovery_algs , generator_samples, fragment_limit
import gc

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

exp_man = u.ExperimentManager('app2.log')


for generator in generators:
    exp_man.build_cartesian_experiments(large_datasets, {generator: generators[generator]}, metamodels, discovery_algs, new_samples=generator_samples,
                                        enable_probabilities=True, fragment_limit=fragment_limit)
    res = exp_man.run_thread_per_dataset()
    exp_man.export_experiments(generator + str("_m"))
    exp_man.reset_experiments()
    gc.collect()
