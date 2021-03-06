import src.main.experiments.ExperimentManager as u
import numpy as np
from src.main.experiments.config.Config import generators, metamodels, discovery_algs
from src.main.experiments.config.DatasetsConfigNew import large_datasets
from src.main.experiments.model.Experiment import ZERO_ONE_SCALING

exp_man = u.ExperimentManager()

np.random.seed(1)

ds_a = {"prim": discovery_algs["prim"]}


# Choose between ZERO_ONE_SCALING, Z_SCORE_SCALING
#
for d_list in large_datasets:
    exp_man.build_cartesian_experiments(
        datasets=d_list,
        generators=generators,
        metamodels=metamodels,
        discovery_algs=ds_a,
        new_samples=2500,
        fragment_limit=30,
        scaling=ZERO_ONE_SCALING,
        min_support=20
    )

    exp_man.add_dummies(datasets=d_list, metamodels=metamodels, discovery_algs=ds_a, fragment_limit=30, scaling=ZERO_ONE_SCALING, min_support=20)
    exp_man.add_perfects(datasets=d_list, metamodels=metamodels, discovery_algs=ds_a, fragment_limit=30, new_samples=2500, scaling=ZERO_ONE_SCALING,
                         min_support=20)
    res = exp_man.run_all_parallel(32)
    exp_man.export_experiments(d_list[0].name)
    exp_man.reset_experiments()
