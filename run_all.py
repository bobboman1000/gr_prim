import src.experiments.config.Config as c
import pandas as pd
import src.experiments.ExperimentManager as u
from src.experiments.model.ExperimentDataset import ExperimentDataset
from src.generators.DummyGenerator import DummyGenerator
from src.generators.PerfectGenerator import PerfectGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel
from src.metamodels.PerfectMetamodel import PerfectMetamodel
import numpy as np
from src.experiments.config.Config import generators, metamodels, discovery_algs
from src.experiments.config.DatasetsConfigNew import large_datasets

exp_man = u.ExperimentManager()

np.random.seed(1)

for d_list in large_datasets:
    exp_man.build_cartesian_experiments(
        datasets=d_list,
        generators=c.generators,
        metamodels=c.metamodels,
        discovery_algs=c.discovery_algs,
        new_samples=10000,
        fragment_limit=30
    )
    exp_man.add_dummies(datasets=d_list, metamodels=c.metamodels, discovery_algs=c.discovery_algs, fragment_limit=30)
    for d in d_list:
        exp_man.add_experiment(dataset=d, generator=PerfectGenerator(), metamodel=PerfectMetamodel(), discovery_alg=c.discovery_algs["prim"],
                           name="perfect_perfect_prim_" + d.name, fragment_limit=30, new_samples=10000, enable_probabilities=False)
    res = exp_man.run_all_parallel(32)
    exp_man.export_experiments(d_list[0].name)
    exp_man.reset_experiments()





