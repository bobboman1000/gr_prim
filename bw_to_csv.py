from typing import List

import src.main.experiments.ExperimentManager as u
from src.main.experiments.model.Experiment import Experiment


# these files are read from output
names = ["avila"]


def filter_size(experiments: List[Experiment], n: int):
    return list(filter(lambda e: (e.name.find("dummy_classRF") >= 0 or e.name.find("dummy_dummy") >= 0) and e.ex_data.fragment_size == n, experiments))

exp_man = u.ExperimentManager("bw")
for name in names:
    exp_man.import_experiments(name, True)
    exps = filter_size(exp_man.experiments, 800)
    exp_man.experiments = exps
    exp_man._update_queues()




