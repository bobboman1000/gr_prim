from typing import List

import src.experiments.ExperimentManager as u
from src.experiments.Visualizer import Visualizer
import pandas as pd

from src.experiments.model.Experiment import Experiment


# these files are read from output
names = ["avila", "credit-cards", "eeg-eye-state", "gamma-telescope", "htru", "jm1", "mozilla", "occupancy", "ring", "shuttle"]


def filter_size(experiments: List[Experiment], n: int):
    return list(filter(lambda e: e.name.find("kde") >= 0, experiments))

exp_man = u.ExperimentManager("bw")
for name in names:
    exp_man.import_experiments(name, True)
    exps = filter_size(exp_man.experiments, 0)
    exp_man.experiments = exps
    exp_man._update_queues()

rows = []
for e in exp_man.experiments:
    rows.append([e.ex_data.name, e.ex_data.fragment_size, e.generator.kde.bandwidth])
df = pd.DataFrame(rows, columns=["name", "f_size", "bandwidth"])
df.to_csv("output/bandwidths.csv")


