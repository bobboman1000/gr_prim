from typing import List

import src.experiments.ExperimentManager as u
from src.experiments.Visualizer import Visualizer
import pandas as pd

from src.experiments.model.Experiment import Experiment

# these files are read from /ouptut
names = ["avila", "credit-cards", "eeg-eye-state", "gamma-telescope", "htru", "jm1", "mozilla", "occupancy", "ring", "shuttle"]

def filter_stuff(experiments: List[Experiment]):
    return list(filter(lambda e: e.name.find("kde") >= 0, experiments))

exp_man = u.ExperimentManager("convert")

for name in names:
    exp_man.import_experiments(name, True)
    exps = filter_stuff(exp_man.experiments)
    exp_man.experiments = exps
    exp_man._update_queues()
v1 = Visualizer(exp_man)



#data.to_csv("results.csv")


