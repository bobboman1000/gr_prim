from src.main.experiments.config.DataUtils import map_target
import src.main.experiments.ExperimentManager as u
from src.main.experiments.model.Experiment import ZERO_ONE_SCALING
from src.main.experiments.model.ExperimentDataset import ExperimentDataset
import src.main.experiments.config.Config as c
from src.main.generators.DummyGenerator import DummyGenerator
from src.main.metamodels.DummyMetamodel import DummyMetaModel
import pandas as pd

exp_man = u.ExperimentManager()

# # sylva 21
sylva = pd.read_csv("resources/data/cleaned/sylva.csv", header=0)
sylva_yname = "label"
sylva = sylva.dropna()
sylva = map_target(sylva, sylva_yname, 1)
sylvas = []
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=200))
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=400))
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=800))
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=1600))

'''
for d in sylvas:
    exp_man.add_experiment(d, DummyGenerator(), DummyMetaModel(), c.discovery_algs["best-interval"], name="dummy_dummy_BI_" + d.name, new_samples=10000, fragment_limit=30, enable_probabilities=True)
    exp_man.add_experiment(d, DummyGenerator(), c.metamodels["classRF"], c.discovery_algs["best-interval"], name="dummy_classRF-prob_BI_" + d.name, new_samples=10000, fragment_limit=30, enable_probabilities=True)
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["best-interval"], name="kde_classRF-prob_BI_" + d.name, new_samples=10000, fragment_limit=30, enable_probabilities=True)

res = exp_man.run_all_parallel(15)
exp_man.export_experiments("sylva_bi")
'''

for d in sylvas:
    exp_man.add_experiment(d, DummyGenerator(), DummyMetaModel(), c.discovery_algs["bid20"], name="dummy_dummy_bid20_" + d.name, new_samples=5000, fragment_limit=30, enable_probabilities=True, scaling=ZERO_ONE_SCALING)
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["bid20"], name="kde_classRF-prob_bid20_" + d.name, new_samples=5000, fragment_limit=30, enable_probabilities=True, scaling=ZERO_ONE_SCALING)

res = exp_man.run_all_parallel(8)
exp_man.export_experiments("sylva_bid20")



