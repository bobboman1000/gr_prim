import src.experiments.ConfigTest as c
import src.experiments.BaselineConfig as bc
import pandas as pd
import src.experiments.Util as u
from src.experiments.model.ExperimentDataset import ExperimentDataset
from src.generators.DummyGenerator import DummyGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel
from src.subgroup_discovery.dssd import DSSD

exp_man = u.ExperimentManager()

def map_target(data: pd.DataFrame, y_col_name: str, to_ones):
    data.loc[:, y_col_name] = data.loc[:, y_col_name].map(lambda e: 1 if e == to_ones else 0)
    return data


def clean(data: pd.DataFrame, value):
    for row in range(data.shape[0]):
        if data.loc[row,:].isin([value]).any():
            data = data.drop(index=row)
    return data


def generate_names(number):
    numbers = range(1,number + 1)
    x_names = list(map(lambda num: "X" + str(num), numbers))
    x_names.append("y")
    return x_names


SAAC2 = pd.read_csv("resources/data/SAAC2.csv", na_values=['?'])
SAAC2_yname = "class"
SAAC2 = SAAC2.dropna()
SAAC2 = map_target(SAAC2, SAAC2_yname, 2)

SAAC2200 = ExperimentDataset("SAAC2200", SAAC2, SAAC2_yname, fragment_size=200)
SAAC2400 = ExperimentDataset("SAAC2400", SAAC2, SAAC2_yname, fragment_size=400)

datasets = [SAAC2200]
var_SAAC2 = DSSD("dssd")
sizes = [200]

for d in datasets:
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], var_SAAC2, name="kde_classRF-prob_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=True)
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], var_SAAC2, name="kde_classRF_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=False)
    exp_man.add_experiment(d, DummyGenerator(), c.metamodels["classRF"], var_SAAC2, name="dummy_classRF-prob_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=True)
    exp_man.add_experiment(d, DummyGenerator(), DummyMetaModel(), var_SAAC2, name="dummy_dummy_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=False)

res = exp_man.run_all()
exp_man.export_experiments("prelim_dssd_SAAC2")



