import src.main.experiments.config.Config as c
import pandas as pd
import src.main.experiments.ExperimentManager as u
from src.main.experiments.model.ExperimentDataset import ExperimentDataset
from src.main.generators.DummyGenerator import DummyGenerator
from src.main.metamodels.DummyMetamodel import DummyMetaModel
from src.main.subgroup_discovery.dssd import DSSD

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


electricity = pd.read_csv("resources/data/electricity-normalized.csv")
electricity_yname = "class"
electricity = electricity.dropna()
map_target(electricity, electricity_yname, "UP")

electricity200 = ExperimentDataset("electricity200", electricity, electricity_yname, fragment_size=200)
electricity400 = ExperimentDataset("electricity400", electricity, electricity_yname, fragment_size=400)

datasets = [electricity200, electricity400]
var_electricity = DSSD("dssd")
sizes = [200, 400]

for d in datasets:
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], var_electricity, name="kde_classRF-prob_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=True)
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], var_electricity, name="kde_classRF_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=False)
    exp_man.add_experiment(d, DummyGenerator(), c.metamodels["classRF"], var_electricity, name="dummy_classRF-prob_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=True)
    exp_man.add_experiment(d, DummyGenerator(), DummyMetaModel(), var_electricity, name="dummy_dummy_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=False)

res = exp_man.run_all()
exp_man.export_experiments("prelim_dssd_electricity")



