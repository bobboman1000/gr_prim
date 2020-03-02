import warnings

from sklearn.exceptions import ConvergenceWarning

import src.experiments.Util as u
from src.experiments.Config import generators, metamodels, discovery_algs
from src.experiments.BaselineConfig import generators as bgenerator
import gc
import pandas as pd

from src.experiments.model.ExperimentDataset import ExperimentDataset
from src.generators.DummyGenerator import DummyGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

exp_man = u.ExperimentManager('app2.log')

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



# # electricity
electricity = pd.read_csv("resources/data/electricity-normalized.csv")
electricity_yname = "class"
electricity = electricity.dropna()
map_target(electricity, electricity_yname, "UP")

click = pd.read_csv("resources/data/click_prediction.csv")
click_yname = "click"
click = click.dropna()

# sylva
sylva = pd.read_csv("resources/data/sylva_prior.csv", header=0)
sylva_yname = "label"
sylva = sylva.dropna()
sylva = sylva.drop(columns=["Rawah_Wilderness_Area","Neota_Wilderness_Area","Comanche_Peak_Wilderness_Area","Cache_la_Poudre_Wilderness_Area",
                            "Soil_Type_1","Soil_Type_2","Soil_Type_3","Soil_Type_4","Soil_Type_5","Soil_Type_6","Soil_Type_7","Soil_Type_8",
                            "Soil_Type_9","Soil_Type_10","Soil_Type_11","Soil_Type_12","Soil_Type_13","Soil_Type_14","Soil_Type_15","Soil_Type_16",
                            "Soil_Type_17","Soil_Type_18","Soil_Type_19","Soil_Type_20","Soil_Type_21","Soil_Type_22","Soil_Type_23","Soil_Type_24",
                            "Soil_Type_25","Soil_Type_26","Soil_Type_27","Soil_Type_28","Soil_Type_29","Soil_Type_30","Soil_Type_31","Soil_Type_32",
                            "Soil_Type_33","Soil_Type_34","Soil_Type_35","Soil_Type_36","Soil_Type_37","Soil_Type_38","Soil_Type_39","Soil_Type_40",
                            "dup_Rawah_Wilderness_Area","dup_Neota_Wilderness_Area","dup_Comanche_Peak_Wilderness_Area",
                            "dup_Cache_la_Poudre_Wilderness_Area","dup_Soil_Type_1","dup_Soil_Type_2","dup_Soil_Type_3","dup_Soil_Type_4",
                            "dup_Soil_Type_5","dup_Soil_Type_6","dup_Soil_Type_7","dup_Soil_Type_8","dup_Soil_Type_9","dup_Soil_Type_10",
                            "dup_Soil_Type_11","dup_Soil_Type_12","dup_Soil_Type_13","dup_Soil_Type_14","dup_Soil_Type_15","dup_Soil_Type_16",
                            "dup_Soil_Type_17","dup_Soil_Type_18","dup_Soil_Type_19","dup_Soil_Type_20","dup_Soil_Type_21","dup_Soil_Type_22",
                            "dup_Soil_Type_23","dup_Soil_Type_24","dup_Soil_Type_25","dup_Soil_Type_26","dup_Soil_Type_27","dup_Soil_Type_28",
                            "dup_Soil_Type_29","dup_Soil_Type_30","dup_Soil_Type_31","dup_Soil_Type_32","dup_Soil_Type_33","dup_Soil_Type_34",
                            "dup_Soil_Type_35","dup_Soil_Type_36","dup_Soil_Type_37","dup_Soil_Type_38","dup_Soil_Type_39","dup_Soil_Type_40"])
sylva = map_target(sylva, sylva_yname, 1)

SAAC2 = pd.read_csv("resources/data/SAAC2.csv", na_values=['?'])
SAAC2_yname = "class"
SAAC2 = SAAC2.dropna()
SAAC2 = map_target(SAAC2, SAAC2_yname, 2)


clean2 = pd.read_csv("resources/data/clean2.tsv", sep="\t")
clean2_yname = "target"
clean2 = clean2.dropna()
clean2 = clean2.drop(columns=["molecule_name", "conformation_name"])
# nomapping


avila = pd.read_csv("resources/data/avila/avila.txt", header=0, names=generate_names(10))
avila_yname = "y"
avila = avila.dropna()
avila = map_target(avila, avila_yname, "A")



for f_size in [300, 600, 1200, 2400]:
    d = ExperimentDataset("avila" + "_" + str(f_size), avila, avila_yname, fragment_size=f_size)
    exp_man.add_experiment(d, generators["kde"], metamodels["classRF"], discovery_algs["prim"], "kde_classRF_" + d.name, new_samples=50000,
                           enable_probabilities=True, fragment_limit=20)
    exp_man.add_experiment(d, DummyGenerator(), metamodels["classRF"], discovery_algs["prim"], "dummy_classRF_" + d.name, new_samples=0,
                           enable_probabilities=True, fragment_limit=20)
    exp_man.add_experiment(d, DummyGenerator(), DummyMetaModel(), discovery_algs["prim"], "dummy_dummy_" + d.name, new_samples=0,
                           enable_probabilities=True, fragment_limit=20)

exp_man.run_all_parallel(24)
exp_man.export_experiments("avila_steps")


