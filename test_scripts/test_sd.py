import pandas as pd
import src.experiments.ExperimentManager as u
from src.experiments.model.ExperimentDataset import ExperimentDataset
from src.experiments.config.DataUtils import map_target, clean, generate_names
import time as t

from src.subgroup_discovery.BI import BestInterval

exp_man = u.ExperimentManager()

gamma_telescope_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha",
                         "fDist", "class"]
gamma_telescope = pd.read_csv("../resources/data/gammatelescope/magic04.data", names=gamma_telescope_names)
gamma_telescope_yname = "class"
gamma_telescope = gamma_telescope.dropna()
gamma_telescope = map_target(gamma_telescope, gamma_telescope_yname, "g")
gamma_telescope = ExperimentDataset("gamma_telescope", gamma_telescope, gamma_telescope_yname, fragment_size=400)

print("Dataset loaded")
# Maps column "y" to 1 if y = 1, 0 else

# Create ExperimentDataset and fragment size. Scales to [0,1] targets by default - if set scaler to None.

X = gamma_telescope.get_subset_compound(0).fragment
y = gamma_telescope.get_subset_compound(0).fragment_y

print("Start sdmap")
time = t.time()
sd = BestInterval()
result = sd.find(X, y)
print("Execution time " + str(t.time() - time))












import src.experiments.config.ConfigTest as c
import pandas as pd
import src.experiments.ExperimentManager as u
from src.experiments.model.ExperimentDataset import ExperimentDataset
from src.generators.DummyGenerator import DummyGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel


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

sylva = pd.read_csv("../resources/data/sylva_prior.csv", header=0)
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
sylva200 = ExperimentDataset("sylva200", sylva, sylva_yname, fragment_size=200)
sylva400 = ExperimentDataset("sylva400", sylva, sylva_yname, fragment_size=400)
sylva800 = ExperimentDataset("sylva800", sylva, sylva_yname, fragment_size=800)
sylva1600 = ExperimentDataset("sylva1600", sylva, sylva_yname, fragment_size=1600)
sylva2400 = ExperimentDataset("sylva2400", sylva, sylva_yname, fragment_size=2400)

avila = pd.read_csv("../resources/data/avila/avila.txt", header=0, names=generate_names(10))
avila_yname = "y"
avila = avila.dropna()
avila = map_target(avila, avila_yname, "A")
avila200 = ExperimentDataset("avila200", avila, avila_yname, fragment_size=200)
avila400 = ExperimentDataset("avila400", avila, avila_yname, fragment_size=400)
avila800 = ExperimentDataset("avil800", avila, avila_yname, fragment_size=800)
avila1600 = ExperimentDataset("avila1600", avila, avila_yname, fragment_size=1600)
avila2400 = ExperimentDataset("avila2400", avila, avila_yname, fragment_size=2400)

datasets = [avila200, avila400] + [sylva200, sylva400]

sizes = [200, 400]

for d in datasets:
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["dssd"], name="kde_classRF-prob_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=True)
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["dssd"], name="kde_classRF_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=False)
    exp_man.add_experiment(d, DummyGenerator(), c.metamodels["classRF"], c.discovery_algs["dssd"], name="dummy_classRF-prob_" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=True)
    exp_man.add_experiment(d, DummyGenerator(), DummyMetaModel(), c.discovery_algs["dssd"], name="dummy_dummy_avila400" + d.name, new_samples=10000, fragment_limit=50, enable_probabilities=False)

res = exp_man.run_all_parallel(4)
exp_man.export_experiments("prelim_dssd")


# The configuration files (src/experiments/XXXXConfig.py) contain configured generators and metamodels.
# New samples is the amount of samples added, enable_probabilities specifies wheter the classification should be done
# using probabilities, the fragment_limit specifies the maximum number of fragments (small datasets).
# exp_man.add_experiment(avila200, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["dssd"],
#                        name="kde_classRF_avila", new_samples=10000, fragment_limit=1, enable_probabilities=True)


# res = exp_man.run_all()
# exp_man.export_experiments("prelim_refine")



