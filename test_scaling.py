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
sylva200 = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=200)
sylva400 = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=400)
sylva800 = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=800)
sylva1600 = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=1600)
sylva2400 = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=2400)

sylva_u = map_target(sylva, sylva_yname, 1)
sylva200_u = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=200, scaler=None)
sylva400_u = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=400, scaler=None)
sylva800_u = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=800, scaler=None)
sylva1600_u = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=1600, scaler=None)
sylva2400_u = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=2400, scaler=None)



d1 = [sylva200, sylva400, sylva800, sylva1600, sylva2400]
d2 = [sylva_u, sylva200_u, sylva800_u, sylva1600_u, sylva2400_u]

for d,D in zip(d1,d2):
    exp_man.add_experiment(d, DummyGenerator(), DummyMetaModel(), c.discovery_algs["best-interval"], name="dummy_dummy" + d.name, new_samples=5000, fragment_limit=20, enable_probabilities=True)
    exp_man.add_experiment(d, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["best-interval"], name="kde_classRF-prob" + d.name, new_samples=5000, fragment_limit=20, enable_probabilities=True)
    exp_man.add_experiment(D, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["best-interval"], name="kde_classRF-prob-unscaled" + d.name, new_samples=5000, fragment_limit=20, enable_probabilities=True)

#exp_man.add_experiment(avila200, PerfectGenerator(), PerfectMetamodel(), c.discovery_algs["best-interval"], name="perfect_perfect_" + "avila200", new_samples=5000, fragment_limit=2, enable_probabilities=False)

res = exp_man.run_all_parallel(15)
exp_man.export_experiments("scaling_test")


# The configuration files (src/experiments/XXXXConfig.py) contain configured generators and metamodels.
# New samples is the amount of samples added, enable_probabilities specifies wheter the classification should be done
# using probabilities, the fragment_limit specifies the maximum number of fragments (small datasets).

# exp_man.add_experiment(avila200, c.generators["kde"], c.metamodels["classRF"], c.discovery_algs["dssd"],
#                        name="kde_classRF_avila", new_samples=10000, fragment_limit=1, enable_probabilities=True)


# res = exp_man.run_all()
# exp_man.export_experiments("prelim_refine")



