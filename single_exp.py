import src.main.experiments.ExperimentManager as u
from generators.RandomSamples import NoiseGenerator
from src.main.experiments.config.DataUtils import map_target, generate_names
from src.main.generators.KernelDensityCV import KernelDensityCV, KernelDensityBW, bw_method_silverman
from src.main.experiments.model.ExperimentDataset import ExperimentDataset
import src.main.experiments.config.Config as c
from src.main.generators.DummyGenerator import DummyGenerator
from src.main.metamodels.DummyMetamodel import DummyMetaModel
from src.main.experiments.model.Experiment import ZERO_ONE_SCALING, Z_SCORE_SCALING
import pandas as pd

exp_man = u.ExperimentManager()

# # jm1 21 attribute
jm1 = pd.read_csv("resources/data/jm1.csv", na_values=['?'])
jm1_yname = "defects"
jm1 = jm1.dropna()
jm1.rename(columns={'v(g)': 'v_g',
                    'ev(g)': 'ev_g',
                    'iv(g)': 'iv_g'},
           inplace=True)
jm1 = map_target(jm1, jm1_yname, True)
jm1s = []
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=200))
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=400))
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=800))
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=1600))

# # occupancy 6 attributes. the data column is changed to the start of the day
occupancy = pd.read_csv("resources/data/cleaned/occupancy.csv", na_values=['?'])
occupancy_yname = "Occupancy"
occupancy = occupancy.dropna()
occupancy = map_target(occupancy, occupancy_yname, 1)
occupancies = []
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=200))
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=400))
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=800))
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=1600))

# # avila 10 attributes
avila = pd.read_csv("resources/data/avila/avila.txt", header=0, names=generate_names(10))
avila_yname = "y"
avila = avila.dropna()
avila = map_target(avila, avila_yname, "A")
avilas = []
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=200))
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=400))
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=800))
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=1600))

# # clean 2 168
clean2 = pd.read_csv("resources/data/clean2.tsv", sep="\t")
clean2_yname = "target"
clean2 = clean2.dropna()
#clean2 = clean2.drop(columns=["molecule_name", "conformation_name"])
# nomapping
cleans = []
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=200))
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=400))
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=800))
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=1600))

datasets = [occupancies, jm1s, cleans, avilas]
kde = KernelDensityBW(bw_method_silverman, hard_limits=True, sampling_multiplier=100)
kde_base = KernelDensityBW(bw_method_silverman, hard_limits=False)

for d in datasets:
    for sd in d:
        exp_man.add_experiment(sd, DummyGenerator(), DummyMetaModel(), c.discovery_algs["prim"], name="dummy_dummy_prim_" + sd.name, new_samples=2500, fragment_limit=30, enable_probabilities=True, scaling=ZERO_ONE_SCALING)
        exp_man.add_experiment(sd, NoiseGenerator(), DummyMetaModel(), c.discovery_algs["prim"], name="kde-cv_classRF_prim_" + sd.name, new_samples=2500, fragment_limit=30, enable_probabilities=True, scaling=Z_SCORE_SCALING)
exp_man.run_all_parallel(32)
exp_man.export_experiments("noise")
exp_man.reset_experiments()
