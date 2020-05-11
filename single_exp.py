import src.main.experiments.ExperimentManager as u
from experiments.config.DataUtils import map_target
from generators.KernelDensityCV import KernelDensityCV
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


datasets = [occupancies, jm1s]
kde = KernelDensityCV(bandwidth_list=[0.0001, 0.001, 0.01, 0.1], hard_limits=True, sampling_multiplier=100)
kde_base = KernelDensityCV(bandwidth_list=[0.0001, 0.001, 0.01, 0.1], hard_limits=False)

for d in datasets:
    for sd in d:
        exp_man.add_experiment(sd, DummyGenerator(), DummyMetaModel(), c.discovery_algs["prim"], name="dummy_dummy_prim_" + sd.name, new_samples=2500, fragment_limit=30, enable_probabilities=True, scaling=ZERO_ONE_SCALING)
        exp_man.add_experiment(sd, kde_base, c.metamodels["classRF"], c.discovery_algs["prim"], name="kde-cv_classRF_prim_" + sd.name, new_samples=2500, fragment_limit=30, enable_probabilities=True, scaling=Z_SCORE_SCALING)
        exp_man.add_experiment(sd, kde, c.metamodels["classRF"], c.discovery_algs["prim"], name="kde-cv-hard_classRF_prim_" + sd.name, new_samples=2500, fragment_limit=30, enable_probabilities=True, scaling=Z_SCORE_SCALING)
    exp_man.run_all_parallel(15)
    exp_man.export_experiments(d[0].name)
    exp_man.reset_experiments()
