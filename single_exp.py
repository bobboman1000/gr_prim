import src.main.experiments.ExperimentManager as u
from src.main.experiments.model.ExperimentDataset import ExperimentDataset
import src.main.experiments.config.Config as c
from src.main.generators.DummyGenerator import DummyGenerator
from src.main.metamodels.DummyMetamodel import DummyMetaModel
from src.main.experiments.model.Experiment import ZERO_ONE_SCALING, Z_SCORE_SCALING
import pandas as pd

exp_man = u.ExperimentManager()

bi_datasets = []

clean2 = pd.read_csv("resources/data/clean2.tsv", sep="\t")
clean2_yname = "target"
clean2 = clean2.dropna()
clean2 = clean2.drop(columns=["molecule_name", "conformation_name"])
# nomapping
ex_data = ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=400)


exp_man.add_experiment(ex_data, DummyGenerator(), DummyMetaModel(), c.discovery_algs["prim"], name="dummy_dummy_prim_clean2", new_samples=2500, fragment_limit=30, enable_probabilities=True, scaling=ZERO_ONE_SCALING)
exp_man.add_experiment(ex_data, c.generators["munge1"], c.metamodels["classRF"], c.discovery_algs["prim"], name="kde-si_classRF_prim_clean2", new_samples=2500, fragment_limit=30, enable_probabilities=True, scaling=Z_SCORE_SCALING)

exp_man.run_all_parallel(2)
exp_man.export_experiments("clean2")
