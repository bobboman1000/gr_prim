import src.experiments.ExperimentManager as u
from src.experiments.Visualizer import Visualizer


exp_man = u.ExperimentManager("convert")
exp_man.import_experiments("prelim_refine")
v1 = Visualizer(exp_man)
v1.export_all_as_csv()

