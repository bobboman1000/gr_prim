import warnings

from sklearn.exceptions import ConvergenceWarning
import src.experiments.ExperimentManager as u
#from src.experiments.DatasetsConfig import datasets
from src.experiments.Visualizer import Visualizer

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


exp_man = u.ExperimentManager("eval_steps")
exp_man.import_experiments("prelim_refine_fres")


v1 = Visualizer(exp_man, detailed_mode=True)
"""v1.plot_detailed_curve(metric=v1.highest_wracc_metric,
                       selected_methods=["dummy_dummy", "dummy_dummy-b5", "dummy_classRF-prob", "kde_classRF", "kde_classRF-prob"],
                                colors=["darkgrey",     "red",              "grey",                    "blue",        "green"],
                       steps_range=[200, 400, 800, 1600, 2400], title="sylva", name="sylva_curve",
                       w_variance=True)"""

#fix(exp_man.experiments)
#exp_man.export_experiments(set)