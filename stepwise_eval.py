import warnings

from sklearn.exceptions import ConvergenceWarning
import src.experiments.ExperimentManager as u
#from src.experiments.DatasetsConfig import datasets
from src.experiments.Visualizer import Visualizer

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


exp_man = u.ExperimentManager("eval_steps")
exp_man.import_experiments("avila_steps")


v1 = Visualizer(exp_man, detailed_mode=True)
v1.plot_detailed_curve(metric=v1.highest_f1_metric,
                       selected_methods=["dummy_dummy", "dummy_classRF", "kde_classRF"],
                                colors=["darkgrey",         "grey",        "blue"],
                       steps_range=[300, 600, 1200, 2400], title="avila", name="avila_curve",
                       w_variance=True)

#fix(exp_man.experiments)
#exp_man.export_experiments(set)