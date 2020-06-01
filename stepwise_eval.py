import warnings

from sklearn.exceptions import ConvergenceWarning
import src.main.experiments.ExperimentManager as u
#from src.main.experiments.DatasetsConfig import datasets
from src.main.experiments.Visualizer import Visualizer

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

exp_man = u.ExperimentManager("eval_steps")

exp_man.import_experiments("test_wracc_prim")


v1 = Visualizer(exp_man)
"""v1.plot_detailed_curve(metric=v1.peeling_trajectory_auc_metric,
                       selected_methods=["dummy_dummy", "dummy_classRF", "perfect_classRF", "gaussian-mixtures_classRF"],
                       colors=["darkgrey", "steelblue",  "red", "blue"],
                       steps_range=[200, 400, 800, 1600], name="avila", w_variance=True,
                       legend=True)"""

#fix(exp_man.experiments)
#exp_man.export_experiments(set)