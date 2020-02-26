import warnings

from sklearn.exceptions import ConvergenceWarning
import src.experiments.Util as u
#from src.experiments.DatasetsConfig import datasets
from src.experiments.Config import metamodels, discovery_algs
from src.experiments.Visualizer import Visualizer

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


exp_man = u.ExperimentManager("eval_steps")
exp_man.import_experiments("clean2_steps")

v1 = Visualizer(exp_man, detailed_mode=True)
v1.plot_detailed_curve(metric=v1.peeling_trajectory_auc_metric, selected_methods=["dummy_dummy", "kde_classRF"],
                       colors=["grey", "blue"], steps_range=range(100,2500,100), title="clean2", name="clean2_curve")

#fix(exp_man.experiments)
#exp_man.export_experiments(set)