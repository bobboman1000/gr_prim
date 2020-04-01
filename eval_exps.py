import warnings

from sklearn.exceptions import ConvergenceWarning

import src.experiments.Util as u
#from src.experiments.DatasetsConfig import datasets
from src.experiments.Visualizer import Visualizer

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)


exp_man = u.ExperimentManager("eval")
#exp_man.import_experiments("dummies")
#exp_man.import_experiments("gaussian-mixtures_m", True)
#exp_man.import_experiments("kde_m", True)
#exp_man.import_experiments("random-normal_m", True)
#exp_man.import_experiments("random-uniform_m", True)
#exp_man.import_experiments("munge_m", True)
size = "600"
v1 = Visualizer(exp_man)


exp_man.import_experiments("prelim_dssd_sylva")
v1.boxplot_all_datasets(metric=v1.highest_wracc_metric, mode=1, display_range=(0.02, 0.07))

