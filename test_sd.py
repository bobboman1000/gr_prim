import src.experiments.ConfigTest as c
import src.experiments.BaselineConfig as bc
import pandas as pd
import src.experiments.Util as u
from src.experiments.model.ExperimentDataset import ExperimentDataset
from src.generators.DummyGenerator import DummyGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel
import time as t

from src.subgroup_discovery.BI import BestInterval

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


gamma_telescope_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha",
                         "fDist", "class"]
gamma_telescope = pd.read_csv("resources/data/gammatelescope/magic04.data", names=gamma_telescope_names)
gamma_telescope_yname = "class"
gamma_telescope = gamma_telescope.dropna()
gamma_telescope = map_target(gamma_telescope, gamma_telescope_yname, "g")
gamma_telescope = ExperimentDataset("gamma_telescope", gamma_telescope, gamma_telescope_yname, fragment_size=400)

print("Dataset loaded")
# Maps column "y" to 1 if y = 1, 0 else

# Create ExperimentDataset and fragment size. Scales to [0,1] targets by default - if set scaler to None.

X = gamma_telescope.get_subset_compound(0).complement
y = gamma_telescope.get_subset_compound(0).complement_y

print("Start sdmap")
time = t.time()
sd = BestInterval()
result = sd.find(X, y)
print("Execution time " + str(t.time() - time))