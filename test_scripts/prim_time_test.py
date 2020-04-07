import os
import time

import pandas as pd
from ema_workbench.analysis import prim as p, prim_util as pu

def map_target(data: pd.DataFrame, y_col_name: str, to_ones):
    data.loc[:, y_col_name] = data.loc[:, y_col_name].map(lambda e: 1 if e == to_ones else 0)
    return data

os.chdir("../../")

# gamma telescope
gamma_telescope_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha",
                         "fDist", "class"]
gamma_telescope = pd.read_csv("../resources/data/gammatelescope/magic04.data", names=gamma_telescope_names)
gamma_telescope_yname = "class"
gamma_telescope = gamma_telescope.dropna()
bank_marketing = map_target(gamma_telescope, gamma_telescope_yname, "g")

y = gamma_telescope[gamma_telescope_yname]
x = gamma_telescope.drop(columns=[gamma_telescope_yname])

t = time.time()


prim = p.Prim(x=x, y=y, threshold=1, mode=p.sdutil.RuleInductionType.BINARY,
              obj_function=pu.PRIMObjectiveFunctions.ORIGINAL)


print(time.time() - t)