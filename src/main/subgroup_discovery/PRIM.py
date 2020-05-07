import numpy as np
import pandas as pd
from ema_workbench.analysis import prim as p, prim_util as pu


class PRIM:

    def __init__(self, threshold=1, mass_min=0.05):
        self.threshold = threshold
        self.mass_min = mass_min

    def find(self, X: pd.DataFrame, y: np.ndarray, regression=True):
        if regression:
            loc_mode = p.sdutil.RuleInductionType.REGRESSION
        else:
            loc_mode = p.sdutil.RuleInductionType.BINARY

        prim = p.Prim(x=X, y=y, threshold=self.threshold, mode=loc_mode, obj_function=pu.PRIMObjectiveFunctions.ORIGINAL, mass_min=self.mass_min)
        box_pred = prim.find_box()

        return box_pred.box_lims

