from typing import Type, List

import numpy as np
import pandas as pd
from ema_workbench.analysis import prim as p, prim_util as pu

import math

import pandas as pd
from rpy2.robjects import pandas2ri  # install any dependency package if you get error like "module not found"
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from src.experiments.model.FragmentResult import get_initial_restrictions


pandas2ri.activate()
base = importr('base')


class BestInterval:

    def __init__(self, beam_size=1):
        self.beam_size = beam_size

    def find(self, X: pd.DataFrame, y: np.ndarray, regression=True) -> List[pd.DataFrame]:
        sdmap = SignatureTranslatedAnonymousPackage(self.get_rstring(), "sdmap")
        X_cols = X.columns
        result = sdmap.beam_refine(X, y, beam_size=self.beam_size)
        result = pd.DataFrame(result, columns=X_cols)
        return [get_initial_restrictions(X), result]

    def get_rstring(self):
        return open("src/R/Refinement.R", mode="r").read()
