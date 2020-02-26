from typing import Type

import numpy as np
import pandas as pd
from ema_workbench.analysis import prim as p, prim_util as pu

import math

import pandas as pd
from rpy2.robjects import pandas2ri  # install any dependency package if you get error like "module not found"
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


pandas2ri.activate()
base = importr('base')


class BestInterval:

    def __init__(self):
        pass

    def find(self, X: pd.DataFrame, y: np.ndarray, regression=True):
        sdmap = SignatureTranslatedAnonymousPackage(self.get_rstring(), "sdmap")
        X_cols = X.columns
        result = sdmap.beam_refine(X, y)
        result = pd.DataFrame(result, columns=X_cols)
        return [self.inital_restrictions(X), result]

    def inital_restrictions(self, X: pd.DataFrame):
        # FIXME make it more pretty - this only works with 0-1 scaling
        X_cols = X.columns
        maximums = np.repeat(1, len(X_cols))
        minimums = np.repeat(0, len(X_cols))
        return pd.DataFrame([minimums, maximums], columns=X_cols)

    def get_rstring(self):
        return open("src/R/Refinement.R", mode="r").read()
