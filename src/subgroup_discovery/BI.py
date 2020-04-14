from typing import Type, List

import numpy as np

import pandas as pd
from rpy2.robjects import pandas2ri  # install any dependency package if you get error like "module not found"
from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from src.experiments.model.FragmentResult import get_initial_restrictions


pandas2ri.activate()
base = importr('base')


class BestInterval:

    def __init__(self, beam_size=1, depth=5):
        self.beam_size = beam_size
        self.depth = depth

    def find(self, X: pd.DataFrame, y: np.ndarray, regression=True) -> List[pd.DataFrame]:
        bi = SignatureTranslatedAnonymousPackage(self.get_rstring(), "bi")
        X_cols = X.columns
        result = bi.beam_refine(X, y, beam_size=self.beam_size, depth=self.depth)
        result = pd.DataFrame(result, columns=X_cols)
        return [get_initial_restrictions(X), result]

    def get_rstring(self):
        return open("src/R/Refinement.R", mode="r").read()
