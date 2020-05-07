import numpy as np
import pandas as pd

from src.main.experiments.model.Exceptions import MalformedExperimentError


class PerfectMetamodel:

    def __init__(self):
        self.complete_y = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, y_complement: pd.DataFrame, **kwargs):
        self.complete_y = y.append(y_complement)
        assert y_complement.index is not None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.complete_y[X.index] is None:
            raise MalformedExperimentError("Dummy can't be used for new datapoints")
        return self.complete_y[X.index].to_numpy(dtype=np.float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise AssertionError("Perfect metamodel can't be used with probabilities. ")

