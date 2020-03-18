import numpy as np
import pandas as pd

from src.experiments.model.Experiment import MalformedExperimentError


class PerfectMetamodel:

    def __init__(self):
        self.y: pd.DataFrame = None
        self.y_complement: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, y_complement: pd.DataFrame, **kwargs):
        self.y_complement = y_complement
        assert y_complement.index is not None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X.shape[0] == len(self.y):
            raise MalformedExperimentError("Dummy can't be used for new datapoints")
        return self.y_complement[X.index].to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise AssertionError("Perfect metamodel can't be used with probabilities. ")

