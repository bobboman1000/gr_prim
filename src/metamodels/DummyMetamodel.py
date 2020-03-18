import numpy as np
import pandas as pd

from src.experiments.model.Experiment import MalformedExperimentError


class DummyMetaModel:

    def __init__(self):
        self.y: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs):
        self.y = y
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X.shape[0] == len(self.y):
            raise MalformedExperimentError("Dummy can't be used for new datapoints")
        return self.y.to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if X.shape[0] == len(self.y):
            raise MalformedExperimentError("Dummy can't be used for new datapoints")
        probabilities = np.ndarray(shape=(len(self.y), 2))
        [self._set_row(self.y.to_numpy()[yi], yi, probabilities) for yi in range(len(self.y))]
        return probabilities

    def _set_row(self, column, row, table):
        table[row, column] = 1
        table[row, 1 - column] = 0

