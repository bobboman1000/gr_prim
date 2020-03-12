import numpy as np
import pandas as pd


class PerfectMetamodel:

    def __init__(self):
        self.y: pd.DataFrame = None
        self.known_ys: pd.DataFrame = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, known_ys: pd.DataFrame, **kwargs):
        self.known_ys = known_ys
        assert known_ys.index is not None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            assert X.shape[0] == len(self.y)
        except AssertionError as e:
            e.args += "Dummy can't be used for new datapoints"
            raise
        return self.known_ys[X.index].to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise AssertionError("Perfect metamodel can't be used with probabilities. ")

