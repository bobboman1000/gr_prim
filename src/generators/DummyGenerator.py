import pandas as pd


class DummyGenerator:
    def __init__(self):
        self.data = None

    def fit(self, X, **kwargs):
        self.data = X
        return self

    def sample(self, n_samples=1) -> pd.DataFrame:
        return pd.DataFrame(columns=self.data.columns)
