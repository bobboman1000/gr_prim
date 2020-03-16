import pandas as pd


class PerfectGenerator:
    def __init__(self):
        self.data = None

    def fit(self, X, X_complement: pd.DataFrame, **kwargs):
        self.data = X
        self.X_complement = X_complement
        return self

    def sample(self, n_samples=1) -> pd.DataFrame:
        return self.X_complement.sample(n_samples)