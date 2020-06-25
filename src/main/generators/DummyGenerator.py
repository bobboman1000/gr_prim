import pandas as pd


class DummyGenerator:
    def __init__(self):
        self.data = None

    def fit(self, X):
        self.data = X
        return self

    def sample(self, n_samples=1) -> pd.DataFrame:
        return pd.DataFrame(columns=self.data.columns)


# =============================================================================
# This generator creates an empty dataset.
# That is, a dataset augmenter with DummyGenerator coincides with 
# original dataset
# 
# df = pd.read_csv("testdata.csv")
# x = DummyGenerator()
# x.fit(df)
# df1 = x.sample()
# =============================================================================

