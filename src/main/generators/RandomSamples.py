import numpy as np
import pandas as pd


class NormalRandomSampleGenerator:

    def __init__(self):
        self.data = None

    def fit(self, X: pd.DataFrame):
        self.data = X
        return self

    def sample(self, size: int) -> pd.DataFrame:
        sample = pd.DataFrame()
        i = 0
        for attr in self.data:
            mu = pd.Series.mean(self.data[attr])
            sigma = pd.Series.std(self.data[attr])
            sample.insert(i, attr, np.random.normal(mu, sigma, size), allow_duplicates=True)
            i += 1
        return sample


class UniformRandomSamplesGenerator:

    def __init__(self):
        self.data = None

    def fit(self, X: pd.DataFrame, **kwargs):
        self.data = X
        return self

    def sample(self, size: int) -> pd.DataFrame:
        sample = pd.DataFrame()
        i = 0
        for col in self.data:
            col_min, col_max = self.__min_max_of_column(self.data, i)
            sample.insert(i, col, self.__random_uniform_series(col_min, col_max, size))
            i += 1
        return sample

    def __random_uniform_series(self, low, high, size):
        return pd.Series(np.random.uniform(low, high, size))

    def __min_max_of_column(self, df: pd.DataFrame, col_idx: int):
        return df.iloc[:, col_idx].min(), df.iloc[:, col_idx].max()



class NoiseGenerator:

    def __init__(self, divisor: float = 100):
        self.data = None
        self.divisor = divisor

    def fit(self, X: pd.DataFrame, **kwargs):
        self.data: pd.DataFrame = X
        return self

    def sample(self, size: int) -> pd.DataFrame:
        mod_data: pd.DataFrame = self.data.copy()
        for col in mod_data:
            col_values = mod_data[col].to_numpy()
            col_values_unique = np.unique(col_values)
            div_mind_dist = min([a - b for a in col_values_unique for b in col_values_unique if b != a])
            div_mind_dist /= self.divisor
            col_values = np.add(col_values, np.random.uniform(0, div_mind_dist, len(col_values)))
            mod_data[col] = col_values
        return mod_data

    def __random_uniform_series(self, low, high, size):
        return pd.Series(np.random.uniform(low, high, size))

    def __min_max_of_column(self, df: pd.DataFrame, col_idx: int):
        return df.iloc[:, col_idx].min(), df.iloc[:, col_idx].max()