import numpy as np
import pandas as pd
import sklearn.neighbors as nn


class TrepanSampler:
    numeric = [int, float, np.int64]

    def __init__(self, kde_bandwidth: float = 0.2):
        self.kde_bandwidth = kde_bandwidth
        self.df = None

    def fit(self, X: pd.DataFrame, **kwargs):
        self.df = X
        return self

    def sample(self, size: int):
        result = None
        for col_idx in range(self.df.shape[1]):
            new_column = self.__sample_from_column(self.df, col_idx, size)
            old_result = result  # TODO is this necessary. wtf python
            result = pd.concat([old_result, new_column], axis=1)
        return result

    def __sample_from_column(self, df: pd.DataFrame, col_idx: int, size: int) -> pd.DataFrame:
        col_name = df.columns[col_idx]
        if self.numeric.__contains__(type(df.iloc[0, col_idx])):  # TODO Integer values? Maybe round them back
            dist = self.__kde_dist_of_column(df, col_idx)
            samples_dict = {col_name: list(dist.sample(size).flatten())}  # TODO Flattening is necessary, but why?
            new_samples = pd.DataFrame(samples_dict)
        else:
            new_samples = df.iloc[:, col_idx].sample(n=size, replace=True).reset_index(drop=True)
        return new_samples

    # https://scikit-learn.org/stable/modules/density.html
    def __kde_dist_of_column(self, df: pd.DataFrame, col_idx: int, kernel: str = 'gaussian') -> nn.KernelDensity:
        column = pd.DataFrame(df.iloc[:, col_idx])
        return nn.KernelDensity(kernel=kernel, bandwidth=0.2).fit(column)
