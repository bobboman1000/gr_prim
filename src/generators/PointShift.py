import math

import numpy as np
import pandas as pd


class PointShift:

    def __init__(self, local_variance: int):
        self.data: pd.DataFrame = None
        self.local_variance = local_variance
        self.response_column = None

    def fit(self, X, response_column=None, **kwargs):
        self.data = X
        self.response_column = response_column
        return self

    def sample(self, size: int):
        assert size > self.data.shape[0]
        k = self.determine_k(size)
        result = self.generate_points(k)
        return result.sample(size)

    def fit_and_sample_with_labels(self, X: pd.DataFrame, y: np.ndarray):
        X = X.insert(0, "y", y, allow_duplicates=True)
        self.data = X

    def determine_k(self, sample_size):
        if sample_size < self.data.shape[0]:
            return 0
        return math.ceil(sample_size / self.data.shape[0])

    def generate_points(self, total_size: int):
        df = pd.DataFrame(data=None, columns=self.data.columns)
        for row_idx in range(self.data.shape[0]):
            df = df.append(self._shift(row_idx, self.data, self.response_column, self.local_variance, total_size))
        return df

    def _shift(self, row_idx: int, data: pd.DataFrame, response_column: str, local_variance: int, size):
        new_data = pd.DataFrame()
        for col_idx in range(data.shape[1]):
            value = data.iloc[row_idx, col_idx]
            if data.columns[col_idx] != response_column:
                new_data.insert(len(new_data.columns), data.columns[col_idx], np.random.normal(value, local_variance, size), True)
            else:
                new_data.insert(len(new_data.columns), response_column, np.repeat(a=value, repeats=size), True)
        return new_data

    def _conditional_copy(self, original_data: pd.DataFrame, deep_copy: bool):
        if deep_copy:
            return original_data.copy()
        return original_data
