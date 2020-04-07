from typing import List, Union

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott
import numpy as np

bw_method_silverman = 'silverman'
bw_method_scott = 'scott'


class KernelDensityBW:

    def __init__(self, method):
        self.kde: KernelDensity = None
        if method == 'silverman':
            self.bw_method = bw_silverman
        elif method == 'scott':
            self.bw_method = bw_scott
        else:
            print("This is not right: method must be scott or silverman")

    def fit(self, X: pd.DataFrame, **kwargs):
        bw: float = self.bw_method(X)
        if isinstance(bw, pd.Series):
            bw = bw.tolist().pop()  # If a list, take any - they are all the same anyway!
        self.kde = KernelDensity(bandwidth=bw)
        self.kde.fit(X)
        return self

    def sample(self, size: int):
        sample = self.kde.sample(size)
        return sample


class KernelDensityCV:

    def __init__(self, bandwidth_list: Union[np.ndarray, List[float]], cv=5):
        self.kde: KernelDensity = KernelDensity()
        self.bandwidth_list: List[float] = bandwidth_list

    def fit(self, X: pd.DataFrame, **kwargs):
        kde_params = {"bandwidth": self.bandwidth_list}
        kde_cv = GridSearchCV(self.kde, kde_params)
        kde_cv.fit(X)
        self.kde = kde_cv.best_estimator_
        self.kde.fit(X)
        return self

    def sample(self, size: int):
        sample = self.kde.sample(size)
        return sample

