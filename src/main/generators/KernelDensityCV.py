from typing import List, Union, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott
import numpy as np

bw_method_silverman = 'silverman'
bw_method_scott = 'scott'


class KernelDensityBW:

    def __init__(self, method, hard_limits=False, sampling_multiplier=None):
        self.kde = None
        if method == 'silverman':
            self.bw_method = bw_silverman
        elif method == 'scott':
            self.bw_method = bw_scott
        else:
            print("This is not right: method must be scott or silverman")
        self.hard_limits = hard_limits
        self._limits = ()
        assert not (hard_limits and sampling_multiplier is None)
        self.sampling_multiplier = sampling_multiplier

    def fit(self, X: pd.DataFrame, **kwargs):
        bw: float = self.bw_method(X)
        if isinstance(bw, pd.Series):
            bw = bw.tolist().pop()  # If a list, take any - they are all the same anyway!
        self.kde = KernelDensity(bandwidth=bw)
        self.kde.fit(X)
        if self.hard_limits:
            self._limits = (X.min(axis=0).to_numpy(), X.max(axis=0).to_numpy())
        return self

    def sample(self, size: int):
        if self.hard_limits:
            samples = _generate_w_hard_limits(self.kde, size, self.sampling_multiplier, self._limits)
        else:
            samples = self.kde.sample(size)
        return samples


class KernelDensityCV:

    def __init__(self, bandwidth_list: Union[np.ndarray, List[float]], cv=5, hard_limits=False, sampling_multiplier: int = None):
        self.kde: KernelDensity = KernelDensity()
        self.bandwidth_list: List[float] = bandwidth_list
        self.cv = cv
        self.hard_limits = hard_limits
        self._limits = ()
        assert not (hard_limits and sampling_multiplier is None)
        self.sampling_multiplier = sampling_multiplier

    def fit(self, X: pd.DataFrame, **kwargs):
        kde_params = {"bandwidth": self.bandwidth_list}
        kde_cv = GridSearchCV(self.kde, kde_params, cv=self.cv)
        kde_cv.fit(X)
        self.kde = kde_cv.best_estimator_
        self.kde.fit(X)
        if self.hard_limits:
            self._limits = (X.min(axis=0).to_numpy(), X.max(axis=0).to_numpy())
        return self

    def sample(self, size: int) -> np.ndarray:
        if self.hard_limits:
            samples = _generate_w_hard_limits(self.kde, size, self.sampling_multiplier, self._limits)
        else:
            samples = self.kde.sample(size)
        return samples


def _generate_w_hard_limits(kde, n: int, sampling_multiplier: int, limits: Tuple) -> np.ndarray:
    result: np.ndarray = _cleaned_sample(kde, n, limits)
    while result.shape[0] != n:
        additional = _cleaned_sample(kde, sampling_multiplier * n, limits)
        p_needed = n - result.shape[0]
        p_available = len(additional)
        p = p_needed if p_needed <= p_available else p_available
        result = np.append(result, additional[:p], axis=0)
        assert result.shape[0] <= n
    return result


def _cleaned_sample(kde, n: int, limits: Tuple) -> np.ndarray:
    new_samples = kde.sample(n)
    new_samples = _in_bounds(new_samples, limits[0], limits[1])
    return new_samples


def _in_bounds(data: np.ndarray, minima: np.ndarray, maxima: np.ndarray) -> np.ndarray:
    logical: np.ndarray = (data <= maxima) & (data >= minima)
    logical = logical.all(axis=1)
    return data[logical]
