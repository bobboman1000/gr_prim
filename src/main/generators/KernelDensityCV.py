
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott


class KernelDensityBW:

    def __init__(self, method = 'silverman', hard_limits = False):
        if method == 'silverman':
            self.bw_method = bw_silverman
        elif method == 'scott':
            self.bw_method = bw_scott
        else:
            sys.exit("This is not right: method must be scott or silverman")
        
        self.model = None
        self.cnames = None
        self.hard_limits = hard_limits
        self._limits = ()

    def fit(self, X: pd.DataFrame):
        bw = pd.Series(self.bw_method(X))
        if bw.max()/bw.min() > 10:
            warnings.warn("Bandwidths for different dimensions differ by order of magnitude. Consider using z-score scaling")
        bw = bw[0]
        self.model = KernelDensity(bandwidth = bw)
        self.model.fit(X)
        self._limits = (X.min(axis = 0).to_numpy(), X.max(axis = 0).to_numpy())
        self.cnames = X.columns
        return self

    def sample(self, n_samples = 1) -> pd.DataFrame:
        if self.hard_limits:
            samples = self._generate_w_hard_limits(n_samples)
        else:
            samples = pd.DataFrame(self.model.sample(n_samples), columns = self.cnames)
        return samples
    
    def _generate_w_hard_limits(self, n_samples) -> pd.DataFrame:
        sample = self._cleaned_sample(n_samples)
        mult = int(min(100, n_samples/max(sample.shape[0], 10) + 1))
        while sample.shape[0] < n_samples:
            additional = self._cleaned_sample(n_samples * mult)
            sample = np.append(sample, additional, axis = 0)
            if (sample.shape[0]/n_samples < 0.01):
                sys.exit("< 0.01 % of generated points are within the limits; please make sure you scaled the data")
        return pd.DataFrame(sample[:n_samples], columns = self.cnames)

    def _cleaned_sample(self, n_samples) -> np.ndarray:
        new_samples = self.model.sample(n_samples)
        filtered = (new_samples <= self._limits[1]) & (new_samples >= self._limits[0])
        new_samples = new_samples[filtered.all(axis = 1)]
        return new_samples

      
# =============================================================================
# # no limits
# 
# df = pd.read_csv("testdata.csv")
# df = df.iloc[:,[1,2]]
# x = KernelDensityBW(method = 'silverman')
# x.fit(df)
# df1 = x.sample(n_samples = 201)
# 
# df.max(axis = 0)-df1.max(axis = 0)
# df1.min(axis = 0)-df.min(axis = 0)
# 
# import matplotlib.pyplot as plt
# df['color'] = np.zeros(len(df))
# df1['color'] = np.ones(len(df1))
# df = pd.concat([df, df1])
# plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c = df['color'])
# 
# # hard limits (1)
# # Here it is likely that too points are outside the limits
# # since the attribute's scales are very different. Should output an exception
# 
# df = pd.read_csv("testdata.csv")
# df = df.iloc[:,0:6]
# x = KernelDensityBW(method = 'silverman', hard_limits = True)
# x.fit(df)
# df1 = x.sample(n_samples = 201)
# 
# # hard limits (2)
# # This should work as the relative scales are similar
# 
# df = pd.read_csv("testdata.csv")
# df = df.iloc[:,[1,2]]
# x = KernelDensityBW(method = 'silverman', hard_limits = True)
# x.fit(df)
# df1 = x.sample(n_samples = 201)
# 
# df.max(axis = 0)-df1.max(axis = 0)
# df1.min(axis = 0)-df.min(axis = 0)
# 
# import matplotlib.pyplot as plt
# df['color'] = np.zeros(len(df))
# df1['color'] = np.ones(len(df1))
# df = pd.concat([df, df1])
# plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c = df['color'])
# =============================================================================



'''
from sklearn.model_selection import GridSearchCV
from typing import List, Union, Tuple

bw_method_silverman = 'silverman'
bw_method_scott = 'scott'

class KernelDensityCV:

    def __init__(self, bandwidth_list: Union[np.ndarray, List[float]], cv=5, hard_limits=False, sampling_multiplier: int = None):
        self.model: KernelDensity = KernelDensity()
        self.bandwidth_list: List[float] = bandwidth_list
        self.cv = cv
        self.hard_limits = hard_limits
        self._limits = ()
        assert not (hard_limits and sampling_multiplier is None)
        self.sampling_multiplier = sampling_multiplier

    def fit(self, X: pd.DataFrame, **kwargs):
        kde_params = {"bandwidth": self.bandwidth_list}
        kde_cv = GridSearchCV(self.model, kde_params, cv=self.cv)
        kde_cv.fit(X)
        self.model = kde_cv.best_estimator_
        self.model.fit(X)
        if self.hard_limits:
            self._limits = (X.min(axis=0).to_numpy(), X.max(axis=0).to_numpy())
        return self

    def sample(self, size: int) -> np.ndarray:
        if self.hard_limits:
            samples = _generate_w_hard_limits(self.model, size, self.sampling_multiplier, self._limits)
        else:
            samples = self.model.sample(size)
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
'''