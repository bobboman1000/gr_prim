from src.main.generators.KernelDensityCV import _in_bounds
from .TestUtil import get_random_df
from src.main.generators.KernelDensityCV import *
import numpy as np
import pandas as pd

test_data: pd.DataFrame = get_random_df((500, 100))


def test_kde_limits():
    """
    Create a large matrix and check if kde generates points beyond [0,1]
    """
    kde = KernelDensityCV([0.1], hard_limits=True, sampling_multiplier=10)
    kde.fit(test_data)
    n_samples = kde.sample(size=1000)
    zeros = np.zeros(100)
    ones = np.ones(100)
    assert _in_bounds(n_samples, zeros, ones).shape == n_samples.shape
    n_samples = np.vstack([n_samples, np.add(ones, ones)])
    assert not _in_bounds(n_samples, zeros, ones).shape == n_samples.shape


def test_kde_limits():
    """
    Create a large matrix and check if kde generates points beyond [0,1]
    """
    kde = KernelDensityBW(bw_method_silverman, hard_limits=True, sampling_multiplier=10)
    kde.fit(test_data)
    n_samples = kde.sample(size=1000)
    zeros = np.zeros(100)
    ones = np.ones(100)
    assert _in_bounds(n_samples, zeros, ones).shape == n_samples.shape
    n_samples = np.vstack([n_samples, np.add(ones, ones)])
    assert not _in_bounds(n_samples, zeros, ones).shape == n_samples.shape