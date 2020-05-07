from .TestUtil import get_random_df
from src.main.generators.KernelDensityCV import KernelDensityCV, _in_bounds
import numpy as np
import pandas as pd

test_data: pd.DataFrame = get_random_df((500, 100))


def test_kde_limits():
    """
    Create a large matrix and check if kde generates points beyond [0,1]
    """
    kde = KernelDensityCV([0.1], hard_limits=True)
    kde.fit(test_data)
    n_samples = kde.sample(size=1000)
    zeros = np.zeros(100)
    ones = np.ones(100)
    assert _in_bounds(n_samples, zeros, ones).shape == n_samples.shape
    n_samples = np.vstack([n_samples, np.add(ones, ones)])
    assert not _in_bounds(n_samples, zeros, ones).shape == n_samples.shape