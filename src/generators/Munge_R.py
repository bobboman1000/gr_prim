import math

import pandas as pd
from rpy2.robjects import pandas2ri  # install any dependency package if you get error like "module not found"
from rpy2.robjects.packages import importr

pandas2ri.activate()
utils = importr("utils")
utils.chooseCRANmirror(ind=1)
packnames = ('munge', 'devtools', 'FNN')
#utils.install_packages(StrVector(packnames))
munge = importr("munge")
fnn = importr("FNN")
base = importr('base')


class MUNGE:
    def __init__(self, local_var=5, p_swap=0.5):
        self.nn_idx = None
        self.data = None
        self.p_swap = p_swap
        self.local_var = local_var

    def fit(self, X: pd.DataFrame, **kwargs):
        self.nn_idx = fnn.knn_index(X, 1, "kd_tree")
        self.data = X
        return self

    def sample(self, size: int):
        if size > self.data.shape[0]:
            k = self.determine_k(size)
            g_data = munge.munge(self.data, reps=k, local_var=self.local_var, p_swap=self.p_swap, nn_ind=self.nn_idx)
        else:
            g_data = self.data
        return g_data.sample(size)

    """
    MUNGE doubles size of dataset with each iteration. For a given dataset with size n_original and a demanden sample of
    size n_samples find k so that:
        n_original * 2^k >= n_samples <=> k >= log2(n_samples) - log2(n_original)
        
        If data doubles each iteration
        
        math.ceil(np.log2(sample_size) - np.log2(self.data.shape[0]))
    """
    def determine_k(self, sample_size):
        if sample_size < self.data.shape[0]:
            return 0
        return math.ceil(sample_size / self.data.shape[0])
