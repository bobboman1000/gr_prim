import pandas as pd
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott

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

    def fit(self, X: pd.DataFrame):
        bw: float = self.bw_method(X)
        if isinstance(bw, pd.Series):
            bw = bw.tolist().pop()  # If a list, take any - they are all the same anyway!
        self.kde = KernelDensity(bandwidth=bw)
        self.kde.fit(X)
        return self

    def sample(self, size: int):
        sample = self.kde.sample(size)
        self.kde = None
        return sample

