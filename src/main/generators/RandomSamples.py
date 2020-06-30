import numpy as np
import pandas as pd


class NormalRandomSampleGenerator:

    def __init__(self):
        self.data = None

    def fit(self, X: pd.DataFrame):
        self.data = X.copy()
        return self

    def sample(self, n_samples = 1) -> pd.DataFrame:
        sample = pd.DataFrame()
        for attr in self.data:
            mu, sigma = pd.Series.mean(self.data[attr]), pd.Series.std(self.data[attr])
            sample[attr] = np.random.normal(mu, sigma, n_samples)
        return sample
    

# =============================================================================
# This generator creates a dataset with independently generated attributes
# from nornal distribution with the same mean values and standard deviations
# as in original data
# 
# df = pd.read_csv("testdata.csv")
# x = NormalRandomSampleGenerator()
# x.fit(df)
# df1 = x.sample(n_samples = 201)
# 
# df.mean(axis = 0)
# df1.mean(axis = 0)
# df.std(axis = 0)
# df1.std(axis = 0)
# 
# import matplotlib.pyplot as plt
# df['color'] = np.zeros(len(df))
# df1['color'] = np.ones(len(df1))
# df = pd.concat([df, df1])
# plt.scatter(df.iloc[:, 3], df.iloc[:, 4], c = df['color'])
# =============================================================================
    

class UniformRandomSamplesGenerator:

    def __init__(self):
        self.data = None

    def fit(self, X: pd.DataFrame):
        self.data = X.copy()
        return self

    def sample(self, n_samples = 1) -> pd.DataFrame:
        sample = pd.DataFrame()
        for attr in self.data:
            attr_min, attr_max = self.data[attr].min(), self.data[attr].max()
            sample[attr] = np.random.uniform(attr_min, attr_max, n_samples)
        return sample


# =============================================================================
# This generator creates a dataset with independently generated attributes
# from uniform distribution with the same ranges as in original data
#
# df = pd.read_csv("testdata.csv")
# x = UniformRandomSamplesGenerator()
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
# plt.scatter(df.iloc[:, 3], df.iloc[:, 4], c = df['color'])
# =============================================================================


class NoiseGenerator:

    def __init__(self, divisor: float = 3):
        self.data = None
        self.divisor = divisor

    def fit(self, X: pd.DataFrame):
        self.data = X.copy()
        return self

    def sample(self, n_samples = 1) -> pd.DataFrame:
        mod_data: pd.DataFrame = self.data.copy()
        for col in mod_data:
            col_values = mod_data[col].to_numpy()
            div_mind_dist = min(np.diff(np.unique(col_values)))/self.divisor
            col_values = np.add(col_values, np.random.uniform(-div_mind_dist, div_mind_dist, len(col_values)))
            mod_data[col] = col_values
        return mod_data
    

# =============================================================================
# df = pd.read_csv("testdata.csv")
# x = NoiseGenerator()
# x.fit(df)
# df1 = x.sample(n_samples = 1000)
# 
# dfn = df.to_numpy()
# df1n = df1.to_numpy()
# np.unique(dfn).size
# np.unique(df1n).size
# # so there are still several duplicated values, althogh much less than in original data.
# 
# import matplotlib.pyplot as plt
# df['color'] = np.zeros(len(df))
# df1['color'] = np.ones(len(df1))
# df = pd.concat([df, df1])
# plt.scatter(df.iloc[:, 3], df.iloc[:, 4], c = df['color'])
# =============================================================================
