import math

import annoy as ann
import numpy as np
import pandas as pd

numeric = [int, float, np.int64, np.float, np.float64]

# Preferred method because its way faster than using exact NNs.


class MUNGE:

    def __init__(self, p=0.5, s=1, approximate=False):
        self.data: pd.DataFrame = None
        self.s = s
        self.p = p
        self.approximate = approximate

    def fit(self, X: pd.DataFrame):
        self.data = X
        return self

    def sample(self, size: int):
        k = self.determine_k(size)
        if self.approximate:
            g_data = self.approximate_munge_annoy(self.data, k, self.p, self.s)
        else:
            g_data = self.munge_annoy(self.data, k, self.p, self.s)
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

    def munge_annoy(self, df: pd.DataFrame, k, p: float = 0.5, s: float = 1) -> pd.DataFrame:
        result = df
        nn_idx: ann.AnnoyIndex = self.__nn_index(df)
        for i in range(k):
            data = df.copy(deep=True)
            data = data.reset_index(drop=True)
            for e_idx, value in data.iterrows():
                nn = nn_idx.get_nns_by_item(e_idx, 2).pop()
                for attr in data:
                    if np.random.random() > p:
                        if type(data[attr][e_idx]) in numeric:
                            sd = np.abs(data.loc[e_idx, attr] - data.loc[nn, attr]) / s
                            nn_sample = np.random.normal(data.loc[e_idx, attr], sd, 1)
                            e_sample = np.random.normal(data.loc[nn, attr], sd, 1)
                            data.loc[e_idx, attr] = e_sample
                            data.loc[nn, attr] = nn_sample
                        else:
                            old_element = data.loc[e_idx, attr]
                            element = data.loc[nn, attr]
                            data.loc[e_idx, attr] = element
                            data.loc[nn, attr] = old_element
                            data = data.reset_index(drop=True) # TODO review if this is correct
            result = result.reset_index(drop=True)
            result = result.append(data)
        return result

    def approximate_munge_annoy(self, df: pd.DataFrame, k, p: float = 0.5, s: float = 1) -> pd.DataFrame:
        result = pd.DataFrame(columns=df.columns)
        nn_idx: ann.AnnoyIndex = self.__nn_index(df)
        orig_data = df.reset_index(drop=True)
        for e_idx in range(orig_data.shape[0] - 1):
            nn = nn_idx.get_nns_by_item(e_idx, 2).pop()
            data = pd.DataFrame()
            for attr in result.columns:
                sd = np.abs(orig_data.loc[e_idx, attr] - orig_data.loc[nn, attr]) / s
                nn_sample = np.random.normal(orig_data.loc[e_idx, attr], sd, round(k * p))
                e_sample = np.random.normal(orig_data.loc[nn, attr], sd, round(k * p))
                data.insert(loc=0, column=attr, value=e_sample + nn_sample)
                data = data.reset_index(drop=True)  # TODO review if this is correct
                result = result.append(data)
                result = result.reset_index(drop=True)
        return result

    def __nn_index(self, data: pd.DataFrame, n_trees: int = 10) -> ann.AnnoyIndex:
        t = ann.AnnoyIndex(data.shape[1], 'euclidean')  # Length of item vector that will be indexed
        for row_idx in range(data.shape[0]):
            t.add_item(row_idx, data.iloc[row_idx, :])
        t.build(n_trees)
        return t


# Deprecated #
# ---------------------------------------------------------------------------------------------------------------- #

def munge(training_data, k, p: float = 0.5, s: float = 1, response_col: str = None) -> pd.DataFrame:
    nn_idx = __naive_nn_index(training_data, response_col)
    for i in range(k):
        data = training_data.copy()
        for row_idx in range(data.shape[0] - 1):
            nn = nn_idx.iloc[row_idx, 0]
            for attr_idx in range(0, data.shape[1] - 1):
                sd = np.abs(data.iloc[row_idx, attr_idx] - data.iloc[nn, attr_idx] / s)
                nn_sample = np.random.normal(data.iloc[row_idx, attr_idx], sd, 1)
                e_sample = np.random.normal(data.iloc[nn, attr_idx], sd, 1)
                if np.random.random() > p:
                    data.iloc[row_idx, attr_idx] = e_sample
                    data.iloc[nn, attr_idx] = nn_sample
        d = d.append(other=data)
        print(i + 1, ".  munge iteration completed" )
    print("Munge completed", d.shape[0], "examples returned")
    return d


def slow_munge(training_data, k, p=0.5, s=1) -> pd.DataFrame:
    d = pd.DataFrame()
    for i in range(1, k):
        data = training_data.copy()
        for e in range(0, data.shape[0] - 1):
            nn = __naive_nn(data, e)
            for attr in range(0, data.shape[1] - 1):
                sd = np.abs(data.iloc[e, attr] - data.iloc[nn[0], attr] / s)
                nn_sample = np.random.normal(data.iloc[e, attr], sd, 1)
                e_sample = np.random.normal(data.iloc[nn[0], attr], sd, 1)
                if np.random.random() > p:
                    data.iloc[e, attr] = e_sample
                    data.iloc[nn[0], attr] = nn_sample
        d = d.append(other=data)
    return d


def __naive_nn(data: pd.DataFrame, exampleIdx: int) -> list:
    min_dist = [-1, math.inf]
    maxIdx = list(range(0, data.shape[0] - 1))
    maxIdx.remove(exampleIdx)
    for i in maxIdx:
        dist = np.linalg.norm(data.iloc[i, :] - data.iloc[exampleIdx, :])
        if dist < min_dist[1]:
            min_dist = [i, dist]
    return min_dist


def __naive_nn_index(data: pd.DataFrame, response: str = None) -> float:
    if response is not None:
        data = data.drop(response, axis=1)

    std_example = {"NN": -1, "dist": math.inf}
    nn = pd.DataFrame(std_example, columns=["NN", "dist"], index=[0])
    nn_index = pd.concat([nn] * (data.shape[0] - 1), ignore_index=True)
    idxlist = list(range(0, data.shape[0] - 1))

    for i in idxlist:
        for k in idxlist:
            if i != k:
                dist = np.linalg.norm(data.iloc[i, :] - data.iloc[k, :])
                if dist < nn_index.iloc[i, 1]:
                    nn_index.iloc[i] = [k, dist]
                if dist < nn_index.iloc[k, 1]:
                    nn_index.iloc[k] = [i, dist]
    return nn_index





