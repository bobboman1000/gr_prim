from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.experiments.model.ExperimentSubsetCompound import ExperimentSubsetCompound


class ExperimentDataset:

    def __init__(self, name: str, data: pd.DataFrame, y_name: str, fragment_size: int):
        assert fragment_size <= data.shape[0]
        self.name: str = name
        self.y_name: str = y_name
        self.data: pd.DataFrame = data
        self.fragment_size: int = fragment_size
        self.fragments: List[pd.DataFrame] = self._divide(fragment_size, self.data)
        self.fragments_count = data.shape[0] / fragment_size

    def get_subset_compound(self, fragment_index: int) -> ExperimentSubsetCompound:
        assert fragment_index < len(self.fragments)

        fragment = self.fragments[fragment_index]
        fragment, fragment_y = self._extract_response(fragment)

        complement = self._get_complement(fragment_index)
        complement, complement_y = self._extract_response(complement)

        return ExperimentSubsetCompound(fragment, complement, fragment_y, complement_y, self.y_name)

    def _divide(self, fragment_size: int, data: pd.DataFrame) -> List[pd.DataFrame]:
        result: List[pd.DataFrame] = []
        while data.shape[0] >= fragment_size:
            sample: pd.DataFrame = data.sample(n=fragment_size, replace=False)
            data = data.drop(sample.index)
            result.append(sample)
        return result

    def _extract_response(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        y: pd.DataFrame = data.loc[:, self.y_name]
        data = data.drop(columns=self.y_name)
        return data, y

    def _get_complement(self, fragment_idx: int) -> pd.DataFrame:
        aggregate = pd.DataFrame(columns=self.fragments[fragment_idx].columns)
        for idx in range(len(self.fragments)):
            if idx != fragment_idx:
                aggregate = aggregate.append(self.fragments[idx])
        return aggregate
