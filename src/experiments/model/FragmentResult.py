from typing import List

import numpy as np
import pandas as pd
from ema_workbench.analysis import scenario_discovery_util as d_u

from src.experiments.model.ExperimentDataset import ExperimentDataset

result_y_key = "y"
mean_key = "mean"
coverage_key = "coverage"
density_key = "box_mean"
mass_key = "box_mass"
f1_key = "f1"
f2_key = "f2"
wracc_key = "wracc"


class BoxResult:
    def __init__(self, df: pd.DataFrame, restrictions: pd.DataFrame = None):
        self.data = df.astype(np.number)
        self.restrictions = restrictions
        if restrictions is not None:
            self.in_box_idxs = self.items_idx_in_box(df[restrictions.columns], restrictions)
        else:
            self.in_box_idxs = df.index

    def items_idx_in_box(self, data: pd.DataFrame, restriction: pd.DataFrame):
        logical = (restriction.loc[0, :].values <= data.values) & \
                  (data.values <= restriction.loc[1, :].values)
        logical = logical.all(axis=1)
        return logical

    def get_items(self) -> pd.DataFrame:
        return self.data

    def get_items_ys(self) -> pd.Series:
        return self.data[result_y_key]

    def _get_items_in_box(self) -> pd.DataFrame:
        return self.data.loc[self.in_box_idxs, :]

    def _get_items_ys_in_box(self) -> pd.Series:
        return self.data.loc[self.in_box_idxs, result_y_key]

    def _get_items_outside_box(self) -> pd.DataFrame:
        return self.data.drop(index=self.in_box_idxs, axis=0)

    def _get_items_ys_outside_box(self) -> pd.Series:
        return self.data.drop(index=self.in_box_idxs, axis=0)[result_y_key]

    def _get_mean_outside_box(self) -> float:
        return pd.Series.mean(self._get_items_ys_outside_box())

    def get_mean(self) -> float:
        return pd.Series.mean(self.data[result_y_key])

    def get_box_mean(self) -> float:
        if len(self._get_items_ys_in_box()) > 0:
            mean = pd.Series.mean(self._get_items_ys_in_box())
        else:
            mean = 0
        return mean

    def get_box_mass(self) -> float:
        return len(self._get_items_ys_in_box()) / len(self.get_items_ys())

    def get_f1(self):
        d = self.get_box_mean()
        c = self.get_coverage()
        if d == 0 or c == 0:
            f1 = 0
        else:
            f1 = (2 * d * c) / (d + c)
        return f1

    def get_f2(self):
        d = self.get_box_mean()
        c = self.get_coverage()
        if d == 0 or c == 0:
            f2 = 0
        else:
            f2 = (5 * d * c) / (4 * d + c)
        return f2

    def getWRacc(self):
        """p(Cond) * (p(Class|Cond) - p(class))"""
        return self.get_box_mass() * (self.get_box_mean() - self.get_mean())

    def get_coverage(self) -> float:
        s = self._get_items_ys_in_box()
        coi = len(s.where(s == 1).dropna())
        t_s = self.get_items_ys()
        t_coi = len(t_s.where(t_s == 1).dropna())
        return coi / t_coi

    def get_kpis(self):
        return {
            mean_key: self.get_mean(),
            density_key: self.get_box_mean(),
            mass_key: self.get_box_mass(),
            coverage_key: self.get_coverage(),
            f1_key: self.get_f2(),
            f2_key: self.get_f1(),
            wracc_key: self.getWRacc()
        }


class FragmentResult:

    def __init__(self, restrictions: List[pd.DataFrame], experiment_dataset: ExperimentDataset, fragment_idx: int, execution_times, min_support=20):
        self.min_support = min_support
        self.fragment_idx = fragment_idx
        self.execution_times = execution_times
        self.initial_restrictions = restrictions[0]
        box_results = self.get_box_results(restrictions, experiment_dataset, fragment_idx)
        leftmost_box_idx = -1
        highest_box_idx = int(np.argmax(list(map(lambda b: b.get_box_mean(), box_results))))  # TODO Do this after cutting off minpts!!!
        highest_f1_index = int(np.argmax(list(map(lambda b: b.get_f1(), box_results))))
        highest_f2_index = int(np.argmax(list(map(lambda b: b.get_f2(), box_results))))
        highest_wracc_index = int(np.argmax(list(map(lambda b: b.getWRacc(), box_results))))
        self.experiment_dataset = experiment_dataset
        self.left_most_box = (box_results[leftmost_box_idx].restrictions.to_dict(orient='list'), box_results[leftmost_box_idx].get_box_mass())
        self.highest_box = (box_results[highest_box_idx].restrictions.to_dict(orient='list'), box_results[highest_box_idx].get_box_mean())
        self.highest_f1 = (box_results[highest_f1_index].restrictions.to_dict(orient='list'), box_results[highest_f1_index].get_f1())
        self.highest_f2 = (box_results[highest_f2_index].restrictions.to_dict(orient='list'), box_results[highest_f2_index].get_f2())
        self.highest_wracc = (box_results[highest_wracc_index].restrictions.to_dict(orient='list'), box_results[highest_wracc_index].getWRacc())
        self.kpis = self.get_kpi_list(box_results)

    def to_restriction(self, box):
        new_restriction = self.initial_restrictions
        for key in box:
            new_restriction.loc[0, key] = box[key][0]
            new_restriction.loc[1, key] = box[key][1]
        return new_restriction

    def revert_scaling(self, restrictions):
        return self.experiment_dataset.scaler.inverse_transform(restrictions)

    def get_box_results(self, restrictions, experiment_dataset, fragment_idx):
        in_box = experiment_dataset.get_subset_compound(fragment_idx).get_complete_complement()
        box_set = []
        for idx in range(len(restrictions)):
            restricted_dims = d_u._determine_restricted_dims(restrictions[idx], self.initial_restrictions)
            br = BoxResult(in_box, restrictions[idx][restricted_dims])
            in_box = in_box.loc[in_box.index, :]
            box_set.append(br)
            if not self.__has_min_box_mass(br.get_box_mass(), experiment_dataset.fragment_size, self.min_support):
                break
        return box_set

    def __has_min_box_mass(self, relative_box_mass: float, fragment_size: int, min_support: float) -> bool:
        if min_support >= 1:
            mass = relative_box_mass * fragment_size
        else:
            mass = relative_box_mass
        return mass >= min_support and mass > 0

    def get_kpi_list(self, results):
        return list(map(lambda box: box.get_kpis(), results))



