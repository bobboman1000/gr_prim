from typing import List

import numpy as np
import pandas as pd
from ema_workbench.analysis import scenario_discovery_util as d_u

from src.experiments.model.ExperimentDataset import ExperimentDataset

RESULT_Y_KEY = "y"
MEAN_KEY = "mean"
COVERAGE_KEY = "coverage"
DENSITY_KEY = "box_mean"
MASS_KEY = "box_mass"
F1_KEY = "f1"
F2_KEY = "f2"
WRACC_KEY = "wracc"

MEAN_IN_BOX = "mean_in_box"
MEAN = "mean_out_box"
POS_IN_BOX = "pos_in_box"
POS = "pos_out_box"
N_IN_BOX = "in_box"
N = "n"

def get_initial_restrictions(data: pd.DataFrame) -> pd.DataFrame:
    maximum: pd.DataFrame = data.max(axis=0)
    minimum: pd.DataFrame = data.min(axis=0)
    return pd.DataFrame(data=[minimum, maximum])


class BoxResult:
    def __init__(self, data: pd.DataFrame, restrictions: pd.DataFrame = None, y_name="y"):
        if restrictions is not None:
            in_box_idxs = self.items_idx_in_box(data[restrictions.columns], restrictions)
        else:
            in_box_idxs = data.index
        self.y_name = y_name
        self.stats = self._compute_stats(data, in_box_idxs)

    def _compute_stats(self, data, in_box_idxs):
        y_in_box: pd.Series = data.loc[in_box_idxs, self.y_name]
        y: pd.Series = data[self.y_name]
        return {
            "mean_in_box": y_in_box.mean(),
            "mean_out_box": y.mean(),
            "in_box": y_in_box.size,
            "n": y.size
        }

    def items_idx_in_box(self, data: pd.DataFrame, restriction: pd.DataFrame) -> List[bool]:
        logical = (restriction.loc[0, :].values <= data.values) & \
                  (data.values <= restriction.loc[1, :].values)
        logical = logical.all(axis=1)
        return logical

    def get_mean_outside_box(self) -> float:
        # TODO check this..
        return self.stats[N] * (self.stats[N_IN_BOX] * self.get_box_mean() - self.get_mean())

    def get_mean(self) -> float:
        return self.stats[MEAN]

    def get_box_mean(self) -> float:
        if self.stats[MEAN_IN_BOX] > 0:
            mean = self.stats[MEAN_IN_BOX]
        else:
            mean = 0
        return mean

    def get_box_mass(self) -> float:
        return self.stats[N_IN_BOX] / self.stats[N]

    # This is not coverage!!! It's an adaption to numeric values
    def get_coverage(self) -> float:
        return self.get_mean() / self.stats[MEAN]

    def get_f1(self) -> float:
        return self.get_f_score(beta=1)

    def get_f2(self) -> float:
        return self.get_f_score(beta=2)

    def get_f_score(self, beta: int):
        d = self.get_box_mean()
        c = self.get_coverage()
        if d == 0 or c == 0:
            f = 0
        else:
            f = ((1 + beta**2) * d * c) / (beta**2 * d + c)
        return f

    def getWRacc(self):
        """p(Cond) * (p(Class|Cond) - p(class))"""
        return self.get_box_mass() * (self.get_box_mean() - self.get_mean())

    def get_kpis(self):
        return {
            MEAN_KEY: self.get_mean(),
            DENSITY_KEY: self.get_box_mean(),
            MASS_KEY: self.get_box_mass(),
            COVERAGE_KEY: self.get_coverage(),
            F1_KEY: self.get_f1(),
            F2_KEY: self.get_f2(),
            WRACC_KEY: self.getWRacc()
        }


class FragmentResult:

    def __init__(self, restrictions: List[pd.DataFrame], training_data: pd.DataFrame, experiment_dataset: ExperimentDataset,
                 fragment_idx: int, execution_times, min_support=20):
        self.min_support = min_support
        self.fragment_idx = fragment_idx
        self.execution_times = execution_times
        self.initial_restrictions_train = get_initial_restrictions(training_data.drop(columns=[experiment_dataset.y_name]))
        self.initial_restrictions_test = get_initial_restrictions(experiment_dataset.get_subset_compound(fragment_idx).complement)

        self.boxes = list(map(lambda restriction: restriction.to_dict(orient='list'), restrictions))
        self.box_results_train = self.get_box_results(training_data, restrictions, experiment_dataset.fragment_size, experiment_dataset.y_name)
        test_data = experiment_dataset.get_subset_compound(fragment_idx).get_complete_complement()
        self.box_results_test = self.get_box_results(test_data, restrictions, experiment_dataset.fragment_size, experiment_dataset.y_name)

        self.leftmost_box_idx = -1
        self.highest_mean_idx = int(np.argmax(list(map(lambda b: b.get_box_mean(), self.box_results_train))))
        self.highest_f1_idx = int(np.argmax(list(map(lambda b: b.get_f1(), self.box_results_train))))
        self.highest_f2_idx = int(np.argmax(list(map(lambda b: b.get_f2(), self.box_results_train))))
        self.highest_wracc_idx = int(np.argmax(list(map(lambda b: b.getWRacc(), self.box_results_train))))

        self.min_mass_box = (self.boxes[self.leftmost_box_idx], self.box_results_test[self.leftmost_box_idx].get_box_mass())
        self.highest_mean_box = (self.boxes[self.highest_mean_idx], self.box_results_test[self.highest_mean_idx].get_box_mean())
        self.highest_f1_box = (self.boxes[self.highest_f1_idx], self.box_results_test[self.highest_f1_idx].get_f1())
        self.highest_f2_box = (self.boxes[self.highest_f2_idx], self.box_results_test[self.highest_f2_idx].get_f2())
        self.highest_wracc_box = (self.boxes[self.highest_wracc_idx], self.box_results_test[self.highest_wracc_idx].getWRacc())
        self.kpis = self.get_kpi_list(self.box_results_test)

    def to_restriction(self, box_idx, train=True):
        new_restriction = self.initial_restrictions_test if train else self.initial_restrictions_test
        box = self.boxes[box_idx][0] if train else self.boxes[box_idx][0]
        for key in box:
            new_restriction.loc[0, key] = box[key][0]
            new_restriction.loc[1, key] = box[key][1]
        return new_restriction

    def get_box_results(self, data: pd.DataFrame, restrictions: List[pd.DataFrame], f_size: int, y_name: str):
        box_set = []
        initial_restrictions = get_initial_restrictions(data)
        for idx in range(len(restrictions)):
            restricted_dims = d_u._determine_restricted_dims(restrictions[idx], initial_restrictions)
            br = BoxResult(data, restrictions[idx][restricted_dims], y_name)
            box_set.append(br)
            if not self.__has_min_box_mass(br.get_box_mass(), f_size, self.min_support):
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
