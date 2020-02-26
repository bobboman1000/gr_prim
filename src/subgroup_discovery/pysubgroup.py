import math

import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup.boolean_target import StandardQF
from pysubgroup.numeric_target import StandardQFNumeric
from pysubgroup.subgroup import SubgroupDescription


class PySubMethod:
    BEAM_SEARCH = "beam_search"  # CN2 -    Clark, P., Niblett, T.: The CN2 induction algorithm. Machine learning 3(4), 261– 283 (1989)
    APRIORI = "apriori"  # apriori sd -   Kavˇsek, B., Lavraˇc, N.: Apriori-sd: Adapting association rule learning to subgroup discovery. Applied Artificial Intelligence 20(7), 543–583 (2006)
    BEST_FIRST_SEARCH = "bfs"  # cluster-grouping  -  Zimmermann, A., De Raedt, L.: Cluster-grouping: from subgroup discovery to clustering. Machine Learning 77(1), 125–159 (2009)
    BSD = "bsd" #   -   Lemmerich, F., Rohlfs, M., Atzmueller, M.: Fast discovery of relevant subgroup patterns. In: International Florida Artificial Intelligence Research Society Confer- ence (FLAIRS). pp. 428–433 (2010)

    methods = [BEAM_SEARCH, APRIORI, BEST_FIRST_SEARCH, BSD]


class PySub:
    def __init__(self, method: PySubMethod):
        self.method = method

    def find(self, X: pd.DataFrame, y: pd.DataFrame):
        df = X.copy(deep=True)
        y_name = "y"
        assert not X.columns.contains("y")
        df.insert(loc=0, column=y_name, value=y)
        target = ps.NumericTarget(y_name)

        searchspace = ps.create_selectors(df, ignore=[y_name])
        task = ps.SubgroupDiscoveryTask(df, target, searchspace, depth=2, qf=StandardQFNumeric(1))

        assert self.method in [method for method in PySubMethod.methods]
        if self.method == PySubMethod.BEAM_SEARCH:
            result = ps.BeamSearch().execute(task)
        elif self.method == PySubMethod.APRIORI:
            result = ps.Apriori().execute(task)
        elif self.method == PySubMethod.BSD:
            result = ps.BSD().execute(task)
        elif self.method == PySubMethod.BEST_FIRST_SEARCH:
            result = ps.BestFirstSearch().execute(task)
        else:
            result = None
            print("This should never happen")
        return self.__to_ex_result(df, result)

    def __to_ex_result(self, data: pd.DataFrame, ps_result: list) -> list:
        return list(map(lambda res: self._to_pd_box(data, res[1].subgroup_description), ps_result))

    # TODO numerics only
    def _to_pd_box(self, data: pd.DataFrame, subgrp_desc: SubgroupDescription):
        minus_inf = np.repeat(-math.inf, data.shape[1])
        plus_inf = np.repeat(math.inf, data.shape[1])
        result = pd.DataFrame((minus_inf, plus_inf), columns=data.columns)
        for selector in subgrp_desc.selectors:
            result.loc[0, selector.attribute_name] = selector.lower_bound
            result.loc[1, selector.attribute_name] = selector.upper_bound
        return result


class PrecisionQF(StandardQF):
    @staticmethod
    def precision_qf(instances_subgroup, positives_subgroup):
        if instances_subgroup == 0:
            return 0
        p_subgroup = positives_subgroup / instances_subgroup
        return p_subgroup

    def __init__(self):
        super().__init__(0)

    def evaluate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return PrecisionQF.precision_qf(instances_subgroup, positives_subgroup)

    def optimistic_estimate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup,
                                            positives_subgroup):
        return PrecisionQF.precision_qf(positives_subgroup, positives_subgroup)


class DistanceQF(StandardQF):
    @staticmethod
    def distance_qf(positives_dataset, instances_subgroup, positives_subgroup):
        if instances_subgroup == 0:
            return 0
        p_subgroup = positives_subgroup / instances_subgroup
        p_dataset = positives_subgroup / positives_dataset
        return np.sqrt((p_subgroup ** 2) * (p_dataset ** 2))

    def __init__(self):
        super().__init__(0)

    def evaluate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup, positives_subgroup):
        return DistanceQF.distance_qf(positives_dataset, instances_subgroup, positives_subgroup)

    def optimistic_estimate_from_statistics(self, instances_dataset, positives_dataset, instances_subgroup,
                                            positives_subgroup):
        return DistanceQF.distance_qf(positives_dataset, positives_subgroup, positives_subgroup)


