import logging
import time
from multiprocessing import Queue, Process
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.main.generators.RandomSamples import NoiseGenerator
from src.main.experiments.model.ExperimentDataset import ExperimentDataset
from src.main.experiments.model.ExperimentSubsetCompound import ExperimentSubsetCompound
from src.main.experiments.model.FragmentResult import FragmentResult
from src.main.generators.PerfectGenerator import PerfectGenerator
from src.main.metamodels.PerfectMetamodel import PerfectMetamodel


class ExecutionTimes:
    def __init__(self):
        self.generator_fit = -1
        self.generator_sample = -1
        self.metamodel_predict = -1
        self.metamodel_fit = -1
        self.subgroup_disc = -1


g_fit = "fitting_generator"
g_sam = "sampling"
m_fit = "metamodel_fit"
m_pred = "metamodel_predict"
sub = "subgroup discovery"
result_y_key = "y"

Z_SCORE_SCALING = "z_score"
ZERO_ONE_SCALING = "zero_one"


class Experiment:
    executed: bool = False

    def __init__(self, ex_data: ExperimentDataset, generator, metamodel, discovery_alg, name, new_sample_size: int,
                 enable_probabilities=True, fragment_limit: int = None, scaling="zero_one", min_support=0):
        self.ex_data: ExperimentDataset = ex_data
        self.generator = generator
        self.metamodel = metamodel
        self.discovery_alg = discovery_alg
        self.result: List[FragmentResult] = []
        self.name: str = name
        self.new_sample_size = new_sample_size
        self.enable_probabilities = enable_probabilities
        self.debug_logger = logging.getLogger('DEBUG')
        self.failed = 0
        self.fragment_limit = fragment_limit
        self.scaling_type = scaling
        self.perfect = type(generator) == PerfectGenerator
        self.min_support = min_support

    def run(self):
        if self.fragment_limit is not None and len(self.ex_data.fragments) > self.fragment_limit:
            indices = range(self.fragment_limit)
        else:
            indices = range(len(self.ex_data.fragments))
        self.result = self._run_fragments(indices)
        self.executed = True
        return self

    def run_parallel(self, threads: int):
        pool = []
        chunked_fragments_idxs = self._chunks(range(len(self.ex_data.fragments)), threads)
        q = Queue()
        for i in range(threads):
            p = Process(target=self._run_fragments_to_q, args=(chunked_fragments_idxs[i], q))
            pool.append(p)
            p.start()

        for i in range(len(self.ex_data.fragments)):
            self.result.append(q.get())

        for p in pool:
            p.join()
            p.close()

    def _run_fragments(self, indices):
        results: List[FragmentResult] = []
        for idx in indices:
            box, g_data, execution_times, original_idx = self._run_fragment(idx)
            if box is not None:
                results.append(self._build_result(box, g_data, idx, execution_times, original_idx))
            else:
                self.failed += 1
        return results

    def _run_fragment(self, idx):
        subset_compound: ExperimentSubsetCompound = self.ex_data.get_subset_compound(idx)
        box, g_data, execution_times, original_idx = self._exec(subset_compound)
        self.debug_logger.debug(self.name + " " + str(idx) + " " + str(execution_times))
        return box, g_data, execution_times, original_idx

    def _build_result(self, box, g_data, idx, execution_times, original_idx):
        start = time.time()
        test_data = self._get_test_data(training_data=g_data, fragment_idx=idx)
        fragment_result = FragmentResult(restrictions=box, fragment_idx=idx, training_data=g_data, test_data=test_data, y_name=self.ex_data.y_name,
                                         execution_times=execution_times, min_support=self.min_support, original_data_idx=original_idx)
        self.debug_logger.debug("fragment_eval " + str(time.time() - start))
        return fragment_result

    def _run_fragments_to_q(self, indices, q):
        for idx in indices:
            box, g_data, execution_times, original_idx = self._run_fragment(idx)
            if box is not None:
                q.put(self._build_result(box, g_data, idx, execution_times, original_idx))
            else:
                self.failed += 1

    def _chunks(self, long_list, chunk_size):
        """Yield successive n-sized chunks from l."""
        return [long_list[i::chunk_size] for i in range(chunk_size)]

    def delete_models(self):
        del self.generator
        del self.discovery_alg
        del self.metamodel

    def _exec(self, subset_compound: ExperimentSubsetCompound):
        execution_times = {}
        scaler = self._get_scaler(self.scaling_type)
        perfect_gen = type(self.generator) == PerfectGenerator
        perfect_meta = type(self.metamodel) == PerfectMetamodel

        if self.scaling_type is not None:
            scaler.fit(subset_compound.fragment)
            x_training = self._scale(subset_compound.fragment, scaler)
            x_complement = self._scale(subset_compound.complement, scaler) if perfect_gen else None  # Don't waste time scaling if it's not a perfect metamodel
        else:
            x_training = subset_compound.fragment
            x_complement = subset_compound.complement

        y_training = subset_compound.fragment_y
        y_complement = subset_compound.complement_y

        if perfect_gen:
            fitted_generator, execution_times[g_fit] = self._fit_perfect_generator(x_training, x_complement)
        else:
            fitted_generator, execution_times[g_fit] = self._fit_generator(x_training)

        if perfect_meta:
            fitted_metamodel, execution_times[m_fit] = self._fit_perfect_metamodel(x_training, y_training, y_complement)
        else:
            fitted_metamodel, execution_times[m_fit] = self._fit_metamodel(x_training, y_training)

        g_data, execution_times[g_sam], original_idx = self._generate_data(x_training, fitted_generator, perfect_gen)
        g_data_y, execution_times[m_pred] = self._label_data(g_data, fitted_metamodel)

        start = time.time()

        if self.scaling_type is not None:  # Revert scaling before starting SD
            g_data = self._scale_inverse(g_data, scaler)

        result = self.discovery_alg.find(g_data, g_data_y, regression=self.enable_probabilities)
        execution_times[sub] = time.time() - start
        g_data.insert(loc=0, column=subset_compound.y_name, value=g_data_y)

        return result, g_data, execution_times, original_idx

    def _fit_generator(self, x_training: pd.DataFrame) -> Tuple:  # TODO Add interface
        start = time.time()
        fitted_generator = self.generator.fit(X=x_training)
        duration = time.time() - start
        return fitted_generator, duration

    def _fit_perfect_generator(self, x_training: pd.DataFrame, x_complement: pd.DataFrame) -> Tuple:  # TODO Add interface
        start = time.time()
        fitted_generator = self.generator.fit(X=x_training, X_complement=x_complement)
        duration = time.time() - start
        return fitted_generator, duration

    def _fit_metamodel(self, x_training: pd.DataFrame, y_training: Union[np.ndarray, pd.DataFrame, pd.Series]):
        start = time.time()
        try:
            fitted_classifier = self.metamodel.fit(X=x_training, y=y_training)
        except ValueError:
            # TODO Implement y guarantee
            # Sometimes these appear if there is only one class in the training fragment. This code simply excludes samples without
            # an adequate guarantee.
            return None, None
        duration = time.time() - start
        return fitted_classifier, duration

    def _fit_perfect_metamodel(self, x_training: pd.DataFrame, y_training, y_complement):
        start = time.time()
        try:
            fitted_classifier = self.metamodel.fit(X=x_training, y=y_training, y_complement=y_complement)
        except ValueError:
            # TODO Implement y guarantee
            # Sometimes these appear if there is only one class in the training fragment. This code simply excludes samples without
            # an adequate guarantee.
            return None, None
        duration = time.time() - start
        return fitted_classifier, duration

    def _generate_data(self, scaled_fragment: pd.DataFrame, fitted_generator, perfect_gen: bool):
        start = time.time()
        g_data = pd.DataFrame(fitted_generator.sample(self.new_sample_size), columns=scaled_fragment.columns)
        if type(self.generator) != NoiseGenerator:
            g_data = g_data.append(scaled_fragment, ignore_index=not perfect_gen)
        original_idx = g_data.tail(scaled_fragment.shape[0]).index
        # assert_subset_equality(scaled_fragment, g_data.loc[original_idx], original_idx, not perfect_gen)
        # TODO Move this to a test
        duration = time.time() - start
        return g_data, duration, original_idx

    def _label_data(self, gerenated_data: pd.DataFrame, fitted_classifier):
        start = time.time()
        if self.enable_probabilities:
            g_data_y: np.ndarray = fitted_classifier.predict_proba(gerenated_data)
            if not(len(g_data_y.shape) > 1 and g_data_y.shape[1] == 2):
                raise AssertionError
            g_data_y: np.ndarray = g_data_y[:, 1]
        else:
            g_data_y = fitted_classifier.predict(gerenated_data)
        duration = time.time() - start
        return g_data_y, duration

    def _scale(self, x: pd.DataFrame, fitted_scaler):
        scaled_data = fitted_scaler.transform(x)
        x = pd.DataFrame(scaled_data, columns=x.columns, index=x.index)
        return x

    def _scale_inverse(self, x: pd.DataFrame, fitted_scaler):
        scaled_data = fitted_scaler.inverse_transform(x)
        x = pd.DataFrame(scaled_data, columns=x.columns, index=x.index)
        return x

    def _get_method_name_by_idx(self, idx: int):
        return self.name.split("_")[idx]

    def get_generator_name(self):
        return self._get_method_name_by_idx(0)

    def get_metamodel_name(self):
        return self._get_method_name_by_idx(1)

    def _get_test_data(self, training_data: pd.DataFrame, fragment_idx: int) -> pd.DataFrame:
        s_compound = self.ex_data.get_subset_compound(fragment_idx)
        test_data = s_compound.get_complete_complement()
        if self.perfect:
            f_data = s_compound.get_complete_fragment()
            test_data = test_data.append(f_data)
            test_data = test_data.drop(index=training_data.index)
        return test_data

    def _get_scaler(self, scaling: str):
        if scaling == Z_SCORE_SCALING:
            scaler = StandardScaler()
        elif scaling == ZERO_ONE_SCALING:
            scaler = MinMaxScaler()
        else:
            scaler = None
        return scaler


def assert_subset_equality(df1: pd.DataFrame, df2: pd.DataFrame, df2_subset_idx: pd.Index, ignore_idx: bool):
    df2_subset: pd.DataFrame = df2.loc[df2_subset_idx]
    if not ignore_idx:
        assert df1.equals(df2_subset)
    else:
        assert df1.reset_index(drop=True).equals(df2.reset_index(drop=True))
