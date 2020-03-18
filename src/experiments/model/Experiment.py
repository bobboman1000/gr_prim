import logging
import time
from multiprocessing import Queue, Process
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.experiments.model.ExperimentDataset import ExperimentDataset
from src.experiments.model.ExperimentSubsetCompound import ExperimentSubsetCompound
from src.experiments.model.FragmentResult import FragmentResult
from src.generators.PerfectGenerator import PerfectGenerator
from src.metamodels.PerfectMetamodel import PerfectMetamodel


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


class Experiment:
    executed: bool = False

    def __init__(self, ex_data: ExperimentDataset, generator, metamodel, discovery_alg, name, new_sample_size: int,
                 enable_probabilities=True, fragment_limit: int = None, scale=True):
        self.ex_data: ExperimentDataset = ex_data
        self.generator = generator
        self.metamodel = metamodel
        self.discovery_alg = discovery_alg
        self.result: List[FragmentResult] = []
        self.name: str = name
        self.new_sample_size = new_sample_size
        self.enable_probabilities = enable_probabilities
        self.debug_logger = logging.getLogger('EXEC-INFO')
        self.failed = 0
        self.fragment_limit = fragment_limit
        self.do_scale = scale
        self.perfect = type(generator) == PerfectGenerator and type(metamodel) == PerfectMetamodel

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
            box, g_data, execution_times = self._run_fragment(idx)
            if box is not None:
                results.append(self._build_result(box, g_data, idx, execution_times))
            else:
                self.failed += 1
        return results

    def _run_fragment(self, idx):
        subset_compound: ExperimentSubsetCompound = self.ex_data.get_subset_compound(idx)
        box, g_data, execution_times = self._exec(subset_compound)
        self.debug_logger.debug(self.name + " " + str(idx) + " " + str(execution_times))
        return box, g_data, execution_times

    def _build_result(self, box, g_data, idx, execution_times):
        start = time.time()
        fragment_result = FragmentResult(restrictions=box, experiment_dataset=self.ex_data, fragment_idx=idx, training_data=g_data,
                                         execution_times=execution_times)
        self.debug_logger.debug("fragment_eval " + str(time.time() - start))
        return fragment_result

    def _run_fragments_to_q(self, indices, q):
        for idx in indices:
            box, g_data, execution_times = self._run_fragment(idx)
            if box is not None:
                q.put(self._build_result(box, g_data, idx, execution_times))
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
        scaler = MinMaxScaler()

        if self.do_scale:
            x_training, scaler = self._scale(subset_compound.fragment, scaler, inverse=False)

        fitted_generator, execution_times[g_fit] = self._fit_generator(subset_compound)
        fitted_metamodel, execution_times[m_fit] = self._fit_metamodel(subset_compound)
        g_data, execution_times[g_sam] = self._generate_data(subset_compound, fitted_generator)
        g_data_y, execution_times[m_pred] = self._label_data(g_data, fitted_metamodel)

        start = time.time()

        if self.do_scale:  # Revert scaling before starting SD
            g_data, scaler = self._scale(g_data, scaler, inverse=True)

        result = self.discovery_alg.find(g_data, g_data_y, regression=self.enable_probabilities)
        execution_times[sub] = time.time() - start
        g_data.insert(loc=0, column=subset_compound.y_name, value=g_data_y)

        return result, g_data, execution_times

    def _fit_generator(self, subset_compound: ExperimentSubsetCompound) -> Tuple:  # TODO Add interface
        start = time.time()
        if not self.perfect:
            fitted_generator = self.generator.fit(X=subset_compound.fragment)
        else:
            fitted_generator = self.generator.fit(X=subset_compound.fragment, X_complement=subset_compound.complement)
        duration = time.time() - start
        return fitted_generator, duration

    def _fit_metamodel(self, subset_compound: ExperimentSubsetCompound):
        start = time.time()
        try:
            if not self.perfect:
                fitted_classifier = self.metamodel.fit(X=subset_compound.fragment, y=subset_compound.fragment_y)
            else:
                fitted_classifier = self.metamodel.fit(X=subset_compound.fragment, y=subset_compound.fragment_y,
                                                       y_complement=subset_compound.complement_y)
        except ValueError:
            # TODO Implement y guarantee
            # Sometimes these appear if there is only one class in the training fragment. This code simply excludes samples without
            # an adequate guarantee.
            return None, None
        duration = time.time() - start
        return fitted_classifier, duration

    def _generate_data(self, subset_compound: ExperimentSubsetCompound, fitted_generator):
        start = time.time()
        g_data = pd.DataFrame(fitted_generator.sample(self.new_sample_size), columns=subset_compound.fragment.columns)
        g_data = g_data.append(subset_compound.fragment)
        duration = time.time() - start
        return g_data, duration

    def _label_data(self, g_data: pd.DataFrame, fitted_classifier):
        start = time.time()
        if self.enable_probabilities:
            g_data_y: np.ndarray = fitted_classifier.predict_proba(g_data)
            if not(len(g_data_y.shape) > 1 and g_data_y.shape[1] == 2):
                raise AssertionError
            g_data_y: np.ndarray = g_data_y[:, 1]
        else:
            g_data_y = fitted_classifier.predict(g_data)
        duration = time.time() - start
        return g_data_y, duration

    def _scale(self, x: pd.DataFrame, scaler, inverse: bool = False):
        fitted_scaler = scaler.fit(x)
        if not inverse:
            scaled_data = fitted_scaler.transform(x)
        else:
            scaled_data = fitted_scaler.inverse_transform(x)

        x = pd.DataFrame(scaled_data, columns=x.columns)
        return x, fitted_scaler

    def _get_method_name_by_idx(self, idx: int):
        return self.name.split("_")[idx]

    def get_generator_name(self):
        return self._get_method_name_by_idx(0)

    def get_metamodel_name(self):
        return self._get_method_name_by_idx(1)

    def _asssert_perfect(self):
        if not (type(self.generator) == PerfectGenerator) ^ (type(self.metamodel) == PerfectMetamodel):
            raise MalformedExperimentError("Please only use Perfect generator with PerfectMetamodel")


class MalformedExperimentError(Exception):
    """Raises if the generator or the metamodel do not fit together. This may happen if you use dummies or perfects incorrectly."""
    pass
