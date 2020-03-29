import logging
import time
from multiprocessing import Queue, Process
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.experiments.model.Exceptions import MalformedExperimentError
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
                 enable_probabilities=True, fragment_limit: int = None, scale=True, min_support=20):
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
                                         execution_times=execution_times, min_support=self.min_support)
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
            scaler.fit(subset_compound.fragment)
            x_training = self._scale(subset_compound.fragment, scaler)
            x_complement = self._scale(subset_compound.complement, scaler) if self.perfect \
                else None  # Don't waste time scaling if it's not a perfect metamodel
        else:
            x_training = subset_compound.fragment
            x_complement = subset_compound.complement

        y_training = subset_compound.fragment_y
        y_complement = subset_compound.complement_y

        if not self.perfect:
            fitted_generator, execution_times[g_fit] = self._fit_generator(x_training)
            fitted_metamodel, execution_times[m_fit] = self._fit_metamodel(x_training, y_training)
        else:
            fitted_generator, execution_times[g_fit] = self._fit_perfect_generator(x_training, x_complement)
            fitted_metamodel, execution_times[m_fit] = self._fit_perfect_metamodel(x_training, y_training, y_complement)

        g_data, execution_times[g_sam] = self._generate_data(x_training, fitted_generator)
        g_data_y, execution_times[m_pred] = self._label_data(g_data, fitted_metamodel)

        start = time.time()

        if self.do_scale:  # Revert scaling before starting SD
            g_data = self._scale_inverse(g_data, scaler)

        result = self.discovery_alg.find(g_data, g_data_y, regression=self.enable_probabilities)
        execution_times[sub] = time.time() - start
        g_data.insert(loc=0, column=subset_compound.y_name, value=g_data_y)

        return result, g_data, execution_times

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

    def _generate_data(self, scaled_fragment: pd.DataFrame, fitted_generator):
        start = time.time()
        g_data = pd.DataFrame(fitted_generator.sample(self.new_sample_size), columns=scaled_fragment.columns)
        g_data = g_data.append(scaled_fragment, ignore_index=not self.perfect)
        duration = time.time() - start
        return g_data, duration

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

    def _asssert_perfect(self):
        if not (type(self.generator) == PerfectGenerator) ^ (type(self.metamodel) == PerfectMetamodel):
            raise MalformedExperimentError("Please only use Perfect generator with PerfectMetamodel")


