import copy
import logging
import os
import pickle
import traceback
from itertools import groupby
from multiprocessing.pool import Pool
from typing import List

from src.experiments.model.Experiment import Experiment, ExperimentDataset
from src.generators.DummyGenerator import DummyGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel


class ExperimentManager:
    debug_logger_key = "DEBUG"
    prod_logger = "PROD"

    def __init__(self, out='app.log'):
        self.experiments = []
        self.experiment_datasets = []

        ensure_folder_in_root("output")
        logging.basicConfig(filename='output/' + out, filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(self.prod_logger)
        logger.setLevel('INFO')

        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
        self.logger = logger
        self.debug_logger = logging.getLogger('EXEC-INFO')
        self.debug_logger.setLevel(self.debug_logger_key)
        self.debug_logger.debug("EXEC-INFO connected")
        self.queues = {}

    def add_experiment(self, dataset: ExperimentDataset, generator, metamodel, discovery_alg, name, new_samples, enable_probabilities=True,
                       fragment_limit=None):
        if not self.experiment_datasets.__contains__(dataset):
            self.experiment_datasets.append(dataset)
        experiment = Experiment(dataset, copy.deepcopy(generator), copy.deepcopy(metamodel), copy.deepcopy(discovery_alg),
                                name, new_sample_size=new_samples, enable_probabilities=enable_probabilities, fragment_limit=fragment_limit)
        self.experiments.append(experiment)
        self._enqueue(experiment)
        self.logger.info("Added experiment " + experiment.name)

    def build_cartesian_experiments(self, datasets: List[ExperimentDataset], generators, metamodels, discovery_algs, new_samples, enable_probabilities=True,
                                    fragment_limit: int = None):
        """ Structure
            generators need fit() and sample()
            metamodels need fit(), predict() and predict_proba()
            discovery algs need find(), return equal-breadth dataframes conatining upper and lower limits.
        """
        for generator in generators:
            for metamodel in metamodels:
                for dataset in datasets:
                    for discovery_alg in discovery_algs:
                        experiment = Experiment(dataset, copy.deepcopy(generators[generator]), copy.deepcopy(metamodels[metamodel]),
                                                copy.deepcopy(discovery_algs[discovery_alg]), self._build_name(generator, metamodel, discovery_alg, dataset),
                                                new_sample_size=new_samples, enable_probabilities=enable_probabilities, fragment_limit=fragment_limit)
                        self._enqueue(experiment)
                        self._update_experiments()
                        self._update_experiment_datasets()
        self.logger.info("Cartesian experiments built")

    def _update_experiments(self):
        self.experiments = [exp for q in list(self.queues.values()) for exp in q]

    def _update_queues(self):
        self.queues = {}
        [self._enqueue(e) for e in self.experiments]

    def _update_experiment_datasets(self):
        for exp in self.experiments:
            if not self.experiment_datasets.__contains__(exp.ex_data.name):
                self.experiment_datasets.append(exp.ex_data)

    def reset_experiments(self):
        self.queues = {}
        self._update_experiments()
        self.experiment_datasets = []

    def _enqueue(self, experiment: Experiment):
        if not self.queues.keys().__contains__(experiment.ex_data.name):
            self.queues[experiment.ex_data.name] = []
        self.queues[experiment.ex_data.name].append(experiment)

    def run_all(self):
        """Run all experiments sequentially"""
        self.logger.info("Start Experiments sequentially")
        self.run_list(self.experiments)
        self.logger.info("Finished experiments")

    def run_fragments_parallel(self, threads: int):
        """Run experiments sequentially, but parallelize the execution of the fragments"""
        i = 1
        self.logger.info("Start Experiments parallel with " + str(threads) + " threads")
        for experiment in self.experiments:
            experiment.run_parallel(threads)
            self.logger.info(str(i) + "th experiment complete. " + str(len(self.experiments)) + " left")
            i += 1
        self.logger.info("Finished Experiments")

    def run_all_parallel_datasetwise(self, threads: int):
        """
        NOT RECOMMENDED

        Run experiments sequentially datasetwise - more memory efficient.
        :param threads:
        :return:
        """
        exps: List[Experiment] = []
        exps = exps + self.experiments
        overall_result = []
        batch_count = 0
        while len(exps) > 0:
            batch = []
            for e in exps:
                if not any(any_o.ex_data == e.ex_data for any_o in batch):
                    batch.append(e)
                    exps.remove(e)
            pool = Pool(threads)
            batch_length = len(batch)
            self.logger.info("Start batch " + str(batch_count) + " with " + str(batch_length) + " elements and " + str(threads) + " threads")
            batch_result1 = pool.map(self.run, batch)
            self.logger.info("Finish batch " + str(batch_count) + " " + str(len(exps)) + " experiments left")
            pool.close()
            pool.join()
            overall_result = overall_result + batch_result1
            batch_count += 1
        self.experiments = overall_result

    def rerun(self, threads):
        """
        Rerun experiments that failed before.
        :param threads: Number of threads
        :return: None
        """
        q_list = list(self.queues.values())
        failed_queues = list(map(lambda q: list(filter(lambda exp: not exp.executed, q)), q_list))
        failed_queues = list(filter(lambda q: len(q) > 0, failed_queues))
        pool = Pool(threads)
        self.logger.info("Start experiments with " + str(threads) + " threads")
        new_queues: List[List[Experiment]] = pool.map(self.run_list, failed_queues)
        self._set_queues(new_queues)
        self._update_experiments()
        self.logger.info("FINISHED")

    def run_thread_per_dataset(self):
        """
        This is the recommended method to execute the experiments.
        Execute experiments per datasets -> This does not require deepcopying the datasets.
        :return: None
        """
        threads = len(self.queues)
        pool = Pool(threads)
        self.logger.info("Start experiments with " + str(threads) + " threads")
        new_queues: List[List[Experiment]] = pool.map(self.run_list, list(self.queues.values()))
        pool.close()
        self._set_queues(new_queues)
        self._update_experiments()
        self.logger.info("FINISHED")

    def run_all_parallel(self, threads):
        """
        """
        pool = Pool(threads)
        self.logger.info("Start experiments with " + str(threads) + " threads")
        new_experiments: List[Experiment] = pool.map(self.run, list(self.experiments))
        pool.close()
        self.experiments = new_experiments
        self._update_queues()
        self.logger.info("FINISHED")

    def _set_queues(self, queues: List[List[Experiment]]):
        for q in queues:
            assert len(q) > 0
            exp_name = q[0].ex_data.name
            self.queues[exp_name] = q

    def run_list(self, experiments: List[Experiment]):
        """
        Run a list of experiments
        :param experiments: A list of Experiments
        :return:
        """
        cnt = 1
        for experiment in experiments:
            self.run_and_log(experiment, cnt)
            cnt += 1
        return experiments

    def run_and_log(self, experiment, count) -> Experiment:
        self.logger.info("Experiment " + experiment.ex_data.name + " " + str(count) + "/" + str(len(self.queues[experiment.ex_data.name]))
                         + " started (" + experiment.name + ")")
        try:
            experiment = experiment.run()
            self.logger.info("Experiment " + experiment.ex_data.name + " " + str(count) + "/" + str(len(self.queues[experiment.ex_data.name])) + " complete")
        except Exception as e:
            self.logger.error("Experiment " + experiment.ex_data.name + " " + str(count) + "/" + str(len(self.queues[experiment.ex_data.name])) + " failed")
            self.logger.error(traceback._cause_message)
            traceback.print_exc()

        return experiment

    def run(self, experiment) -> Experiment:
        self.logger.info("Experiment " + str(self.experiments.index(experiment)) + " started (" + experiment.name + ")")
        try:
            experiment = experiment.run()
            self.logger.info("Experiment " + str(self.experiments.index(experiment)) + " complete")
        except Exception as e:
            self.logger.error("Experiment " + str(self.experiments.index(experiment)) + " failed")
            self.debug_logger.debug(str(e.__traceback__) + str(e))
        return experiment

    def import_experiments(self, filename, add=False):
        new_exps = self.load_from_file(filename)
        if not add:
            self.reset_experiments()
            self.experiments = new_exps
        else:
            self.experiments = self.experiments + new_exps
        self._update_queues()

    def load_from_file(self, filename):
        if filename is None:
            f = "results"
        else:
            f = filename
        ensure_folder_in_root("output")
        with open('output/' + f, 'rb') as infile:
            new_exps = pickle.load(infile)
        return new_exps

    def export_experiments(self, filename: str = None):
        if filename is None:
            f = "results"
        else:
            f = filename
        ensure_folder_in_root("output")
        with open('output/' + f, 'wb') as outfile:
            pickle.dump(self.experiments, outfile)

    def _build_name(self, generator: str, metamodel: str, discovery_alg: str, dataset: ExperimentDataset):
        return generator + "_" + metamodel + "_" + discovery_alg + "_" + dataset.name

    def add_dummies(self, datasets, metamodels, discovery_algs, fragment_limit=None, enable_probabilities=True):
        self.build_cartesian_experiments(datasets, {"dummy": DummyGenerator()}, metamodels, discovery_algs, new_samples=0,
                                         enable_probabilities=enable_probabilities, fragment_limit=fragment_limit)
        self.add_double_dummies(datasets, discovery_algs, fragment_limit=fragment_limit)

    def add_double_dummies(self, datasets, discovery_algs, fragment_limit=None):
        for dataset in datasets:
            for discovery_alg in discovery_algs:
                self.add_experiment(dataset, DummyGenerator(), DummyMetaModel(), discovery_algs[discovery_alg], name="dummy_dummy", new_samples=0,
                                    enable_probabilities=False, fragment_limit=fragment_limit)

    def get_technique_dict(self):
        """
        Get a dictionary of all methods indexed by their name
        :return: Dictionary (string, object)
        """
        experiments_by_technique = {}
        for e in self.experiments:
            technique_name = self.dirty_type_string(e, "generator") + "_" + self.dirty_type_string(e, "metamodel")
            if not experiments_by_technique.__contains__(technique_name): experiments_by_technique[technique_name] = []
            experiments_by_technique[technique_name].append(e)
        return experiments_by_technique

    def dirty_type_string(self, experiment: Experiment, key):
        return str(type(getattr(experiment, key))).split(".").pop()

    def get_unexecuted_idxs(self):
        idxs = list(filter(lambda e: not e.executed, self.experiments))
        return list(map(lambda idx: self.experiments.index(idx), idxs))

    def get_grouped_by(self, mode) -> dict:
        new_qs = {k: [] for k in self.queues.keys()}
        modes = [self.by_gen, self.by_metamodel]
        # [(key, []) for key in list(self.queues.values())]
        for q_key in self.queues:
            q = self.queues[q_key]
            q_sorted = sorted(q, key=modes[mode])
            q_grouped = [list(it) for k, it in groupby(q_sorted, modes[mode])]
            q_grouped = self.bring_dummies_to_top(q_grouped, mode)
            new_qs[q_key] = q_grouped
        return new_qs

    def bring_dummies_to_top(self, q_grouped, mode):
        for group in q_grouped:
            for e in group:
                if mode == 0:
                    if type(e.generator) == DummyGenerator:
                        shuffle_to_head(group, q_grouped)
                    if type(e.metamodel) == DummyMetaModel:
                        shuffle_to_head(e, group)
                elif mode == 1:
                    if type(e.metamodel) == DummyMetaModel:
                        shuffle_to_head(group, q_grouped)
                    if type(e.generator) == DummyGenerator:
                        shuffle_to_head(e, group)
        return q_grouped

    def by_metamodel(self, e):
        return e.name.split("_")[1]

    def by_gen(self, e):
        return e.name.split("_")[0]


def shuffle_to_head(target, t_list: List, index=False):
    if not index:
        t_list.remove(target)
        t_list.insert(0, target)
    else:
        o = t_list.pop(target)
        t_list.insert(0, o)


def ensure_folder_in_root(name: str):
    if not os.path.exists(name):
        os.mkdir(name)
