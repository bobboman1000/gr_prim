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
from src.generators.PerfectGenerator import PerfectGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel
from src.metamodels.PerfectMetamodel import PerfectMetamodel


class ExperimentManager:
    debug_logger_key = "DEBUG"
    prod_logger = "EXEC"

    def __init__(self, out='app.log'):
        self.experiments = []
        self.experiment_datasets = []
        self.queues = {}

        ensure_folder_in_root("output")
        self.ex_logger = logging.getLogger(self.prod_logger)
        self.ex_logger.setLevel('INFO')

        self.time_logger = logging.getLogger('DEBUG')
        self.time_logger.setLevel("DEBUG")

        default_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(default_format)

        debug_fh = logging.FileHandler("output/debug.log")
        debug_fh.setLevel("DEBUG")
        debug_fh.setFormatter(default_format)

        fh = logging.FileHandler("output/info.log")
        fh.setLevel("INFO")
        fh.setFormatter(default_format)

        # add ch to logger
        self.ex_logger.addHandler(ch)
        self.ex_logger.addHandler(fh)
        self.ex_logger.addHandler(debug_fh)
        self.time_logger.addHandler(debug_fh)
        self.ex_logger.info("------------------------------    Startup   ------------------------------")

    def add_experiment(self, dataset: ExperimentDataset, generator, metamodel, discovery_alg, name, new_samples, scaling, enable_probabilities=True,
                       fragment_limit=None, min_support=0):
        if not self.experiment_datasets.__contains__(dataset):
            self.experiment_datasets.append(dataset)
        experiment = Experiment(dataset, copy.deepcopy(generator), copy.deepcopy(metamodel), copy.deepcopy(discovery_alg),
                                name, new_sample_size=new_samples, enable_probabilities=enable_probabilities, fragment_limit=fragment_limit,
                                min_support=min_support, scaling=scaling)
        self.experiments.append(experiment)
        self._enqueue(experiment)
        self.ex_logger.info("Added experiment " + experiment.name)

    def build_cartesian_experiments(self, datasets: List[ExperimentDataset], generators, metamodels, discovery_algs, new_samples, scaling,
                                    enable_probabilities=True, fragment_limit: int = None, min_support=0):
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
                                                new_sample_size=new_samples, enable_probabilities=enable_probabilities, fragment_limit=fragment_limit,
                                                scaling=scaling, min_support=min_support)
                        self._enqueue(experiment)
                        self._update_experiments()
                        self._update_experiment_datasets()
        self.ex_logger.info("Cartesian experiments built")

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
        self.ex_logger.info("Start Experiments sequentially")
        self._run_list(self.experiments)
        self.ex_logger.info("Finished experiments")

    def run_fragments_parallel(self, threads: int):
        """Run experiments sequentially, but parallelize the execution of the fragments"""
        i = 1
        self.ex_logger.info("Start Experiments parallel with " + str(threads) + " threads")
        for experiment in self.experiments:
            experiment.run_parallel(threads)
            self.ex_logger.info(str(i) + "th experiment complete. " + str(len(self.experiments)) + " left")
            i += 1
        self.ex_logger.info("Finished Experiments")

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
            self.ex_logger.info("Start batch " + str(batch_count) + " with " + str(batch_length) + " elements and " + str(threads) + " threads")
            batch_result1 = pool.map(self.run, batch)
            self.ex_logger.info("Finish batch " + str(batch_count) + " " + str(len(exps)) + " experiments left")
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
        self.ex_logger.info("Start experiments with " + str(threads) + " threads")
        new_queues: List[List[Experiment]] = pool.map(self._run_list, failed_queues)
        self._set_queues(new_queues)
        self._update_experiments()
        self.ex_logger.info("FINISHED")

    def run_thread_per_dataset(self):
        """
        This is the recommended method to execute the experiments.
        Execute experiments per datasets -> This does not require deepcopying the datasets.
        :return: None
        """
        threads = len(self.queues)
        pool = Pool(threads)
        self.ex_logger.info("Start experiments with " + str(threads) + " threads")
        new_queues: List[List[Experiment]] = pool.map(self._run_list, list(self.queues.values()))
        pool.close()
        self._set_queues(new_queues)
        self._update_experiments()
        self.ex_logger.info("FINISHED")

    def run_all_parallel(self, threads):
        pool = Pool(threads)
        self.ex_logger.info("Start experiments with " + str(threads) + " threads")
        new_experiments: List[Experiment] = pool.map(self.run, list(self.experiments))
        pool.close()
        self.experiments = new_experiments
        self._update_queues()
        self.ex_logger.info("FINISHED")

    def _set_queues(self, queues: List[List[Experiment]]):
        for q in queues:
            assert len(q) > 0
            exp_name = q[0].ex_data.name
            self.queues[exp_name] = q

    def _run_list(self, experiments: List[Experiment]):
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
        self.ex_logger.info("Experiment " + experiment.ex_data.name + " " + str(count) + "/" + str(len(self.queues[experiment.ex_data.name]))
                         + " started (" + experiment.name + ")")
        try:
            experiment = experiment.run()
            self.ex_logger.info("Experiment " + experiment.ex_data.name + " " + str(count) + "/" + str(len(self.queues[experiment.ex_data.name])) + " complete")
        except Exception as e:
            self.ex_logger.info("Experiment " + experiment.ex_data.name + " " + str(count) + "/" + str(len(self.queues[experiment.ex_data.name])) + " failed")
            self.time_logger.exception(str(e))
            self.ex_logger.error(traceback._cause_message)
            traceback.print_exc()

        return experiment

    def run(self, experiment) -> Experiment:
        self.ex_logger.info("Experiment " + str(self.experiments.index(experiment)) + " started (" + experiment.name + ")")
        try:
            experiment = experiment.run()
            self.ex_logger.info("Experiment " + str(self.experiments.index(experiment)) + " complete")
        except Exception as e:
            self.ex_logger.info("Experiment " + str(self.experiments.index(experiment)) + " failed")
            self.ex_logger.error(str(e.__traceback__))
            self.ex_logger.exception(str(e))
        return experiment

    def import_experiments(self, filename: str, add=False):
        new_exps = self.load_from_file(filename)
        if not add:
            self.reset_experiments()
            self.experiments = new_exps
        else:
            self.experiments = self.experiments + new_exps
        self._update_queues()

    def load_from_file(self, filename: str):
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

    def add_perfects(self, datasets: List[ExperimentDataset], metamodels: dict, discovery_algs: dict, new_samples: int, scaling: str, fragment_limit=None,
                     enable_probabilities=True, min_support=0):
        perfect_gen_dict = {"perfect": PerfectGenerator()}
        perfect_meta_dict = {"perfect": PerfectMetamodel()}
        self.build_cartesian_experiments(datasets=datasets, generators=perfect_gen_dict, metamodels=metamodels, discovery_algs=discovery_algs,
                                         new_samples=new_samples, enable_probabilities=enable_probabilities, fragment_limit=fragment_limit, scaling=scaling,
                                         min_support=min_support)
        self.build_cartesian_experiments(datasets=datasets, generators=perfect_gen_dict, metamodels=perfect_meta_dict, discovery_algs=discovery_algs,
                                         new_samples=new_samples, enable_probabilities=enable_probabilities, fragment_limit=fragment_limit, scaling=scaling,
                                         min_support=min_support)

    def add_dummies(self, datasets, metamodels, discovery_algs, scaling, fragment_limit=None, enable_probabilities=True, min_support=0):
        dummy_gen_dict = {"dummy": DummyGenerator()}
        dummy_meta_dict = {"dummy": DummyMetaModel()}
        self.build_cartesian_experiments(datasets, dummy_gen_dict, metamodels, discovery_algs, new_samples=0,
                                         enable_probabilities=enable_probabilities, fragment_limit=fragment_limit, scaling=scaling, min_support=min_support)
        self.build_cartesian_experiments(datasets, dummy_gen_dict, dummy_meta_dict, discovery_algs, new_samples=0,
                                         enable_probabilities=enable_probabilities, fragment_limit=fragment_limit, scaling=scaling, min_support=min_support)

    def get_method_dict(self):
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
