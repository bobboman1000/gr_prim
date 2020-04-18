import math
from decimal import Decimal, getcontext
from typing import List, Tuple, Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from sklearn.metrics import auc

from src.experiments.ExperimentManager import ExperimentManager, ensure_folder_in_root
from src.experiments.model.Experiment import Experiment
from src.experiments.model.FragmentResult import BoxResult
from src.generators.DummyGenerator import DummyGenerator
from src.metamodels.DummyMetamodel import DummyMetaModel

coverage_key = "coverage"
density_key = "box_mean"
mass_key = "box_mass"
f1_key = "f1"
f2_key = "f2"
wracc_key = "wracc"

# Standard colors need some updates - we should probably not use them. Instead configure it for each experiment
gen_c = {
    "dummy": "darkgrey",
    "random-uniform": "steelblue",
    "random-normal": "rosybrown",
    "gaussian-mixtures": "darkgreen",
    "kde": "blue",
    "munge": "red",
}

metamodels_c = {
    "dummy": "darkgrey",
    "classRF": "blue",
    "bNN": "yellow",
    "kriging": "green",
    "neural-net": "turquoise",
    "SVC-calibrated": "orange",
    "nb-calibrated": "turquoise",
    "classRF-calibrated": "blue",
}

abbreviations_dict = {
    "dummy": "dummy",
    "random-uniform": "rand-uni",
    "random-normal": "rand-norm",
    "gaussian-mixtures": "gmm",
    "kde": "kde",
    "munge": "munge",
    "munge1": "munge1",
    "munge0.2": "munge0.2",
    "classRF": "classRF",
    "bNN": "bNN",
    "kriging": "kriging",
    "neural-net": "ann",
    "SVC-calibrated": "SVC-c",
    "nb-calibrated": "nb-c",
    "classRF-calibrated": "classRF-c",
}


# TODO This class needs refactoring - badly structured - consolidate private methods
# TODO Fix mode 0; Mode 1 works well
class Visualizer:
    box_width = 1.5
    offset_in_group = 1.9
    offset_of_groups = 8

    boxplot_top_offset = 1.19

    def __init__(self, experiment_mananger, detailed_mode=False):
        self.detailed_mode = detailed_mode
        self.exp_man: ExperimentManager = experiment_mananger
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.sans-serif'] = 'Linux Libertine O'
        plt.rcParams.update({'font.size': 14})
        rcParams["savefig.dpi"] = 100
        rcParams["figure.dpi"] = 100
        ensure_folder_in_root("result_grafics")
        if self.box_width > self.offset_in_group:
            raise Warning("box_width larger than offset: Boxplots will overlap")

    def boxplot_dataset(self, q: List[List[Experiment]], q_name: str, metric, mode, vert=True, selected_methods=None, abbreviate=False, display_range=None):
        plt.figure(figsize=(7, 7))
        plt.title(q_name)
        if display_range is None:
            display_range = (0, self.boxplot_top_offset)
        if not vert:
            plt.xlim(display_range)
            plt.xlabel(self.get_metric_name(metric))
        else:
            plt.ylim(display_range)
            plt.ylabel(self.get_metric_name(metric))

        ticks = []
        model_no = [len(q[-1]), len(q)]  # [ len(inactive mode), len(active mode) ]
        i = 0
        bps = []
        assert len(q) > 0
        for e_list in q:
            model_name = e_list[0].name.split("_")[mode]
            ticks.append(model_name)
            boxplot, sorted_experiments = self._get_boxplot_from_elist(e_list, metric, self._get_positions(i, len(e_list), model_no[0]), mode=mode, vert=vert,
                                                                       selected_methods=selected_methods)
            bps.append(boxplot)
            i += 1
        if abbreviate:
            ticks = self._abbreviate(ticks)
        if not vert:
            plt.yticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks)
            plt.ylim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        else:
            plt.xticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks, rotation=45)
            plt.xlim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        self._add_legend(sorted_experiments, bps, mode, abbreviate=abbreviate)
        plt.savefig("result_grafics/" + q_name + "_" + self.get_metric_name(metric, short=True) + ".pdf", bbox_inches='tight')
        plt.show()

    def boxplot_all_datasets(self, metric, mode: int, vert=True, abbreviate=False, selected_methods=None, display_range=None):
        qs = self.exp_man.get_grouped_by(mode)
        for q_name in qs:
            self.boxplot_dataset(qs[q_name], q_name, metric, mode, vert, abbreviate=abbreviate, selected_methods=selected_methods, display_range=display_range)

    def boxplot_all_datasets_as_one(self, metric, mode: int = 1, vert=True, abbreviate=False, selected_methods=None, legend=True, title=None):
        grouped_qs = self.exp_man.get_grouped_by(mode)
        plt.figure(figsize=(14, 7))
        if title is not None:
            plt.title(title)
        if not vert:
            plt.xlim((0, self.boxplot_top_offset))
            plt.xlabel(self.get_metric_name(metric))
        else:
            plt.ylim((0, self.boxplot_top_offset))
            plt.ylabel(self.get_metric_name(metric))

        q = grouped_qs.popitem()[1]
        for e_list_idx in range(len(q)):
            q[e_list_idx] = [q[e_list_idx]]
            for q_name in grouped_qs:
                q[e_list_idx].append(grouped_qs[q_name][e_list_idx])

        ticks = []
        model_no = [len(q[-1][-1]), len(q)]  # [ len(inactive mode), len(active mode) ]
        i = 0
        bps = []
        assert len(q) > 0
        for e_lists in q:
            model_name = e_lists[0][0].name.split("_")[mode]
            inner_length = len(e_lists[-1])
            ticks.append(model_name)
            boxplot, sorted_experiments = self._get_boxplot_from_elists(e_lists=e_lists, metric=metric, pos=self._get_positions(i, inner_length, model_no[0]),
                                                                        mode=mode, vert=vert, selected_methods=selected_methods)
            bps.append(boxplot)
            i += 1
        if abbreviate:
            ticks = self._abbreviate(ticks)
        if not vert:
            plt.yticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks)
            plt.ylim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        else:
            plt.xticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks, rotation=45)
            plt.xlim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        if legend:
            self._add_legend(sorted_experiments, bps, mode, abbreviate=abbreviate)
        plt.savefig("result_grafics/" + "All" + "_" + self.get_metric_name(metric, short=True) + ".pdf", bbox_inches='tight')
        plt.show()

    def csv_datasets_as_one(self, metric, mode: int = 1, vert=True, abbreviate=False, selected_methods=None, legend=True, title=None):
        grouped_qs = self.exp_man.get_grouped_by(mode)
        q = grouped_qs.popitem()[1]
        for e_list_idx in range(len(q)):
            q[e_list_idx] = [q[e_list_idx]]
            for q_name in grouped_qs:
                q[e_list_idx].append(grouped_qs[q_name][e_list_idx])

        ticks = []
        model_no = [len(q[-1][-1]), len(q)]  # [ len(inactive mode), len(active mode) ]
        i = 0
        bps = []
        assert len(q) > 0
        for e_lists in q:
            model_name = e_lists[0][0].name.split("_")[mode]
            inner_length = len(e_lists[-1])
            ticks.append(model_name)
            boxplot, sorted_experiments = self._get_boxplot_from_elists(e_lists=e_lists, metric=metric, pos=self._get_positions(i, inner_length, model_no[0]),
                                                                        mode=mode, vert=vert, selected_methods=selected_methods)
            bps.append(boxplot)
            i += 1
        if abbreviate:
            ticks = self._abbreviate(ticks)
        if not vert:
            plt.yticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks)
            plt.ylim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        else:
            plt.xticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks, rotation=45)
            plt.xlim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        if legend:
            self._add_legend(sorted_experiments, bps, mode, abbreviate=abbreviate)
        plt.savefig("result_grafics/" + "All" + "_" + self.get_metric_name(metric, short=True) + ".pdf", bbox_inches='tight')
        plt.show()

    def export_all_as_csv(self, name="results", no_auc=False):
        exps: List[Experiment] = self.exp_man.experiments
        cols = ["dataset-name", "N", "L", "generator", "metamodel", "SD", "fragment-id", "WRacc_box", "WRAcc_box_resdims", "precision_box", "precision_box_resdim",
                "WRacc", "Precision", "AUC"]
        all_exps = []
        for e in exps:
            fragments = []
            for f_res in e.result:
                data = [
                    e.ex_data.name,
                    e.ex_data.fragment_size,
                    e.new_sample_size,
                    e.name.split("_")[0],
                    e.name.split("_")[1],
                    e.name.split("_")[2],
                    f_res.fragment_idx,
                    f_res.highest_wracc_box[0],
                    f_res.restricted_dims[f_res.highest_wracc_idx],
                    f_res.highest_mean_box[0],
                    f_res.restricted_dims[f_res.highest_mean_idx],
                    f_res.highest_wracc_box[1],
                    f_res.highest_mean_box[1],
                    0
                ]
                fragments.append(data)
            if e.name.split("_")[2] == "prim" and not no_auc:
                aucs = self.peeling_trajectory_auc_metric(e)
                for i in range(len(fragments)):
                    fragments[i][-1] = aucs[i]
            all_exps += fragments
        pd.DataFrame(all_exps, columns=cols).to_csv("output/" + name + ".csv", sep=";", index=False)

    def boxplot_selected_methods(self, metric, selected_methods: List[str], name: str, colors=None, title=None, abbreviate=False, vert=True, legend=True,
                                 legend_above=False):
        qs = self.exp_man.queues.copy()
        for q_name in qs:
            filtered = list(filter(lambda e: self._name_matches_any(e, selected_methods), qs[q_name]))
            qs[q_name] = self._sort_by_methods(filtered, selected_methods)
        self._plot_datasets_together(qs, metric, selected_methods, name, vert=True, title=title, abbreviate=abbreviate, legend=legend, colors=colors,
                                     legend_above=legend_above)

    def plot_detailed_curve(self, metric, selected_methods: List[str], colors: List[str], steps_range, name: str, title=None, w_variance=False, median=False,
                            legend=False):
        assert self.detailed_mode
        qs: Dict[str, Any] = self.exp_man.queues
        methods_list = self._map_methods_list_and_apply_metric(qs, selected_methods, metric)
        for i in range(len(selected_methods)):
            if not median:
                means = self._aggregate(methods_list[i], np.mean)
            else:
                means = self._aggregate(methods_list[i], np.median)
            plt.plot(list(steps_range), means, color=colors[i], label=selected_methods[i])
            if w_variance:
                stds = np.array(list(map(lambda e_list: np.std(e_list), methods_list[i])))
                plt.fill_between(list(steps_range), means + stds, means - stds,
                                 alpha=0.2, edgecolor=colors[i], facecolor=colors[i], linewidth=2, antialiased=True)
        ex0: Experiment = list(qs.values())[0][0]  # take the first experiment to read general parameters
        if title is not None:
            plt.title(title)
        else:
            plt.title(ex0.ex_data.name)
        if legend:
            plt.legend(loc='best')
        desc = "SD algorithm:" + ex0.name.split("_")[2] + "; fragment limit = " \
               + str(ex0.fragment_limit) + "; generated points = " + str(ex0.new_sample_size)
        plt.ylabel(self.get_metric_name(metric))
        plt.xlabel("Number of points for fragment size |d| \n\n" + desc)
        plt.savefig("result_grafics/" + name + "_" + self.get_metric_name(metric, short=True) + "_"
                    + "f_lim" + str(ex0.fragment_limit) + "_gen_pt" + str(ex0.new_sample_size) + ".pdf", bbox_inches='tight')
        plt.show()

    def points_needed_ratio(self, metric, selected_methods: List[str], steps_range, median=False, epsilon=0.05):
        """Compare two methods how many points they need to produce similar results: A needs 2 times more points than B to produce similar results"""
        assert self.detailed_mode
        assert selected_methods is not None and len(selected_methods) == 2
        qs = self.exp_man.queues
        methods_list = self._map_methods_list_and_apply_metric(qs, selected_methods, metric)

        if not median:
            dummy_vals: np.ndarray = self._aggregate(methods_list[0], np.mean)
            method_vals: np.ndarray = self._aggregate(methods_list[1], np.mean)
        else:
            dummy_vals: np.ndarray = self._aggregate(methods_list[0], np.median)
            method_vals: np.ndarray = self._aggregate(methods_list[1], np.median)

        assert dummy_vals[0] <= method_vals[0]

        number_of_points: int = steps_range[0]

        idx_similar = self._find_first_higher_in_list(method_vals[0], list(dummy_vals), epsilon)
        number_of_points_dummy = steps_range[idx_similar]

        print("Points Method:", str(steps_range[0]), "Value method:", str(method_vals[0]))
        print("Points Dummy:", str(steps_range[idx_similar]), "Value Dummy:", str(dummy_vals[idx_similar]))
        print("Difference:", str(method_vals[0] - dummy_vals[idx_similar]))
        print("Ratio:", str(number_of_points / number_of_points_dummy))

    def _aggregate(self, e_lists: List[List[float]], function) -> np.ndarray:
        return np.array(list(map(lambda e_list: function(e_list), e_lists)))

    def _find_first_higher_in_list(self, value, target_list: List[float], epsilon: float) -> int:
        """Finds the """
        bounds = None
        for val_idx in range(len(target_list)):
            if target_list[val_idx] > value * (1 - epsilon):
                bounds = target_list[val_idx - 1: val_idx + 1]
                break
        assert bounds is not None
        lower_or_upper = self._find_closest(value, bounds[0], bounds[1])
        return val_idx -lower_or_upper

    def _find_closest(self, value: float, a: float, b: float) -> int:
        """Returns 0 if a is closer to the value or 1 if not"""
        a_diff, b_diff = abs(value - a), abs(value - b)
        return 0 if a_diff > b_diff else 1

    def _trajectory_from_fragment(self, fragment_result, color, linestyle, alpha, marker=None):
        cov = list(map(lambda kpis: kpis[coverage_key], fragment_result.kpis))
        density = list(map(lambda kpis: kpis[density_key], fragment_result.kpis))
        plt.plot(cov, density, color=color, linestyle=linestyle, alpha=alpha, marker=marker)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("coverage")
        plt.ylabel("denisty")

    def plot_trajectory_from_fragment(self, fragment_result, color, linestyle, alpha, marker=None):
        self._trajectory_from_fragment(fragment_result, color, linestyle, alpha, marker=marker)
        plt.show()

    def plot_trajectories_from_experiment(self, experiment, color, linestyle, alpha, marker=None):
        for f_res in experiment.result:
            self._trajectory_from_fragment(f_res, color, linestyle, alpha, marker=marker)
        print("Trajectory plot created for " + experiment.name + "; fsize: " + str(experiment.ex_data.fragment_size)) + "\n Use plt.show() to display"

    def export_avg_metrics_to_csv(self, metric, abbreviate=True, filename=None, median=False, relative=True):
        qs = self.exp_man.get_grouped_by(1)
        any_q = list(qs.values())[0]
        generators = [e.name.split("_")[0] for e in any_q[-1]]
        metamodels = [e_list[-1].name.split("_")[1] for e_list in any_q]
        table = pd.DataFrame(0, columns=metamodels, index=generators)
        for q_name in qs:
            dummy_dummy_performance = np.mean(metric(qs[q_name][0][0]))
            assert type(qs[q_name][0][0].generator) == DummyGenerator and type(qs[q_name][0][0].metamodel) == DummyMetaModel
            for e_list in qs[q_name]:
                for e in e_list:
                    generator_name, metamodel_name = e.name.split("_")[0], e.name.split("_")[1]

                    if not median:
                        performance = np.mean(metric(e))
                    else:
                        performance = np.median(metric(e))
                    table.loc[generator_name, metamodel_name] += performance / dummy_dummy_performance
        table /= len(qs)
        if abbreviate:
            table.rename(columns=abbreviations_dict, index=abbreviations_dict, inplace=True)
        plt.subplots(figsize=(12,6))
        table = table.drop(columns=["dummy"]).astype(float)
        if filename is not None:
            table.to_csv("result_grafics/" + filename)
        sns.heatmap(table, cmap="Blues", annot=True, fmt='.4g')
        plt.title(self.get_metric_name(metric))
        plt.savefig("result_grafics/heatmap.pdf", bbox_inches='tight')
        plt.show()

    def export_to_csv(self, metric, abbreviate=True, filename=None, median=False):
        qs = self.exp_man.get_grouped_by(1)
        any_q = list(qs.values())[0]
        generators = [e.name.split("_")[0] for e in any_q[-1]]
        metamodels = [e_list[-1].name.split("_")[1] for e_list in any_q]
        table = pd.DataFrame(0, columns=metamodels, index=generators)
        for q_name in qs:
            dummy_dummy_performance = np.mean(metric(qs[q_name][0][0]))
            assert type(qs[q_name][0][0].generator) == DummyGenerator and type(qs[q_name][0][0].metamodel) == DummyMetaModel
            for e_list in qs[q_name]:
                for e in e_list:
                    generator_name, metamodel_name = e.name.split("_")[0], e.name.split("_")[1]

                    if not median:
                        performance = np.mean(metric(e))
                    else:
                        performance = np.median(metric(e))
                    table.loc[generator_name, metamodel_name] += performance
        table /= len(qs)
        if abbreviate:
            table.rename(columns=abbreviations_dict, index=abbreviations_dict, inplace=True)
        table.to_csv("result_grafics/" + filename)

    def _map_to_metric_value(self, qs, selected_methods, metric, mode):
        new_qs = qs
        for q_name in qs:
            for q_idx in range(len(qs[q_name])):
                q = qs[q_name][q_idx]
                for e_list_idx in range(len(q)):
                    e_list = q[e_list_idx]
                    qs[q_name][q_idx][e_list_idx] = np.array(self._map_method_list(e_list, selected_methods, metric, mode))
        return new_qs

    def _leftadd_q(self, q_a: List[List[List[np.array]]], q_b: List[List[List[np.array]]]):
        new_q = q_a
        for i in range(len(q_a)):
            for j in range(len(q_a[i])):
                new_q[i][j] += q_b[i][j]
        return new_q

    def _apply_scalar_to_q(self, q, factor):
        new_q = q
        for i in range(len(q)):
            for j in range(len(q[i])):
                new_q[i][j] *= factor
        return new_q

    def _map_methods_list_and_apply_metric(self, qs: dict, selected_methods: List[str], metric) -> List[List[List[float]]]:
        methods_list = [[] for m in selected_methods]
        for q_name in qs:
            for e in qs[q_name]:
                pos = self._name_at_position(e, selected_methods)
                if pos > -1:
                    methods_list[pos].append(metric(e))
        assert all(methods_list)
        return methods_list

    def _map_method_list(self, e_list, selected_methods: List[str], metric, mode) -> List[List[float]]:
        mapped_list = [[] for m in selected_methods]
        for e in e_list:
            pos = self._name_at_position(e, selected_methods, mode)
            if pos > -1:
                mapped_list[pos].append(metric(e))
        assert all(mapped_list)
        return mapped_list

    def _sort_by_methods(self, e_list: List[Experiment], selected_methods) -> List[Experiment]:
        result_q: List[Union[Experiment]] = list(np.repeat(None, len(e_list)))
        for e in e_list:
            pos = self._name_at_position(e, selected_methods)
            assert pos > -1
            result_q[pos] = e
        assert all(result_q)
        return result_q

    def _sort_by_method(self, e_list: List[Experiment], selected_methods, mode) -> List[Experiment]:
        # TODO Implmenent filter
        result_q: List[Union[Experiment]] = list(np.repeat(None, len(e_list)))
        for e in e_list:
            pos = self._name_at_position(e, selected_methods, mode)
            assert pos > -1
            result_q[pos] = e
        assert all(result_q)
        return result_q

    def _plot_datasets_together(self, qs: dict, metric, selected_methods, name, title: str = None, vert=False, legend=True, colors=None, abbreviate=False,
                                legend_above=False):
        plt.figure(figsize=(14, 7))
        if title is not None: plt.title(title)
        if not vert:
            plt.xlim((0, self.boxplot_top_offset))
            plt.xlabel(self.get_metric_name(metric))
        else:
            plt.ylim((0, self.boxplot_top_offset))
            plt.ylabel(self.get_metric_name(metric))

        ticks = list(qs.keys())
        model_no = [len(selected_methods), len(ticks)]  # [ len(inactive mode), len(active mode) ]
        i = 0
        bps = []
        for q_name in qs:
            boxplot, sorted_experiments = self._get_boxplot_from_elist(qs[q_name], metric, self._get_positions(i, len(qs[q_name]), model_no[0]), mode=1,
                                                                       vert=vert, different_dummy_colors=True, colors=colors)
            bps.append(boxplot)
            i += 1
        if not vert:
            plt.yticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks)
            plt.ylim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        else:
            plt.xticks(range(0, (model_no[0] + self.offset_of_groups) * model_no[1], model_no[0] + self.offset_of_groups), ticks, rotation=45)
            plt.xlim(- (model_no[0] + self.offset_of_groups), model_no[1] * (model_no[0] + self.offset_of_groups))
        if legend:
            self._add_legend(qs[q_name], bps, 1, show_both_methods=True, abbreviate=True, above=legend_above)
        plt.savefig("result_grafics/" + name + "_" + self.get_metric_name(metric, short=True) + ".pdf", bbox_inches='tight')
        plt.show()

    def _abbreviate(self, names: List[str]) -> List[str]:
        return list(map(lambda name: abbreviations_dict[name], names))

    def _name_at_position(self, experiment: Experiment, methods: List[str], mode=None) -> int:
        if self._name_matches_any(experiment, methods, mode):
            pos = [self._name_matches(experiment, method, mode=mode) for method in methods].index(True)
        else:
            pos = -1
        return pos

    def _name_matches_any(self, experiment: Experiment, methods: List[str], mode=None):
        return any([self._name_matches(experiment, method, mode) for method in methods])

    def _name_matches(self, experiment: Experiment, matching_string: str, mode=None):
        methods = experiment.name.split("_")
        generator_name, metamodel_name = methods[0], methods[1]
        if mode is None:
            matches = (generator_name + "_" + metamodel_name) == matching_string
        else:
            matches = [generator_name, metamodel_name][1 - mode] == matching_string
        return matches

    def _get_positions(self, y_idx, group_size, max_group_size):
        getcontext().prec = 6

        center = Decimal(y_idx * (max_group_size + self.offset_of_groups))
        step_number = Decimal(group_size)
        step = Decimal(self.offset_in_group)

        max_dist = (step_number - 1) / 2 * step
        r = np.arange(center - max_dist, center + max_dist + step, step)
        assert group_size == len(r)
        return r.astype(dtype=np.number)

    def _get_boxplot_from_elist(self, experiments: List[Experiment], metric, pos, mode=None, selected_methods=None, vert=False,
                                different_dummy_colors=False, colors=None) -> Tuple:
        assert len(experiments) > 0
        if selected_methods is not None:
            experiments = self._sort_by_method(experiments, selected_methods, mode)
        max_values_list = []
        for e in experiments:
            if e.result is not None:
                max_values = metric(e)
                max_values_list.append(max_values)
        data = list(max_values_list)
        b = plt.boxplot(data, patch_artist=True, positions=pos, sym=".",
                        widths=self.box_width, vert=vert, autorange=True)
        if mode is not None:
            self._colorize(experiments, b, mode, different_dummy_colors, custom_colors=colors)
        return b, experiments

    def _get_boxplot_from_elists(self, e_lists: List[List[Experiment]], metric, pos, mode=None, selected_methods=None,
                                 vert=False, different_dummy_colors=False) -> Tuple:
        e_lists_copy = e_lists.copy()
        metric_values_list = []
        if selected_methods is not None:
            e_lists_copy = [self._sort_by_method(some_elist, selected_methods, mode) for some_elist in e_lists_copy]
        assert len(e_lists_copy) > 0
        some_elist = e_lists_copy[0]
        for e_idx in range(len(some_elist)):
            avgs = [np.mean(metric(e_list[e_idx])) for e_list in e_lists_copy]
            metric_values_list.append(avgs)
        data = list(metric_values_list)
        b = plt.boxplot(data, patch_artist=True, positions=pos, sym=".",
                        widths=self.box_width, vert=vert, autorange=True)
        if mode is not None:
            self._colorize(some_elist, b, mode, different_dummy_colors)
        return b, some_elist

    def _colorize(self, experiments, boxplot, mode, different_dummy_colors=False, custom_colors=None):
        mode = 1 - mode  # Colorize the opposite
        color_dicts = [gen_c, metamodels_c]
        if custom_colors is None:
            colors = list(map(lambda e: self.get_color_for_method(e, color_dicts, mode, different_dummy_colors), experiments))
        else:
            colors = custom_colors
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set(color=color, linewidth=2)
            patch.set(facecolor=color)

    def get_color_for_method(self, e: Experiment, color_dicts: List[dict], mode: int, different_dummy_colors: bool = False):
        if different_dummy_colors and type(e.generator) == DummyGenerator and type(e.metamodel) == DummyMetaModel:
            color = "lightgrey"
        else:
            color = color_dicts[mode][e.name.split("_")[mode]]
        return color

    def _add_legend(self, grouped_e_list, bps, mode, show_both_methods=False, abbreviate=False, above=False):
        assert len(grouped_e_list) > 0
        if not show_both_methods:
            names = list(map(lambda e: e.name.split("_")[1 - mode], grouped_e_list))
            if abbreviate:
                names = list(map(lambda name: abbreviations_dict[name], names))
        else:
            if abbreviate:
                names = list(map(lambda e: abbreviations_dict[e.name.split("_")[0]] + " & " + abbreviations_dict[e.name.split("_")[1]], grouped_e_list))
            else:
                names = list(map(lambda e: e.name.split("_")[0] + " & " + e.name.split("_")[1], grouped_e_list))
        mode = 1 - mode  # Colorize the opposite
        bps = [bps[-1]["boxes"][i] for i in range(len(names))]
        # Place a legend to the right of this smaller subplot.
        n_col = math.ceil(len(grouped_e_list) / 2)
        if not above:
            plt.legend(bps, names, loc=1, frameon=False, ncol=n_col)
        else:
            plt.legend(bps, names, frameon=False, ncol=len(names), loc='upper center', bbox_to_anchor=(0,1.02,1,0.2))

    def highest_box_metric(self, experiment: Experiment) -> List[float]:
        return [f_res.highest_mean_box[1] for f_res in experiment.result]

    def highest_f1_metric(self, experiment: Experiment) -> List[float]:
        return [f_res.highest_f1_box[1] for f_res in experiment.result]

    def highest_wracc_metric(self, experiment: Experiment) -> List[float]:
        return [f_res.highest_wracc_box[1] for f_res in experiment.result]

    def highest_f2_metric(self, experiment: Experiment) -> List[float]:
        return [f_res.highest_f2_box[1] for f_res in experiment.result]

    def restricted_dims_on_highest_metric(self, experiment: Experiment):
        return [(len(f_res.highest_mean_box[1]) / f_res.initial_restrictions_train.shape[1]) for f_res in experiment.result]

    def restricted_dims_on_highestf2_metric(self, experiment: Experiment):
        return [(len(f_res.highest_f2_box[1]) / f_res.initial_restrictions_train.shape[1]) for f_res in experiment.result]

    def restricted_dims_on_highestf1_metric(self, experiment: Experiment):
        return [(len(f_res.highest_f1_box[1]) / f_res.initial_restrictions_train.shape[1]) for f_res in experiment.result]

    def consistency_v_highest_metric(self, experiment: Experiment) -> List[float]:
        consistency = []
        res_0 = experiment.result[0]
        full_res_0: pd.DataFrame = res_0.to_restriction(res_0.highest_mean_idx)
        for f_res in experiment.result[1:]:
            full_res_i = f_res.to_restriction(f_res.highest_mean_idx)
            consistency.append(self._compute_overlap(full_res_0, full_res_i))
        return consistency

    def consistency_v_highestf1_metric(self, experiment: Experiment) -> List[float]:
        consistency = []
        res_0 = experiment.result[0]
        full_res_0: pd.DataFrame = res_0.to_restriction(res_0.highest_f1_idx)
        for f_res in experiment.result[1:]:
            full_res_i = f_res.to_restriction(f_res.highest_f1_idx)
            consistency.append(self._compute_overlap(full_res_0, full_res_i))
        return consistency

    def _compute_overlap(self, box_res_a: pd.DataFrame, box_res_b: pd.DataFrame):
        # TODO Talk to Vadim about this!
        inter = self._intersection(box_res_a, box_res_b)
        sides1 = box_res_a.iloc[1, :].to_numpy() - box_res_a.iloc[0, :].to_numpy()
        sides2 = box_res_b.iloc[1, :].to_numpy() - box_res_b.iloc[0, :].to_numpy()
        sides_inter = inter.iloc[1, :].to_numpy() - inter.iloc[0, :].to_numpy()
        if any(sides_inter <= 0):
            res = 0
        else:
            res = 1 / (np.prod(sides1 / sides_inter) + np.prod(sides2 / sides_inter) - 1)
        return res

    def _intersection(self, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        merged: pd.DataFrame = a.append(b)
        upper_bounds: pd.DataFrame = merged.iloc[[1, 3], :]
        lower_bounds: pd.DataFrame = merged.iloc[[0, 2], :]
        lower = lower_bounds.max(axis=0)
        upper = upper_bounds.min(axis=0)
        inter_res = pd.concat([lower, upper], axis=1).transpose()
        return inter_res

    def consistency_d_highest_metric(self, experiment: Experiment) -> List[float]:
        # TODO This method goes way too deep, add methods in fragment result!
        consistency = []
        res_0 = pd.DataFrame(experiment.result[0].highest_f1_box[0])
        b_0 = BoxResult(experiment.result[0].experiment_dataset.get_subset_compound(0).get_complete_complement(), res_0)
        for i in range(1, len(experiment.result)):
            res_i = pd.DataFrame(experiment.result[i].highest_f1_box[0])
            b_i = BoxResult(experiment.result[i].experiment_dataset.get_subset_compound(i).get_complete_complement(), res_i)
            v_u = sum(b_0.in_box_idxs | b_i.in_box_idxs)
            v_i = sum(b_0.in_box_idxs & b_i.in_box_idxs)
            consistency.append(v_i / v_u)
        return consistency

    def consistency_d_leftmost_metric(self, experiment: Experiment) -> List[float]:
        # TODO This method goes way too deep, add methods in fragment result!
        consistency = []
        res_0 = pd.DataFrame(experiment.result[0].min_mass_box[0])
        b_0 = BoxResult(experiment.result[0].experiment_dataset.get_subset_compound(0).get_complete_complement(), res_0)
        for i in range(1, len(experiment.result)):
            res_i = pd.DataFrame(experiment.result[i].min_mass_box[0])
            b_i = BoxResult(experiment.result[i].experiment_dataset.get_subset_compound(i).get_complete_complement(), res_i)
            v_u = b_0.in_box_idxs | b_i.in_box_idxs
            v_i = b_0.in_box_idxs & b_i.in_box_idxs
            consistency.append(v_i / v_u)
        return consistency

    def peeling_trajectory_auc_metric(self, experiment: Experiment) -> List[float]:
        aocs = []
        for f_res in experiment.result:
            result_boxes = f_res.kpis
            kpis: List[Tuple[int, int]] = list(map(lambda kpi: (kpi[coverage_key], kpi[density_key]), result_boxes))
            kpis.sort(key=lambda kpi: kpi[0], reverse=True)
            x = list(map(lambda box: box[0], kpis))
            y = list(map(lambda box: box[1], kpis))
            x.append(0)  # Add 0 dummy to get full auc
            y.append(y[-1])

            x = pd.Series(x)
            y = pd.Series(y)
            traj_auc = auc(x, y) - y[0]
            aocs.append(traj_auc)
        return aocs

    def get_metric_name(self, metric, short=False) -> str:
        if metric == self.peeling_trajectory_auc_metric:
            return "Area under curve" if not short else "AUC"
        elif metric == self.highest_box_metric:
            return "Highest density" if not short else "hd"
        elif metric == self.restricted_dims_on_highest_metric:
            return "Restricted dimensions on box with highest density" if not short else "resdim_d"
        elif metric == self.restricted_dims_on_highestf1_metric:
            return "Restricted dimensions on box with highest f1" if not short else "resdim_f1"
        elif metric == self.restricted_dims_on_highestf2_metric:
            return "Restricted dimensions on box with highest f2" if not short else "resdim_f2"
        elif metric == self.consistency_d_highest_metric:
            return "Consistency box with HD" if not short else "cons_hd_d"
        elif metric == self.consistency_d_leftmost_metric:
            return "Consistency leftmost box" if not short else "cons_left_d"
        elif metric == self.consistency_v_highest_metric:
            return "Consistency box with HD" if not short else "cons_hd"
        elif metric == self.consistency_v_highestf1_metric:
            return "Consistency box with F1" if not short else "cons_f1"
        elif metric == self.highest_f1_metric:
            return "F1-score" if not short else "F1"
        elif metric == self.highest_wracc_metric:
            return "WRAcc"
        else:
            return "F2-score" if not short else "F2"