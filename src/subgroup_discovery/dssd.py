import time
from typing import List, Union

import pandas as pd
import numpy as np
import subprocess
import os
import shutil
import arff


class Config:
    
    def __init__(self, dsName,  maxTime = 0, takeItEasy = 0, maxDepth = 5, topK = 10000, minCoverage = 10, floatNumSplits = 5, beamWidth = 100):
        ### DSSD configuration file ###
        # Class
        self.taskclass = "diverse"
        # Command
        self.command = "dssd"
        # TakeItEasy(tm) -- ( 0 | 1 ) If enabled, process runs with low priority.
        self.takeItEasy = takeItEasy

        ## Basics
        # Dataset
        self.dsName = dsName
        # Maximum time (in minutes; 0 for no maximum)
        self.maxTime = maxTime
        # Save intermediate results after each level (bfs,beam only)
        self.outputIntermediate = 1
        # Save subsets of final resultset
        self.outputSubsets = 1
        # Save models belonging to subgroups of final resultset
        self.outputModels = 0
        # Log beam selection that is made on each level
        self.beamLogSelection = 0
        # How to deal with induced subgroup models; caching costs more memory but is faster [cache | rebuild]
        self.sgModels = "cache"

        ## Controlling the 3 phases
        # Phase 1: Number of results to keep during initial search phase
        self.topK = topK
        # Phase 2: Post-processing methods to apply (any of [dominance, equalcover, qualitysort], separated by -)
        self.postProcess = "dominance-qualitysort"
        # Phase 3: Subgroup set selection (0 to disable, any other integer specifies number of desired results)
        self.postSelect = 100
        # Selection strategy to be used for post-selection (specify only if different from `beamStrategy')
        #postSelectBeam = cover

        ### Search parameters
        # [beam | dfs | iter-0.x-[dfs|beam] (x=0 -> sequential covering, weighted covering otherwise)]
        self.searchType = "beam"
        # Maximum depth for the search (ie the maximum number of conditions in a subgroup description)
        self.maxDepth = maxDepth
        # Minimum subgroup size
        self.minCoverage = minCoverage
        # Number of split points for on-the-fly discretisation; numeric values splitted into floatNumSplits+1 intervals
        self.floatNumSplits = floatNumSplits

        ## Beam search settings
        # Beam selection strategy [quality | description | cover | compression]
        # `Quality' is the standard top-k search, the other three correspond to the DSSD strategies
        self.beamStrategy = "quality"
        # Fixed beam width, or maximum beam width when variable beam width is used
        self.beamWidth = beamWidth
        # Variable beam width (disabled when set to 'false', which is the default; effect depends on beamStrategy)
        self.beamVarWidth = "false"
        # Multiplicative weight covering multiplier for cover-based selection strategy
        self.coverBeamMultiplier = 0.9
        # Cover strategy for cover-based selection strategy [sequential | multiplicative | additive]
        # (For the DSSD experiments, this was always set to multiplicative)
        self.coverBeamStrategy = "multiplicative"

        ### Quality
        ## Quality measure to use [WRAcc | WKL | KL | meantest | ChiSquared]
        self.measure = "WRAcc"

        ## WRAcc 
        # WRAcc mode [single | 1vsAll(default) | 1vsAllWeighted | 1vs1]
        self.WRAccMode = "single"

    def to_string(self) -> str:
        self_string = self.__dict__
        formatted_string = replace_s(str(self_string))
        return formatted_string

    def __alt_to_string(self):
        """Alternative string method if the other one does not work"""
        config_dict = self.__dict__
        string_builder = ""
        for attr in config_dict:
            string_builder += attr + " = " + config_dict[attr] + "\n"


def replace_s(input_string: str):
    for key in replace_dict.keys():
        input_string = input_string.replace(key, replace_dict[key])
    return input_string


replace_dict = {
    "'": "",
    "{": "",
    "}": "",
    ":": " =",
    ",": "\n",
}


class DSSD:

    def __init__(self, dsName, number_of_rules: int = 1, maxTime = 0, takeItEasy = 0, maxDepth = 5, topK = 10000,
                 minCoverage = 10, floatNumSplits = 5, beamWidth = 100):
        self.config = Config(dsName, maxTime, takeItEasy, maxDepth, topK, minCoverage, floatNumSplits, beamWidth)
        self.dssd_root = "src/subgroup_discovery/dssd/"
        self.data_file = self.dssd_root + "data/datasets/"
        self.number_of_rules = number_of_rules
    
    def find(self, X: pd.DataFrame, y: np.ndarray, regression=True):
        key = self.config.dsName
        self.write_config_to_disk(key)
        self.write_dataset_to_disk(key=key, X=X, y=y, y_name="y")
        self.launch_dssd(key=key)
        df = self.get_results(key=key)
        df = df.nlargest(self.number_of_rules, columns=["quality"])
        initial_restriction = get_initial_df(0, 1, X)
        box_lims = [initial_restriction]
        for index, row in df.iterrows():
            # TODO currently no explicit bounds computation to save time!
            box_lims.append(parse_rule(row["description"], initial_restriction))
        self.cleanup(key)
        return box_lims

    def write_dataset_to_disk(self, key: str, X: pd.DataFrame, y, y_name: str, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = []
        data_dir: str = self.data_file + key + "/"
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        data = X.copy()
        data.insert(data.shape[1], y_name, y, allow_duplicates=True)
        arff_dict = as_arff_dict(X, y)
        # TODO change mode to "x"
        # arff.dumps writes uppercase types, but dssd requires the types to be lowercase
        with open(data_dir + key + ".arff", "w") as f:
            uppercase = arff.dumps(arff_dict)
            lowercase = uppercase.replace("NUMERIC", "numeric")
            f.write(lowercase)

        with open(data_dir + key + ".emm", "w") as emm_file:
            emm_file.write(self._get_emm(data, exclude_columns, target=y_name))

    def write_config_to_disk(self, key: str):
        config_string = self.config.to_string()
        #TODO change to x
        with open(self.dssd_root + "bin/" + key + ".conf", "w") as file:
            file.write(config_string)

    def _get_emm(self, data: pd.DataFrame, exlude_list: List[str], target) -> str:
        """generate .emm file content for *single target* """
        result = "descriptionAtts = "
        if len(exlude_list) == 0:
             result += "*"
        else:
            filtered_cols = list(filter(lambda col_name: exlude_list.__contains__(col_name) or col_name == target, data.columms))
            filtered_cols_str = str(filtered_cols).replace("[", "").replace("]", "").replace(" ", "")
            result += filtered_cols_str
        result += "\nmodelAtts = "
        return result + target

    def launch_dssd(self, key):
        print(time.localtime())
        w_dir = "src/subgroup_discovery/dssd/bin/"
        p = subprocess.Popen("dssd " + key + ".conf", cwd=w_dir, shell=True)
        p.wait()

    def get_results(self, key: str) -> pd.DataFrame:
        result_folder_path = self.get_result_folder_name(key)
        sg_df: pd.DataFrame = self.get_stats_as_df(target_folder=result_folder_path)
        return sg_df

    def get_result_folder_name(self, substring: str) -> str:
        """Returns the subfolder under xps which contains the resulting subgroups"""
        result_folders: List[str] = [f.path for f in os.scandir(self.dssd_root + "xps/dssd") if f.is_dir()]
        result = None
        for sub_f in result_folders:
            # TODO This could be a regex, might be problematic with unix/win paths
            # Under win10 the last subfolder is indicated by a double backslash
            # C:/some/path\\actual_folder
            if sub_f.find(substring) > 0:
                result = sub_f.split("\\")[-1]
        return result

    def get_stats_as_df(self, target_folder: str) -> pd.DataFrame:
        """Get all subgroups as pd.DataFrame from a target foler"""
        result_folders: List[str] = [f.path for f in os.scandir(self.dssd_root + "xps/dssd/" + target_folder)]
        last_file = None
        for sub_f in result_folders:
            # TODO This could be a regex, might be problematic with unix/win paths
            # It really shouldn't exceed 100 stats-files - otherwise it probably won't terminate
            for i in range(1, 100):
                if sub_f.find("stats") > 0:
                    last_file = sub_f
        assert last_file is not None
        stats = pd.read_csv(last_file, sep=";")
        return stats

    def cleanup(self, key: str):
        shutil.rmtree(self.dssd_root + "data/datasets/" + key)
        shutil.rmtree(self.dssd_root + "xps/dssd/" + self.get_result_folder_name(key))
        os.remove(self.dssd_root + "bin/" + key + ".conf")


def as_arff_dict(X: pd.DataFrame, y: np.ndarray, regression=True) -> dict:
    attributes = [(c, 'NUMERIC') for c in X.columns.values]
    target_type = 'NUMERIC' if regression else to_set_string(y)
    attributes += [("y", target_type)]

    if type(y) == pd.Series:
        y = y.to_numpy()

    data_np = np.append(X.to_numpy(), y.reshape((y.shape[0], 1)), axis=1)

    arff_dic = {
    'attributes': attributes,
    'data': data_np,
    'relation': 'name',
    'description': ''
    }
    return arff_dic


def to_set_string(array: np.ndarray) -> str:
    return str(np.unique(array).tolist())\
                .replace(", ", ",")\
                .replace("[", "{")\
                .replace("]", "}")


def get_initial_df(lower_bound: Union[int, float], upper_bound : Union[int, float], data: pd.DataFrame):
    lower = np.repeat(lower_bound, data.shape[1])
    upper = np.repeat(upper_bound, data.shape[1])
    return pd.DataFrame((lower, upper), columns=data.columns)


def parse_rule(dssd_rule_string: str, initial_restriction: pd.DataFrame) -> pd.DataFrame:
    conditions = dssd_rule_string.split(sep=" && ")
    restriction = initial_restriction.copy()
    for cond in conditions:
        cond_parts = cond.split(" ")
        attribute, operator, value = cond_parts[0], cond_parts[1], cond_parts[2]  # Attr > Value
        operator = 1 if operator == "<" else 0
        restriction.loc[operator, attribute] = value
    return restriction.astype(dtype='float')
