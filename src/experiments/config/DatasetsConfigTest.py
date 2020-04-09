import logging
from typing import *
from src.experiments.config.DataUtils import map_target, clean, generate_names

import pandas as pd

from src.experiments.model.ExperimentDataset import ExperimentDataset

datasets: List[ExperimentDataset] = []
large_datasets = []

standard_f_size = 300
logger = logging.getLogger("DATA")
logger.setLevel("INFO")

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# click
click = pd.read_csv("resources/data/click_prediction.csv")
click_yname = "click"
click = click.dropna()
# nomapping
datasets.append(ExperimentDataset("click", click, click_yname, fragment_size=standard_f_size))
