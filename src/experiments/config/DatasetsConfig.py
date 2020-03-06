import logging

import pandas as pd
from scipy.io import arff

from src.experiments.model.ExperimentDataset import ExperimentDataset

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


large_datasets = []

def map_target(data: pd.DataFrame, y_col_name: str, to_ones):
    data.loc[:, y_col_name] = data.loc[:, y_col_name].map(lambda e: 1 if e == to_ones else 0)
    return data


def clean(data: pd.DataFrame, value):
    for row in range(data.shape[0]):
        if data.loc[row,:].isin([value]).any():
            data = data.drop(index=row)
    return data

# Seizures
#seizure: pd.DataFrame = pd.read_csv('resources/data/seizure.csv', header=0, index_col=0)
#seizure_yname = "y"
#seizure = map_target(seizure, seizure_yname, 1)
#large_datasets.append(ExperimentDataset("seizures_mapped_50", seizure, seizure_yname, fragment_size=standard_f_size))
#logger.info("1 Dateset loaded")


# # electricity
electricity = pd.read_csv("resources/data/electricity-normalized.csv")
electricity_yname = "class"
electricity = electricity.dropna()
map_target(electricity, electricity_yname, "UP")
large_datasets.append(ExperimentDataset("electricity", electricity, electricity_yname, fragment_size=standard_f_size))
logger.info("1 Datesets loaded")


# # higgs
higgs = pd.read_csv("resources/data/higgs.csv", low_memory=False, na_values=['?'])
higgs_yname = "class"
higgs = higgs.dropna()
higgs.rename(columns={'jet1b-tag': 'jet1btag',
                    'jet2b-tag': 'jet2btag',
                    'jet3b-tag': 'jet3btag',
                    'jet4b-tag': 'jet4btag'}, inplace=True)
large_datasets.append(ExperimentDataset("higgs", higgs, higgs_yname, fragment_size=standard_f_size))
logger.info("2 Datesets loaded")


# # nomao - 7,8,15,16,24,31,32,39,40,47,48,55,56,63,64,72,80,87,88,92,96,100,104,108,112,116 nominals
# nomao = pd.read_csv("resources/data/nomao.csv")
# nomao = nomao.drop(columns=['V7','V8','V15','V16','V24','V31','V32','V39','V40','V47','V48','V55','V56','V63','V64','V72','V80','V87','V88',
#                                   'V92','V96','V100','V104','V108','V112','V116'])
# nomao_yname = "Class"
# nomao = map_target(nomao, nomao_yname, 1)
# large_datasets.append(ExperimentDataset("nomao", nomao, nomao_yname, fragment_size=standard_f_size))
# logger.info("4 Datesets loaded")


# # numerai
numerai = pd.read_csv("resources/data/numerai.csv")
numerai_yname = "attribute_21"
numerai = numerai.dropna()
# nomapping
large_datasets.append(ExperimentDataset("numerai", numerai, numerai_yname, fragment_size=standard_f_size))
logger.info("3 Datesets loaded")


# ring
ring = pd.read_csv("resources/data/ring.tsv", sep='\t')
ring_yname = "target"
ring = ring.dropna()
# ring = map_target(ring, ring_yname, 1)
large_datasets.append(ExperimentDataset("ring", ring, ring_yname, fragment_size=standard_f_size))
logger.info("4 Datesets loaded")


# sylva = pd.read_csv("resources/data/sylva_prior.csv")
# sylva_yname = "label"
# sylva = sylva.dropna()
# sylva = map_target(sylva, sylva_yname, 1)
# large_datasets.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=standard_f_size))
# logger.info("5 Datesets loaded")

# # shuttle
shuttle = pd.read_csv("resources/data/shuttle.tsv", sep='\t')
shuttle_yname = "target"
shuttle = map_target(shuttle, shuttle_yname, 1)
shuttle = shuttle.dropna()
# nomapping
large_datasets.append(ExperimentDataset("shuttle", shuttle, shuttle_yname, fragment_size=standard_f_size))
logger.info("5 Datesets loaded")


# # sleep
# sleep = pd.read_csv("resources/data/sleep.tsv", sep='\t')
# sleep_yname = "target"
# sleep = sleep.dropna()
# sleep = map_target(sleep, sleep_yname, 0)
# large_datasets.append(ExperimentDataset("sleep", sleep, sleep_yname, fragment_size=standard_f_size))
# logger.info("6 Datesets loaded")


# # credit cards
cc = pd.read_excel("resources/data/credit_cards.xls", index_col=0, header=1)
cc_wo_nominals = cc.drop(columns=['SEX', 'EDUCATION', 'MARRIAGE',
                                  'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'])  # Nominals
cc_wo_nominals = cc_wo_nominals.dropna()
cc_wo_nominals_yname = "y"
cc_wo_nominals.rename(columns={'default payment next month': 'y'}, inplace=True)
# nomapping
large_datasets.append(ExperimentDataset("credit_cards", cc_wo_nominals, cc_wo_nominals_yname, fragment_size=standard_f_size))
logger.info("6 Datesets loaded")

# eeg-eye-state
eye = pd.read_csv("resources/data/eeg-eye-state.csv")
eye_yname = "Class"
eye = eye.dropna()
eye = map_target(eye, eye_yname, 2)
# nomapping
large_datasets.append(ExperimentDataset("eeg-eye-state", eye, eye_yname, fragment_size=standard_f_size))
logger.info("1 Datesets loaded")

# jm1
jm1 = pd.read_csv("resources/data/jm1.csv", na_values=['?'])
jm1_yname = "defects"
jm1 = jm1.dropna()
jm1 = jm1.drop(columns=["lOCode", "lOComment", "lOBlank", "lOCode", "locCodeAndComment"])
jm1.rename(columns={'v(g)': 'v_g',
                    'ev(g)': 'ev_g',
                    'iv(g)': 'iv_g'},
           inplace=True)
jm1 = map_target(jm1, jm1_yname, True)
large_datasets.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=standard_f_size))
logger.info("2 Datesets loaded")

bankruptcy3 = arff.loadarff("resources/data/bankruptcy/3year.arff")
bankruptcy3 = pd.DataFrame(bankruptcy3[0])
bankruptcy3 = bankruptcy3.dropna()
bankruptcy3_yname = "class"
bankruptcy3 = map_target(bankruptcy3, bankruptcy3_yname, b'1')
large_datasets.append(ExperimentDataset("bankruptcy3", bankruptcy3, bankruptcy3_yname, fragment_size=standard_f_size))
logger.info("3 Datesets loaded")

# gamma telescope
gamma_telescope_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha",
                         "fDist", "class"]
gamma_telescope = pd.read_csv("resources/data/gammatelescope/magic04.data", names=gamma_telescope_names)
gamma_telescope_yname = "class"
gamma_telescope = gamma_telescope.dropna()
bank_marketing = map_target(gamma_telescope, gamma_telescope_yname, "g")
large_datasets.append(ExperimentDataset("gamma_telescope", bank_marketing, gamma_telescope_yname, fragment_size=standard_f_size))
logger.info("6 Datesets loaded")

