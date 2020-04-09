import logging
import pandas as pd
from src.experiments.config.DataUtils import map_target, clean, generate_names

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


# ====


# # mozilla 4
mozilla = pd.read_csv("resources/data/cleaned/mozilla.csv", na_values=['?'])
mozilla_yname = "state"
mozilla = mozilla.dropna()
mozilla = map_target(mozilla, mozilla_yname, 1)
large_datasets.append(ExperimentDataset("mozilla", mozilla, mozilla_yname, fragment_size=standard_f_size))
logger.info("mozilla is loaded")


# # occupancy 6
occupancy = pd.read_csv("resources/data/cleaned/occupancy.csv", na_values=['?'])
occupancy_yname = "Occupancy"
occupancy = occupancy.drop(columns=["date", ])
large_datasets.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=standard_f_size))
logger.info("occupancy is loaded")


# # shuttle 6
shuttle = pd.read_csv("resources/data/cleaned/shuttle.csv", na_values=['?'])
shuttle_yname = "target"
shuttle = map_target(shuttle, shuttle_yname, 4)
shuttle = shuttle.dropna()
large_datasets.append(ExperimentDataset("shuttle", shuttle, shuttle_yname, fragment_size=standard_f_size))
logger.info("shuttle is loaded")


# # htru 9
htru = pd.read_csv("resources/data/HTRU_2.csv", header=-1, names=generate_names(8))
htru_yname = "y"
htru = htru.dropna()
htru = ExperimentDataset("htru", htru, htru_yname, fragment_size=standard_f_size)
large_datasets.append(htru)
logger.info("htru is loaded")


# # click 9
click = pd.read_csv("resources/data/cleaned/click.csv", na_values=['?'])
click_yname = "click"
click = click.dropna()
# nomapping
large_datasets.append(ExperimentDataset("click", click, click_yname, fragment_size=standard_f_size))
logger.info("click is loaded")


# # avila 10
avila = pd.read_csv("resources/data/cleaned/avila.csv", na_values=['?'])
avila_yname = "V11"
avila = avila.dropna()
avila = map_target(avila, avila_yname, "A")
avila = ExperimentDataset("avila", avila, avila_yname, fragment_size=standard_f_size)
large_datasets.append(avila)
logger.info("avila is loaded")


# # gamma telescope 11
gamma_telescope_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha",
                         "fDist", "class"]
gamma_telescope = pd.read_csv("resources/data/gammatelescope/magic04.data", names=gamma_telescope_names)
gamma_telescope_yname = "class"
gamma_telescope = gamma_telescope.dropna()
bank_marketing = map_target(gamma_telescope, gamma_telescope_yname, "g")
large_datasets.append(ExperimentDataset("gamma_telescope", bank_marketing, gamma_telescope_yname, fragment_size=standard_f_size))
logger.info("gamma_telescope is loaded")


# # eeg-eye-state 15
eye = pd.read_csv("resources/data/eeg-eye-state.csv")
eye_yname = "Class"
eye = eye.dropna()
eye = map_target(eye, eye_yname, 2)
# nomapping
large_datasets.append(ExperimentDataset("eeg-eye-state", eye, eye_yname, fragment_size=standard_f_size))
logger.info("eeg-eye-state is loaded")


# # credit cards 16
cc = pd.read_csv("resources/data/cleaned/cleaned/credit_cards.csv", na_values=['?'])
cc = cc.dropna()
cc_yname = "y"
cc.rename(columns={'default': 'y'}, inplace=True)
# nomapping
large_datasets.append(ExperimentDataset("credit_cards", cc, cc_yname, fragment_size=standard_f_size))
logger.info("credit cards is loaded")


# # jm1 17
jm1 = pd.read_csv("resources/data/cleaned/jm1.csv", na_values=['?'])
jm1_yname = "defects"
jm1 = jm1.dropna()
jm1 = map_target(jm1, jm1_yname, True)
large_datasets.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=standard_f_size))
logger.info("jm1 is loaded")


# # ring 21
ring = pd.read_csv("resources/data/ring.tsv", sep='\t')
ring_yname = "target"
ring = ring.dropna()
# ring = map_target(ring, ring_yname, 1)
large_datasets.append(ExperimentDataset("ring", ring, ring_yname, fragment_size=standard_f_size))
logger.info("ring is loaded")


# # sylva 21
sylva = pd.read_csv("resources/data/cleaned/sylva.csv", header=0)
sylva_yname = "label"
sylva = sylva.dropna()
sylva = map_target(sylva, sylva_yname, 1)
sylva = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=standard_f_size)
large_datasets.append(sylva)
logger.info("sylva is loaded")


# # SAAC2 21
SAAC2 = pd.read_csv("resources/data/cleaned/SAAC2.csv", na_values=['?'])
SAAC2_yname = "class"
SAAC2 = SAAC2.dropna()
SAAC2 = map_target(SAAC2, SAAC2_yname, 2)
large_datasets.append(ExperimentDataset("SAAC2", SAAC2, SAAC2_yname, fragment_size=standard_f_size))
logger.info("SAAC2 is loaded")


# # numerai 22
numerai = pd.read_csv("resources/data/numerai.csv")
numerai_yname = "attribute_21"
numerai = numerai.dropna()
# nomapping
large_datasets.append(ExperimentDataset("numerai", numerai, numerai_yname, fragment_size=standard_f_size))
logger.info("numerai is loaded")


# # higgs 25
higgs = pd.read_csv("resources/data/cleaned/higgs.csv", low_memory=False, na_values=['?'])
higgs_yname = "class"
higgs = higgs.dropna()
large_datasets.append(ExperimentDataset("higgs", higgs, higgs_yname, fragment_size=standard_f_size))
logger.info("higgs is loaded")


# # Sensorless 49
sensorless = pd.read_csv("resources/data/cleaned/sensorless.csv", na_values=['?'])
sensorless_yname = "V49"
sensorless = sensorless.dropna()
sensorless = map_target(sensorless, sensorless_yname, 1)
sensorless = ExperimentDataset("sensorless", sensorless, sensorless_yname, fragment_size=standard_f_size)
large_datasets.append(sensorless)
logger.info("sensorless is loaded")


# # bankruptcy 63
bankruptcy3 = pd.read_csv("resources/data/cleaned/bankruptcy.csv", na_values=['?'])
bankruptcy3 = pd.DataFrame(bankruptcy3[0])
bankruptcy3 = bankruptcy3.dropna()
bankruptcy3_yname = "class"
bankruptcy3 = map_target(bankruptcy3, bankruptcy3_yname, b'1')
large_datasets.append(ExperimentDataset("bankruptcy3", bankruptcy3, bankruptcy3_yname, fragment_size=standard_f_size))
logger.info("bankruptcy is loaded")


# ==================================


# # gas 129
gas = pd.read_csv("resources/data/gas_sensors.csv")
gas_yname = "y"
gas = gas.dropna()
gas = map_target(gas, gas_yname, 1)
large_datasets.append(ExperimentDataset("gas", gas, gas_yname, fragment_size=standard_f_size))
logger.info("gas is loaded")



# # clean 2 162
clean2 = pd.read_csv("resources/data/cleaned/clean2.csv", na_values=['?'])
clean2_yname = "target"
clean2 = clean2.dropna()
#clean2 = clean2.drop(columns=["molecule_name", "conformation_name"])
# nomapping
large_datasets.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=standard_f_size))
logger.info("clean2 is loaded")



# # seizure 179
seizure: pd.DataFrame = pd.read_csv('resources/data/seizure.csv', header=0, index_col=0)
seizure_yname = "y"
seizure = map_target(seizure, seizure_yname, 1)
large_datasets.append(ExperimentDataset("seizures_mapped_50", seizure, seizure_yname, fragment_size=standard_f_size))
logger.info("seizure is loaded")


