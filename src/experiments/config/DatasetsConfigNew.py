import logging
import pandas as pd
from scipy.io import arff
from src.experiments.config.DataUtils import map_target, clean, generate_names

from src.experiments.model.ExperimentDataset import ExperimentDataset


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

'''
# # mozilla 4. do not use: no description
mozilla = pd.read_csv("resources/data/cleaned/mozilla.csv", na_values=['?'])
mozilla_yname = "state"
mozilla = mozilla.dropna()
mozilla = map_target(mozilla, mozilla_yname, 1)
mozillas = []
mozillas.append(ExperimentDataset("mozilla", mozilla, mozilla_yname, fragment_size=200))
mozillas.append(ExperimentDataset("mozilla", mozilla, mozilla_yname, fragment_size=400))
mozillas.append(ExperimentDataset("mozilla", mozilla, mozilla_yname, fragment_size=800))
mozillas.append(ExperimentDataset("mozilla", mozilla, mozilla_yname, fragment_size=1600))
large_datasets.append(mozillas)
logger.info("mozilla is loaded")
'''

# # occupancy 6 attributes. the data column is changed to the start of the day
occupancy = pd.read_csv("resources/data/cleaned/occupancy.csv", na_values=['?'])
occupancy_yname = "Occupancy"
occupancy = occupancy.dropna()
occupancy = map_target(occupancy, occupancy_yname, 1)
occupancies = []
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=200))
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=400))
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=800))
occupancies.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=1600))
large_datasets.append(occupancies)
logger.info("occupancy is loaded")


# # shuttle 9 attributes.
shuttle = pd.read_csv("resources/data/shuttle.tsv", sep='\t')
shuttle_yname = "target"
shuttle = map_target(shuttle, shuttle_yname, 1)
shuttle = shuttle.dropna()
shuttles = []
shuttles.append(ExperimentDataset("shuttle", shuttle, shuttle_yname, fragment_size=200))
shuttles.append(ExperimentDataset("shuttle", shuttle, shuttle_yname, fragment_size=400))
shuttles.append(ExperimentDataset("shuttle", shuttle, shuttle_yname, fragment_size=800))
shuttles.append(ExperimentDataset("shuttle", shuttle, shuttle_yname, fragment_size=1600))
large_datasets.append(shuttles)
logger.info("shuttle is loaded")


# # htru 8 attributes
htru = pd.read_csv("resources/data/HTRU_2.csv", header=-1, names=generate_names(8))
htru_yname = "y"
htru = htru.dropna()
htrus = []
htrus.append(ExperimentDataset("htru", htru, htru_yname, fragment_size=200))
htrus.append(ExperimentDataset("htru", htru, htru_yname, fragment_size=400))
htrus.append(ExperimentDataset("htru", htru, htru_yname, fragment_size=800))
htrus.append(ExperimentDataset("htru", htru, htru_yname, fragment_size=1600))
large_datasets.append(htrus)
logger.info("htru is loaded")

'''
# # click 9. do not use: time series?
click = pd.read_csv("resources/data/cleaned/click.csv", na_values=['?'])
click_yname = "click"
click = click.dropna()
# nomapping
clicks = []
clicks.append(ExperimentDataset("click", click, click_yname, fragment_size=200))
clicks.append(ExperimentDataset("click", click, click_yname, fragment_size=400))
clicks.append(ExperimentDataset("click", click, click_yname, fragment_size=800))
clicks.append(ExperimentDataset("click", click, click_yname, fragment_size=1600))
large_datasets.append(clicks)
logger.info("click is loaded")
'''

# # avila 10 attributes
avila = pd.read_csv("resources/data/avila/avila.txt", header=0, names=generate_names(10))
avila_yname = "y"
avila = avila.dropna()
avila = map_target(avila, avila_yname, "A")
avilas = []
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=200))
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=400))
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=800))
avilas.append(ExperimentDataset("avila", avila, avila_yname, fragment_size=1600))
large_datasets.append(avilas)
logger.info("avila is loaded")


# # gamma telescope 10 attributes.
gamma_telescope_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha",
                         "fDist", "class"]
gamma_telescope = pd.read_csv("resources/data/gammatelescope/magic04.data", names=gamma_telescope_names)
gamma_telescope_yname = "class"
gamma_telescope = gamma_telescope.dropna()
bank_marketing = map_target(gamma_telescope, gamma_telescope_yname, "g")
gammas = []
gammas.append(ExperimentDataset("gamma-telescope", bank_marketing, gamma_telescope_yname, fragment_size=200))
gammas.append(ExperimentDataset("gamma-telescope", bank_marketing, gamma_telescope_yname, fragment_size=400))
gammas.append(ExperimentDataset("gamma-telescope", bank_marketing, gamma_telescope_yname, fragment_size=800))
gammas.append(ExperimentDataset("gamma-telescope", bank_marketing, gamma_telescope_yname, fragment_size=1600))
large_datasets.append(gammas)
logger.info("gamma_telescope is loaded")


# # eeg-eye-state 14 attributes
eye = pd.read_csv("resources/data/eeg-eye-state.csv")
eye_yname = "Class"
eye = eye.dropna()
eye = map_target(eye, eye_yname, 1)
eyes = []
eyes.append(ExperimentDataset("eeg-eye-state", eye, eye_yname, fragment_size=200))
eyes.append(ExperimentDataset("eeg-eye-state", eye, eye_yname, fragment_size=400))
eyes.append(ExperimentDataset("eeg-eye-state", eye, eye_yname, fragment_size=800))
eyes.append(ExperimentDataset("eeg-eye-state", eye, eye_yname, fragment_size=1600))
large_datasets.append(eyes)
logger.info("eeg-eye-state is loaded")


# # credit cards 14 attributes
cc = pd.read_csv("resources/data/cleaned/credit_cards.csv", na_values=['?'])
cc = cc.dropna()
cc_yname = "default"
cc = map_target(cc, cc_yname, 1)
ccs = []
ccs.append(ExperimentDataset("credit-cards", cc, cc_yname, fragment_size=200))
ccs.append(ExperimentDataset("credit-cards", cc, cc_yname, fragment_size=400))
ccs.append(ExperimentDataset("credit-cards", cc, cc_yname, fragment_size=800))
ccs.append(ExperimentDataset("credit-cards", cc, cc_yname, fragment_size=1600))
large_datasets.append(ccs)
logger.info("credit cards is loaded")


# # jm1 21 attribute
jm1 = pd.read_csv("resources/data/jm1.csv", na_values=['?'])
jm1_yname = "defects"
jm1 = jm1.dropna()
jm1.rename(columns={'v(g)': 'v_g',
                    'ev(g)': 'ev_g',
                    'iv(g)': 'iv_g'},
           inplace=True)
jm1 = map_target(jm1, jm1_yname, True)
jm1s = []
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=200))
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=400))
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=800))
jm1s.append(ExperimentDataset("jm1", jm1, jm1_yname, fragment_size=1600))
large_datasets.append(jm1s)
logger.info("jm1 is loaded")


# # ring 20 attributes
ring = pd.read_csv("resources/data/ring.tsv", sep='\t')
ring_yname = "target"
ring = ring.dropna()
# ring = map_target(ring, ring_yname, 1)
rings = []
rings.append(ExperimentDataset("ring", ring, ring_yname, fragment_size=200))
rings.append(ExperimentDataset("ring", ring, ring_yname, fragment_size=400))
rings.append(ExperimentDataset("ring", ring, ring_yname, fragment_size=800))
rings.append(ExperimentDataset("ring", ring, ring_yname, fragment_size=1600))
large_datasets.append(rings)
logger.info("ring is loaded")


# # sylva 20 attributes
sylva = pd.read_csv("resources/data/cleaned/sylva.csv", header=0)
sylva_yname = "label"
sylva = sylva.dropna()
sylva = map_target(sylva, sylva_yname, 1)
sylvas = []
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=200))
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=400))
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=800))
sylvas.append(ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=1600))
large_datasets.append(sylvas)
logger.info("sylva is loaded")

'''
# # SAAC2 21
SAAC2 = pd.read_csv("resources/data/cleaned/SAAC2.csv", na_values=['?'])
SAAC2_yname = "class"
SAAC2 = SAAC2.dropna()
SAAC2 = map_target(SAAC2, SAAC2_yname, 2)
saacs = []
saacs.append(ExperimentDataset("SAAC2", SAAC2, SAAC2_yname, fragment_size=200))
saacs.append(ExperimentDataset("SAAC2", SAAC2, SAAC2_yname, fragment_size=400))
saacs.append(ExperimentDataset("SAAC2", SAAC2, SAAC2_yname, fragment_size=800))
saacs.append(ExperimentDataset("SAAC2", SAAC2, SAAC2_yname, fragment_size=1600))
large_datasets.append(saacs)
logger.info("SAAC2 is loaded")
'''

# # numerai 22
numerai = pd.read_csv("resources/data/numerai.csv")
numerai_yname = "attribute_21"
numerai = numerai.dropna()
numerai = map_target(numerai, numerai_yname, 1)
numerais = []
numerais.append(ExperimentDataset("numerai", numerai, numerai_yname, fragment_size=200))
numerais.append(ExperimentDataset("numerai", numerai, numerai_yname, fragment_size=400))
numerais.append(ExperimentDataset("numerai", numerai, numerai_yname, fragment_size=800))
numerais.append(ExperimentDataset("numerai", numerai, numerai_yname, fragment_size=1600))
large_datasets.append(numerais)
logger.info("numerai is loaded")


# # higgs 7 attributes
higgs = pd.read_csv("resources/data/cleaned/higgs.csv", low_memory=False, na_values=['?'])
higgs_yname = "class"
higgs = higgs.dropna()
higgs = map_target(higgs, higgs_yname, 1)
higgss = []
higgss.append(ExperimentDataset("higgs", higgs, higgs_yname, fragment_size=200))
higgss.append(ExperimentDataset("higgs", higgs, higgs_yname, fragment_size=400))
higgss.append(ExperimentDataset("higgs", higgs, higgs_yname, fragment_size=800))
higgss.append(ExperimentDataset("higgs", higgs, higgs_yname, fragment_size=1600))
large_datasets.append(higgss)
logger.info("higgs is loaded")


# # higgs 17 attributes
higgso = pd.read_csv("resources/data/cleaned/higgs_o.csv", low_memory=False, na_values=['?'])
higgso_yname = "class"
higgso = higgso.dropna()
higgso = map_target(higgso, higgso_yname, 1)
higgsso = []
higgsso.append(ExperimentDataset("higgso", higgso, higgso_yname, fragment_size=200))
higgsso.append(ExperimentDataset("higgso", higgso, higgso_yname, fragment_size=400))
higgsso.append(ExperimentDataset("higgso", higgso, higgso_yname, fragment_size=800))
higgsso.append(ExperimentDataset("higgso", higgso, higgso_yname, fragment_size=1600))
large_datasets.append(higgsso)
logger.info("higgs_o is loaded")


# # Sensorless 48 attributes
sensorless = pd.read_csv("resources/data/cleaned/sensorless.csv", na_values=['?'])
sensorless_yname = "V49"
sensorless = sensorless.dropna()
sensorless = map_target(sensorless, sensorless_yname, 1)
sensorlesses = []
sensorlesses.append(ExperimentDataset("sensorless", sensorless, sensorless_yname, fragment_size=200))
sensorlesses.append(ExperimentDataset("sensorless", sensorless, sensorless_yname, fragment_size=400))
sensorlesses.append(ExperimentDataset("sensorless", sensorless, sensorless_yname, fragment_size=800))
sensorlesses.append(ExperimentDataset("sensorless", sensorless, sensorless_yname, fragment_size=1600))
large_datasets.append(sensorlesses)
logger.info("sensorless is loaded")


# # bankruptcy 64 attributes
bankruptcy3 = arff.loadarff("resources/data/bankruptcy/3year.arff")
bankruptcy3 = pd.DataFrame(bankruptcy3[0])
bankruptcy3 = bankruptcy3.dropna()
bankruptcy3_yname = "class"
bankruptcy3 = map_target(bankruptcy3, bankruptcy3_yname, b'1')
bankruptcies = []
bankruptcies.append(ExperimentDataset("bankruptcy3", bankruptcy3, bankruptcy3_yname, fragment_size=200))
bankruptcies.append(ExperimentDataset("bankruptcy3", bankruptcy3, bankruptcy3_yname, fragment_size=400))
bankruptcies.append(ExperimentDataset("bankruptcy3", bankruptcy3, bankruptcy3_yname, fragment_size=800))
bankruptcies.append(ExperimentDataset("bankruptcy3", bankruptcy3, bankruptcy3_yname, fragment_size=1600))
large_datasets.append(bankruptcies)
logger.info("bankruptcy is loaded")


# ==================================


# # gas 128
gas = pd.read_csv("resources/data/gas_sensors.csv")
gas_yname = "y"
gas = gas.dropna()
gas = map_target(gas, gas_yname, 1)
gases = []
gases.append(ExperimentDataset("gas", gas, gas_yname, fragment_size=200))
gases.append(ExperimentDataset("gas", gas, gas_yname, fragment_size=400))
gases.append(ExperimentDataset("gas", gas, gas_yname, fragment_size=800))
gases.append(ExperimentDataset("gas", gas, gas_yname, fragment_size=1600))
large_datasets.append(gases)
logger.info("gas is loaded")


# # clean 2 168
clean2 = pd.read_csv("resources/data/clean2.tsv", sep="\t")
clean2_yname = "target"
clean2 = clean2.dropna()
#clean2 = clean2.drop(columns=["molecule_name", "conformation_name"])
# nomapping
cleans = []
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=200))
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=400))
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=800))
cleans.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=1600))
large_datasets.append(cleans)
logger.info("clean2 is loaded")


# # seizure 178
seizure: pd.DataFrame = pd.read_csv('resources/data/seizure.csv', header=0, index_col=0)
seizure_yname = "y"
seizure = map_target(seizure, seizure_yname, 1)
seizrues = []
seizrues.append(ExperimentDataset("seizures-mapped", seizure, seizure_yname, fragment_size=200))
seizrues.append(ExperimentDataset("seizures-mapped", seizure, seizure_yname, fragment_size=400))
seizrues.append(ExperimentDataset("seizures-mapped", seizure, seizure_yname, fragment_size=800))
seizrues.append(ExperimentDataset("seizures-mapped", seizure, seizure_yname, fragment_size=1600))
large_datasets.append(seizrues)
logger.info("seizure is loaded")


