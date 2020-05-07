import logging

import pandas as pd
from scipy.io import arff
from src.main.experiments.config.DataUtils import map_target, generate_names

from src.main.experiments.model.ExperimentDataset import ExperimentDataset

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

# Seizures
#seizure: pd.DataFrame = pd.read_csv('resources/data/seizure.csv', header=0, index_col=0)
#seizure_yname = "y"
#seizure = map_target(seizure, seizure_yname, 1)
#large_datasets.append(ExperimentDataset("seizures_mapped_50", seizure, seizure_yname, fragment_size=standard_f_size))
#logger.info("1 Dateset loaded")

#
# # Sensorless
# sensorless = pd.read_csv("resources/data/Sensorless_drive_diagnosis.txt", sep=" ", header=-1, names=generate_names(47))
# sensorless_yname = "y"
# sensorless = sensorless.dropna()
# sensorless = map_target(sensorless, sensorless_yname, 1)
# sensorless = ExperimentDataset("sensorless", sensorless, sensorless_yname, fragment_size=standard_f_size)
# large_datasets.append(sensorless)
# logger.info("1 Datesets loaded")


# sylva
sylva = pd.read_csv("resources/data/sylva_prior.csv", header=0)
sylva_yname = "label"
sylva = sylva.dropna()
sylva = sylva.drop(columns=["Rawah_Wilderness_Area","Neota_Wilderness_Area","Comanche_Peak_Wilderness_Area","Cache_la_Poudre_Wilderness_Area",
                            "Soil_Type_1","Soil_Type_2","Soil_Type_3","Soil_Type_4","Soil_Type_5","Soil_Type_6","Soil_Type_7","Soil_Type_8",
                            "Soil_Type_9","Soil_Type_10","Soil_Type_11","Soil_Type_12","Soil_Type_13","Soil_Type_14","Soil_Type_15","Soil_Type_16",
                            "Soil_Type_17","Soil_Type_18","Soil_Type_19","Soil_Type_20","Soil_Type_21","Soil_Type_22","Soil_Type_23","Soil_Type_24",
                            "Soil_Type_25","Soil_Type_26","Soil_Type_27","Soil_Type_28","Soil_Type_29","Soil_Type_30","Soil_Type_31","Soil_Type_32",
                            "Soil_Type_33","Soil_Type_34","Soil_Type_35","Soil_Type_36","Soil_Type_37","Soil_Type_38","Soil_Type_39","Soil_Type_40",
                            "dup_Rawah_Wilderness_Area","dup_Neota_Wilderness_Area","dup_Comanche_Peak_Wilderness_Area",
                            "dup_Cache_la_Poudre_Wilderness_Area","dup_Soil_Type_1","dup_Soil_Type_2","dup_Soil_Type_3","dup_Soil_Type_4",
                            "dup_Soil_Type_5","dup_Soil_Type_6","dup_Soil_Type_7","dup_Soil_Type_8","dup_Soil_Type_9","dup_Soil_Type_10",
                            "dup_Soil_Type_11","dup_Soil_Type_12","dup_Soil_Type_13","dup_Soil_Type_14","dup_Soil_Type_15","dup_Soil_Type_16",
                            "dup_Soil_Type_17","dup_Soil_Type_18","dup_Soil_Type_19","dup_Soil_Type_20","dup_Soil_Type_21","dup_Soil_Type_22",
                            "dup_Soil_Type_23","dup_Soil_Type_24","dup_Soil_Type_25","dup_Soil_Type_26","dup_Soil_Type_27","dup_Soil_Type_28",
                            "dup_Soil_Type_29","dup_Soil_Type_30","dup_Soil_Type_31","dup_Soil_Type_32","dup_Soil_Type_33","dup_Soil_Type_34",
                            "dup_Soil_Type_35","dup_Soil_Type_36","dup_Soil_Type_37","dup_Soil_Type_38","dup_Soil_Type_39","dup_Soil_Type_40"])
sylva = map_target(sylva, sylva_yname, 1)
sylva = ExperimentDataset("sylva", sylva, sylva_yname, fragment_size=standard_f_size)
large_datasets.append(sylva)
logger.info("2 Datesets loaded")


# click
click = pd.read_csv("resources/data/click_prediction.csv")
click_yname = "click"
click = click.dropna()
# nomapping
large_datasets.append(ExperimentDataset("click", click, click_yname, fragment_size=standard_f_size))
logger.info("3 Datesets loaded")

# avila
avila = pd.read_csv("resources/data/avila/avila.txt", header=0, names=generate_names(10))
avila_yname = "y"
avila = avila.dropna()
avila = map_target(avila, avila_yname, "A")
avila = ExperimentDataset("avila", avila, avila_yname, fragment_size=standard_f_size)
large_datasets.append(avila)
logger.info("5 Datesets loaded")

SAAC2 = pd.read_csv("resources/data/SAAC2.csv", na_values=['?'])
SAAC2_yname = "class"
SAAC2 = SAAC2.dropna()
SAAC2 = map_target(SAAC2, SAAC2_yname, 2)
large_datasets.append(ExperimentDataset("SAAC2", SAAC2, SAAC2_yname, fragment_size=standard_f_size))
logger.info("6 Datesets loaded")

mozilla = pd.read_csv("resources/data/mozilla4.csv", na_values=['?'])
mozilla_yname = "state"
mozilla = mozilla.dropna()
mozilla = map_target(mozilla, mozilla_yname, 1)
large_datasets.append(ExperimentDataset("mozilla", mozilla, mozilla_yname, fragment_size=standard_f_size))
logger.info("7 Datesets loaded")

# # nomao - 7,8,15,16,24,31,32,39,40,47,48,55,56,63,64,72,80,87,88,92,96,100,104,108,112,116 nominals
# nomao = pd.read_csv("resources/data/nomao.csv")
# nomao = nomao.drop(columns=['V7','V8','V15','V16','V24','V31','V32','V39','V40','V47','V48','V55','V56','V63','V64','V72','V80','V87','V88',
#                                    'V92','V96','V100','V104','V108','V112','V116'])
# nomao_yname = "Class"
# nomao = map_target(nomao, nomao_yname, 1)
# large_datasets.append(ExperimentDataset("nomao", nomao, nomao_yname, fragment_size=standard_f_size))
# logger.info("8 Datesets loaded")

occupancy = pd.read_csv("resources/data/occupancy_data/datatest.txt", na_values=['?'])
occupancy2 = pd.read_csv("resources/data/occupancy_data/datatest2.txt", na_values=['?'])
occupancy3 = pd.read_csv("resources/data/occupancy_data/datatraining.txt", na_values=['?'])
occupancy_yname = "Occupancy"
occupancy.append(occupancy2)
occupancy.append(occupancy3)
occupancy = occupancy.drop(columns=["date", ])
large_datasets.append(ExperimentDataset("occupancy", occupancy, occupancy_yname, fragment_size=standard_f_size))
logger.info("9 Datesets loaded")

# htru
htru = pd.read_csv("resources/data/HTRU_2.csv", header=-1, names=generate_names(8))
htru_yname = "y"
htru = htru.dropna()
htru = ExperimentDataset("htru", htru, htru_yname, fragment_size=standard_f_size)
large_datasets.append(htru)
logger.info("10 Datesets loaded")

clean2 = pd.read_csv("resources/data/clean2.tsv", sep="\t")
clean2_yname = "target"
clean2 = clean2.dropna()
clean2 = clean2.drop(columns=["molecule_name", "conformation_name"])
# nomapping
large_datasets.append(ExperimentDataset("clean2", clean2, clean2_yname, fragment_size=standard_f_size))
logger.info("2 Datesets loaded")

gas = pd.read_csv("resources/data/gas_sensors.csv")
gas_yname = "y"
gas = gas.dropna()
gas = map_target(gas, gas_yname, 1)
large_datasets.append(ExperimentDataset("gas", gas, gas_yname, fragment_size=standard_f_size))
logger.info("3 Datesets loaded")

# Seizures
seizure: pd.DataFrame = pd.read_csv('resources/data/seizure.csv', header=0, index_col=0)
seizure_yname = "y"
seizure = map_target(seizure, seizure_yname, 1)
large_datasets.append(ExperimentDataset("seizures_mapped_50", seizure, seizure_yname, fragment_size=standard_f_size))
logger.info("1 Dateset loaded")