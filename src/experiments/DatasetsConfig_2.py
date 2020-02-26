import logging

import pandas as pd

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

def generate_names(number):
    numbers = range(1,number + 1)
    x_names = list(map(lambda num: "X" + str(num), numbers))
    x_names.append("y")
    return x_names


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