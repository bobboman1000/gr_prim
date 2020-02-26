import subprocess
from io import StringIO

import pandas as pd


# Note: This is static!


class NaiveBayesEstimation:

    def __init__(self):
        subprocess.run(r"""cd generators/NBEWrapper/nbe
        make""", shell=True)

    def fit(self, data: pd.DataFrame, holdout: pd.DataFrame):
        self.write_nbe_string(self.to_nbe_string(data), "generators/NBEWrapper/nbe/", "temp_data.data")
        self.write_nbe_string(self.to_nbe_string(holdout), "generators/NBEWrapper/nbe/", "temp_data.hold")
        subprocess.run(r"""cd generators/NBEWrapper/nbe
        ./nbetrain -v temp_data.data temp_data.hold model
        """, shell=True)
        print("Model ready")
        return NBEModelWrapper(data)

    def to_nbe_string(self, data: pd.DataFrame):
        result = ""
        for row_idx in range(data.shape[0]):
            result += "("
            for col_idx in range(data.shape[1]):
                result += str(data.iloc[row_idx, col_idx])
                result += " " if col_idx < data.shape[1] - 1 else ""
            result += ")"

            if row_idx != data.shape[0] - 1:
                result += "\n"
        return result

    def write_nbe_string(self, nbe_string, path, filename):
        text_file = open(path + "/" + filename, "w")
        text_file.write(nbe_string)
        text_file.close()


class NBEModelWrapper:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def sample(self, number: int):
        print("Sample:" + str(number))

        output: str = str(subprocess.check_output(r"""cd generators/NBEWrapper/nbe
                ./nbesample model 10""" + str(number), shell=True))
        output = output.replace("b", "").replace("'","")
        return self.nbe_string_to_dataframe(output)

    def nbe_string_to_dataframe(self, string: str):
        string = StringIO(string.replace("(", "").replace(")", ";").replace("\\n", ""))
        df = pd.read_csv(string, sep=" ", low_memory=False, header=None, lineterminator=";")
        return df
