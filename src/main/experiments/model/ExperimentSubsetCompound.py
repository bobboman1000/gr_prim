import pandas as pd


class ExperimentSubsetCompound:

    def __init__(self, fragment: pd.DataFrame, complement: pd.DataFrame, fragment_y: pd.DataFrame, complement_y: pd.DataFrame, y_name="y"):
        self.fragment: pd.DataFrame = fragment
        self.fragment_y: pd.DataFrame = fragment_y
        self.complement: pd.DataFrame = complement
        self.complement_y: pd.DataFrame = complement_y
        self.y_name: str = y_name

    def get_complete_fragment(self):
        return self.combine(self.fragment, self.fragment_y)

    def get_complete_complement(self) -> pd.DataFrame:
        return self.combine(self.complement, self.complement_y)

    def combine(self, x: pd.DataFrame, y: pd.DataFrame):
        df, df_y = x.copy(True), y.copy(True)
        df.insert(loc=0, column=self.y_name, value=df_y)
        return df
