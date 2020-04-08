from pysynth.ipf import IPFSynthesizer
import pandas as pd


class IPFSynthesizerInterface(IPFSynthesizer):

    def fit(self, X: pd.DataFrame) -> None:
        super().fit(dataframe=X)
