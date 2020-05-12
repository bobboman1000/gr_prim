from src.test.TestUtil import get_random_df
from src.main.generators.RandomSamples import NoiseGenerator
import numpy as np
import pandas as pd

test_data: pd.DataFrame = get_random_df((500, 100))

def test_noise_limits():
    """
    Basic test to see if everything runs thorugh
    :return:
    """
    gen = NoiseGenerator()
    gen.fit(test_data)
