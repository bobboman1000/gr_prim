from typing import Tuple, Union

import numpy as np
import pandas as pd
import src.main.experiments.config.DataUtils as du


def get_random_df(shape: Tuple[int, int], lower: Union[list, np.ndarray] = None, upper: Union[list, np.ndarray] = None):
    arr = np.random.random(shape)
    col_names = du.generate_names(shape[1], add_y=False)
    return pd.DataFrame(data=arr, columns=col_names)