import pandas as pd


def map_target(data: pd.DataFrame, y_col_name: str, to_ones):
    data.loc[:, y_col_name] = data.loc[:, y_col_name].map(lambda e: 1 if e == to_ones else 0)
    return data


def clean(data: pd.DataFrame, value):
    for row in range(data.shape[0]):
        if data.loc[row,:].isin([value]).any():
            data = data.drop(index=row)
    return data


def generate_names(number, add_y=True):
    numbers = range(1,number + 1)
    x_names = list(map(lambda num: "X" + str(num), numbers))
    if add_y:
        x_names.append("y")
    return x_names
