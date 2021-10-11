from typing import Any
import pandas as pd


def store_df_in_cache(df, name: str) -> None:
    path_no_ext = '../cache/' + 'dataframes/' + name
    df.to_csv(path_no_ext + '.csv')


def retrieve_df_from_cache(name: str) -> Any:
    path_no_ext = '../cache/' + 'dataframes/' + name
    obj = pd.read_csv(path_no_ext + '.csv')
    return obj
