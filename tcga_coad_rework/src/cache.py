import pandas as pd
import json


# TODO: these files are large. perhaps they could be compressed before saving to disc?
def write_df_to_cache(df, name: str) -> None:
    path = '../cache/' + 'dataframes/' + name + '.csv'
    df.to_csv(path)


def read_df_from_cache(name: str) -> pd.DataFrame:
    path = '../cache/' + 'dataframes/' + name + '.csv'
    obj = pd.read_csv(path)
    return obj


def write_api_response_to_cache(response, name: str) -> None:
    path = '../cache/' + 'gdc_api/' + name + '.json'
    with open(path, 'w') as f:
        json.dump(response, f)


def read_api_response_from_cache(name: str) -> dict:
    path = '../cache/' + 'gdc_api/' + name + '.json'
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj
