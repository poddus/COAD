import logging
import pandas as pd
import json
import joblib


def write_df_to_cache(df, name: str) -> None:
    path = '../cache/' + 'dataframes/' + name + '.csv.gz'
    logging.info('writing {} to file {}'.format(name, path))
    df.to_csv(path, compression='gzip')


def read_df_from_cache(name: str) -> pd.DataFrame:
    path = '../cache/' + 'dataframes/' + name + '.csv.gz'
    logging.info('reading {} from file {}'.format(name, path))
    obj = pd.read_csv(path, compression='gzip', index_col=0)
    return obj


def write_api_response_to_cache(response, name: str) -> None:
    path = '../cache/' + 'gdc_api/' + name + '.json'
    logging.info('writing {} to file {}'.format(name, path))
    with open(path, 'w') as f:
        json.dump(response, f)


def read_api_response_from_cache(name: str) -> dict:
    path = '../cache/' + 'gdc_api/' + name + '.json'
    logging.info('reading {} from file {}'.format(name, path))
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def write_model_to_cache(clf, name: str) -> None:
    path = '../cache/' + 'models/' + name + '.joblib'
    logging.info('writing {} to file {}'.format(name, path))
    joblib.dump(clf, path)


def read_model_from_cache(name: str):
    path = '../cache/' + 'models/' + name + '.joblib'
    logging.info('reading {} from file {}'.format(name, path))
    obj = joblib.load(path)
    return obj
