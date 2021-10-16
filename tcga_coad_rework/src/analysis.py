import logging
import pandas as pd
from matplotlib import pyplot as plt

from src import config
from src.classifiers import LogitClassifier, RFClassifier
from src.cache import write_model_to_cache, read_model_from_cache


def classify_logit(df: pd.DataFrame, name: str):
    if config.USE_CACHED_MODELS:
        logit = read_model_from_cache(name + '_logit')
    else:
        logit = LogitClassifier(df, name)

    logit.train_test_roc()

    if config.UPDATE_CACHE:
        write_model_to_cache(logit, name + '_logit')


def classify_rf(df, name: str):
    if config.USE_CACHED_MODELS:
        rforest = read_model_from_cache(name + '_rf')
    else:
        rforest = RFClassifier(df, name)

    rforest.train_test_roc()

    if config.UPDATE_CACHE:
        write_model_to_cache(rforest, name + '_rf')


def find_collinearity(df: pd.DataFrame):
    # unfortunately, the implementation of df.corr() does not include p-values. computing the p-values using method
    # scipy.stats.pearsonr is infeasible as it is pure python and as such very slow.
    # TODO: however, we may be able to calculate p-values only on combinations with high correlation
    logging.info('computing pair-wise pearsons correlation...')
    correlation = df.corr()
    logging.info('done')
    logging.info('pruning correlation matrix')
    pruned = correlation[correlation > config.CORRELATION_CUTOFF]
    pruned.replace(to_replace=1.0, value=None, inplace=True)
    pruned.dropna(axis=0, how='all', inplace=True)

    for col in pruned.columns:
        print(col, pruned.index[pruned[col] >= config.CORRELATION_CUTOFF].tolist())

    plt.matshow(pruned)
