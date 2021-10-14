from src import config
from src.classifiers import LogitClassifier
from src.cache import write_model_to_cache, read_model_from_cache


def analysis(df, name_prefix: str):
    if config.USE_CACHED_MODELS:
        if config.LOGIT:
            logit = read_model_from_cache(name_prefix + 'logit')
    else:
        if config.LOGIT:
            logit = LogitClassifier(df)

    logit.evaluate()

    if config.UPDATE_CACHE:
        if config.LOGIT:
            write_model_to_cache(logit, name_prefix + 'logit')
