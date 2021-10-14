from src import config
from src.classifiers import LogitClassifier, RFClassifier
from src.cache import write_model_to_cache, read_model_from_cache




def classify(df, name_prefix: str):
    if config.LOGIT:
        if config.USE_CACHED_MODELS:
            logit = read_model_from_cache(name_prefix + '_logit')
        else:
            logit = LogitClassifier(df)
        logit.evaluate()
        if config.UPDATE_CACHE:
            write_model_to_cache(logit, name_prefix + '_logit')

    if config.RANDOMFOREST:
        if config.USE_CACHED_MODELS:
            rforest = read_model_from_cache(name_prefix + '_logit')
        else:
            rforest = RFClassifier(df)
        rforest.evaluate()
        if config.UPDATE_CACHE:
            write_model_to_cache(rforest, name_prefix + '_logit')
