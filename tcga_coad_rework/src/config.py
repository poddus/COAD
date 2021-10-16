import random
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

pd.set_option('display.expand_frame_repr', False)

# analyses to be prepared and performed
GENOME = True
TRANSCRIPTOME = True
LOGIT = True
RANDOMFOREST = True

USE_CACHED_API_RESPONSE = True
USE_CACHED_DATA = True
USE_CACHED_MODELS = True
UPDATE_CACHE = False

REMOVE_ANNOTATED_CASES = True
MUT_REMOVE_KNRAS = True
MUT_REMOVE_SYNONYMOUS_VARIANTS = True
MUT_REMOVE_HYPERMUTATED = False
MUT_HYPERMUTATION_CUTOFF = 250
MUT_USE_UVI = False
MUT_REMOVE_RARE_VARIANTS = True
MUT_RARE_VARIANT_CUTOFF = 1
CORRELATION_CUTOFF = 0.8

RF_HYPERPARAMETER_TUNING = False
CLASSIFIER_RANDOM_STATE = 3257
# CLASSIFIER_RANDOM_STATE = random.randint(0,4294967295)
CLASSIFIER_TEST_SIZE = 0.33

ANATOMIC_RIGHT = ('Cecum',
                  'Ascending Colon',
                  'Hepatic Flexure',
                  'Transverse Colon')
ANATOMIC_LEFT = ('Descending Colon',
                 'Sigmoid Colon',
                 'Rectosigmoid Junction',
                 'Splenic Flexure')
