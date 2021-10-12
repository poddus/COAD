import pandas as pd
import logging


logging.basicConfig(level=logging.DEBUG)

pd.set_option('display.expand_frame_repr', False)

# analyses to be prepared and performed
GENOME_ANALYSIS = True
TRANSCRIPTOME_ANALYSIS = True

USE_CACHED_API_RESPONSE = True
USE_CACHED_DATA = True
UPDATE_CACHE = False

MUT_REMOVE_ANNOTATED_CASES = True
MUT_REMOVE_KNRAS = True
MUT_REMOVE_SYNONYMOUS_VARIANTS = True
MUT_REMOVE_HYPERMUTATED = False
MUT_HYPERMUTATION_CUTOFF = 250
MUT_USE_UVI = False


ANATOMIC_RIGHT = ('Cecum',
                  'Ascending Colon',
                  'Hepatic Flexure')
                    # 'Transverse Colon'
ANATOMIC_LEFT = ('Descending Colon',
                 'Sigmoid Colon',
                 'Rectosigmoid Junction')
                    #  'Splenic Flexure'