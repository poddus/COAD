import logging

logging.basicConfig(level=logging.DEBUG)

USE_CACHED_DATA = False
UPDATE_CACHE = True

# analyses to be prepared and performed
GENOME_ANALYSIS = True
TRANSCRIPTOME_ANALYSIS = True

REMOVE_ANNOTATED_CASES = True
REMOVE_KNRAS = True
REMOVE_SYNONYMOUS_VARIANTS = True
REMOVE_HYPERMUTATED = True
HYPERMUTATION_CUTOFF = 250


ANATOMIC_RIGHT = ('Cecum',
                  'Ascending Colon',
                  'Hepatic Flexure')
                    # 'Transverse Colon'
ANATOMIC_LEFT = ('Descending Colon',
                 'Sigmoid Colon',
                 'Rectosigmoid Junction')
                    #  'Splenic Flexure'