import csv
import gzip
import logging
import os
from collections import defaultdict

import pandas as pd

from src import config

from src.gdc_api import get_case_association, TRANSCRIPTOME_PAYLOAD
from src.cache import read_df_from_cache, write_df_to_cache
from src.munge_common import extract_uuid_and_filenames_from_manifest, build_association_df


def extract_gz_data():
    pass


def munge_transcriptome() -> pd.DataFrame:
    if config.USE_CACHED_DATA:
        return read_df_from_cache('trans')

    trans_c_to_f = get_case_association(TRANSCRIPTOME_PAYLOAD)
    trans_uid_to_fn = extract_uuid_and_filenames_from_manifest(
        'transcriptome_profiling',
        '../manifest/gdc_manifest.transcriptome_profiling.txt')
    trans_file_df = build_association_df(trans_c_to_f, trans_uid_to_fn)
    # TODO: extract transcriptome data

    logging.info('generating case_to_mut. this may take a while...')
    transcriptome = defaultdict(dict)
    for case, file in trans_file_df.iterrows():
        # `file` contains uuid and path
        filepath = file[1]
        if not os.path.exists(filepath):
            logging.debug('File {}\nCase {}\nDoes not exist!\n--------------'.format(file, case))
        else:
            logging.debug('case {}'.format(case))
            with gzip.open(filepath, 'rt') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    transcriptome[case][row[0]] = row[1]
    logging.debug('done')
    logging.debug('converting defaultdict to DataFrame...')
    case_to_trans_df = pd.DataFrame.from_dict(transcriptome, orient='index', dtype='float')
    logging.debug('done')

    if config.UPDATE_CACHE:
        write_df_to_cache(case_to_trans_df, 'trans')

    return case_to_trans_df
