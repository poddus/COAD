import logging
import os
from pathlib import Path
from xml.etree import ElementTree
import pandas as pd

from src import config
from src.cache import read_df_from_cache, write_df_to_cache
from src.gdc_api import get_case_association, CLINICAL_PAYLOAD
from src.munge_common import extract_uuid_and_filenames_from_manifest, build_association_df


def extract_tumor_location(df: pd.DataFrame) -> pd.DataFrame:
    tumor_location = {}
    msi_status = {}  # only 85 of 388 cases (about 0.22) have reported MSI Status, so this information is not used
    for case, file in df.iterrows():
        # `file` contains uuid and path
        filepath = file[1]
        log_n_not_left_or_right = 0

        if not os.path.exists(filepath):
            logging.debug('File {}\nCase {}\nDoes not exist!\n--------------'.format(file, case))
        else:
            with open(filepath, 'r') as f:
                tree = ElementTree.parse(f)
                root = tree.getroot()

                msi_status[case] = root[1][45].text

                anatomic_neoplasm_subdivision = root[1][34].text
                if anatomic_neoplasm_subdivision in config.ANATOMIC_RIGHT:
                    tumor_location[case] = False
                elif anatomic_neoplasm_subdivision in config.ANATOMIC_LEFT:
                    tumor_location[case] = True
                elif anatomic_neoplasm_subdivision is None:
                    logging.info('no tumor location for case uuid: {}'.format(case))
                    tumor_location[case] = None
                else:
                    tumor_location[case] = None
                    log_n_not_left_or_right += 1
                    logging.debug(
                        '"{}" for case {} not in right or left group'.format(anatomic_neoplasm_subdivision, case))
        if log_n_not_left_or_right > 0:
            logging.info(
                '{} cases were not considered as they are neither `left` nor `right`'.format(log_n_not_left_or_right))

    df['tumor_left'] = pd.Series(tumor_location, dtype=bool)
    return df


def mark_annotated_cases(df: pd.DataFrame) -> pd.DataFrame:
    is_annotated = {}
    for case, file in df.iterrows():
        # `file` contains uuid and path
        filepath = file[1]
        containing_folder = Path(filepath).parent
        if os.listdir(containing_folder).count('annotations.txt') > 0:
            is_annotated[case] = True
        else:
            is_annotated[case] = False
    df['is_annotated'] = pd.Series(is_annotated, dtype=bool)
    logging.info('{} cases have clinical annotations'.format(len(is_annotated)))
    return df


def munge_clinical() -> pd.DataFrame:
    if config.USE_CACHED_DATA:
        return read_df_from_cache('clin')

    clin_c_to_f = get_case_association(CLINICAL_PAYLOAD)
    clin_uid_to_fn = extract_uuid_and_filenames_from_manifest(
        'clinical',
        '../manifest/gdc_manifest.clinical_supplement.txt')
    clin_file_df = build_association_df(clin_c_to_f, clin_uid_to_fn)

    if config.REMOVE_ANNOTATED_CASES:
        clin_file_df = mark_annotated_cases(clin_file_df)
        log_n_annotated_cases = len(clin_file_df[clin_file_df.is_annotated == True].index)
        clin_file_df.drop(clin_file_df[clin_file_df.is_annotated == True].index, inplace=True)
        logging.info('{} annotated cases removed from clinical cases.'.format(log_n_annotated_cases))
        clin_file_df.drop(['is_annotated'], axis=1, inplace=True)

    clin_df = extract_tumor_location(clin_file_df)
    clin_df.drop(['file_uuid', 'filename'], axis=1, inplace=True)

    if config.UPDATE_CACHE:
        write_df_to_cache(clin_df, 'clin')
    return clin_df
