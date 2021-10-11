import logging
import os
from pathlib import Path
from xml.etree import ElementTree as et
import pandas as pd

from src import config


def extract_tumor_location(df):
    tumor_location = {}
    for case, file in df.iterrows():
        # `file` contains uuid and path
        filepath = file[1]
        if not os.path.exists(filepath):
            logging.debug('File {}\nCase {}\nDoes not exist!\n--------------'.format(file, case))
        else:
            with open(filepath, 'r') as f:
                tree = et.parse(f)
                root = tree.getroot()

                log_n_not_left_or_right = 0
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
            logging.info(
                '{} cases were not considered as they are neither `left` nor `right`'.format(log_n_not_left_or_right))

    df['tumor_left'] = pd.Series(tumor_location)
    df.drop(['file_uuid', 'filename'], axis=1, inplace=True)
    return df


def mark_annotated_cases(df):
    is_annotated = {}
    for case, file in df.iterrows():
        # `file` contains uuid and path
        filepath = file[1]
        containing_folder = Path(filepath).parent
        if os.listdir(containing_folder).count('annotations.txt') > 0:
            is_annotated[case] = True
        else:
            is_annotated[case] = False
    df['is_annotated'] = pd.Series(is_annotated)
    logging.info('{} cases have clinical annotations'.format(len(is_annotated)))
    return df
