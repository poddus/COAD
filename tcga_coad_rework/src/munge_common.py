import csv
from typing import Dict

import pandas as pd


def extract_uuid_and_filenames_from_manifest(data_folder: str, manifest_file_location: str) -> Dict:
    uuid_filename = {}

    with open(manifest_file_location) as f:
        data = csv.reader(f, delimiter='\t')
        next(data, None)  # skip header row of manifest file
        for row in data:
            uuid_filename[row[0]] = '../gdc_data/' + data_folder + '/{}/{}'.format(row[0], row[1])

    return uuid_filename


def build_association_df(c_to_f: Dict, uuid_to_filename: Dict) -> pd.DataFrame:
    c_to_f_df = pd.DataFrame.from_dict(c_to_f, orient='index', columns=['file_uuid'])
    uuid_to_filename_df = pd.DataFrame.from_dict(uuid_to_filename, orient='index', columns=['filename'])

    case_file_filename_df = c_to_f_df.join(uuid_to_filename_df, on='file_uuid')
    case_file_filename_df.dropna(subset=['filename'], inplace=True)
    return case_file_filename_df


