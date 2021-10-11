import csv
import json
import os.path
from typing import Tuple, Dict
import pandas as pd
import xml.etree.ElementTree as et

from src.api_query import get_case_association, CLINICAL_PAYLOAD, TRANSCRIPTOME_PAYLOAD
from src.config import ONLINE, UPDATE_CACHE, ANATOMIC_RIGHT, ANATOMIC_LEFT


def extract_uuid_and_filenames_from_manifest(file_location: str) -> Dict:
    uuid_filename = {}

    with open(file_location) as f:
        data = csv.reader(f, delimiter='\t')
        next(data, None)  # skip header row of manifest file
        for row in data:
            uuid_filename[row[0]] = row[1]

    return uuid_filename


def get_case_to_file() -> Tuple[Dict, Dict]:
    if not ONLINE:
        with open('../cache/associations/clinical_case_to_file.json', 'r') as f:
            clinical_case_to_file = json.load(f)
        with open('../cache/associations/transcriptome_case_to_file.json', 'r') as f:
            transcriptome_case_to_file = json.load(f)
    else:
        clinical_case_to_file = get_case_association(CLINICAL_PAYLOAD)
        transcriptome_case_to_file = get_case_association(TRANSCRIPTOME_PAYLOAD)

        if UPDATE_CACHE:
            with open('../cache/associations/clinical_case_to_file.json', 'w') as f:
                f.write(json.dumps(clinical_case_to_file))
            with open('../cache/associations/transcriptome_case_to_file.json', 'w') as f:
                f.write(json.dumps(transcriptome_case_to_file))

    return clinical_case_to_file, transcriptome_case_to_file


def get_uid_to_fns() -> Tuple[Dict, Dict, Dict]:
    mutations_uid_to_fn = extract_uuid_and_filenames_from_manifest(
        '../manifest/gdc_manifest.mutect_masked_somatic_mutations.txt')
    clinical_uid_to_fn = extract_uuid_and_filenames_from_manifest(
        '../manifest/gdc_manifest.clinical_supplement.txt')
    transcriptome_uid_to_fn = extract_uuid_and_filenames_from_manifest(
        '../manifest/gdc_manifest.transcriptome_profiling.txt')

    return mutations_uid_to_fn, clinical_uid_to_fn, transcriptome_uid_to_fn


def build_association_df(c_to_f: Dict, uuid_to_filename: Dict) -> pd.DataFrame:
    c_to_f_df = pd.DataFrame.from_dict(c_to_f, orient='index', columns=['file_uuid'])
    uuid_to_filename_df = pd.DataFrame.from_dict(uuid_to_filename, orient='index', columns=['filename'])

    case_file_filename_df = c_to_f_df.join(uuid_to_filename_df, on='file_uuid')
    case_file_filename_df.dropna(subset=['filename'], inplace=True)
    return case_file_filename_df


def extract_tumor_location(df):
    tumor_location = {}
    for case, file in df.iterrows():
        filepath = '../gdc_data/clinical/{}/{}'.format(file[0], file[1])
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                tree = et.parse(f)
                root = tree.getroot()

                if root[1][34].text is None:
                    tumor_location[case] = None
                elif root[1][34].text in ANATOMIC_RIGHT:
                    tumor_location[case] = False
                elif root[1][34].text in ANATOMIC_LEFT:
                    tumor_location[case] = True

    # TODO: discrepant lengths
    df['tumor_location'] = tumor_location


def main():
    # get associations
    clin_c_to_f, trans_c_to_f = get_case_to_file()
    mut_uid_to_fn, clin_uid_to_fn, trans_uid_to_fn = get_uid_to_fns()

    # build DataFrames for clinical and transcriptome
    clin_df = build_association_df(clin_c_to_f, clin_uid_to_fn)
    trans_df = build_association_df(trans_c_to_f, trans_uid_to_fn)

    case_loc_df = extract_tumor_location(clin_df)

main()

