import csv
import gzip
import logging
from typing import Dict

import pandas as pd

from src import config
from src.cache import retrieve_df_from_cache, store_df_in_cache
from src.clinical import extract_tumor_location, mark_annotated_cases

from src.gdc_api import get_case_association, CLINICAL_PAYLOAD, TRANSCRIPTOME_PAYLOAD


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


def munge_clinical() -> pd.DataFrame:
    if config.USE_CACHED_DATA:
        clin_df = retrieve_df_from_cache('clin_df')
    else:
        clin_c_to_f = get_case_association(CLINICAL_PAYLOAD)
        clin_uid_to_fn = extract_uuid_and_filenames_from_manifest(
            'clinical',
            '../manifest/gdc_manifest.clinical_supplement.txt')
        clin_file_df = build_association_df(clin_c_to_f, clin_uid_to_fn)
        clin_file_df = mark_annotated_cases(clin_file_df)
        clin_df = extract_tumor_location(clin_file_df)

    if config.REMOVE_ANNOTATED_CASES:
        clin_df.drop(clin_df[clin_df.is_annotated == True].index, inplace=True)

    if config.UPDATE_CACHE:
        store_df_in_cache(clin_df, 'clin_df')
    return clin_df


def munge_genome() -> pd.DataFrame:
    # TODO: this is a silly way to get the maf manifest, but at least it's consistent
    mutations_uid_to_fn = extract_uuid_and_filenames_from_manifest(
        'masked_somatic_mutations',
        '../manifest/gdc_manifest.mutect_masked_somatic_mutations.txt')

    maf_file_location = list(mutations_uid_to_fn.values())[0]
    with gzip.open(maf_file_location, 'rb') as f:
        maf_df = pd.read_csv(
            f,
            sep='\t',
            usecols=[
                'Hugo_Symbol',
                # 'Reference_Allele',
                'HGVSc',
                'HGVSp',
                # 'Allele',
                # 'Gene',
                'Consequence',
                'case_id'
            ],
            header=5
        )

    # TODO: something in these next blocks does not work. in the original, the first step was the intersection of sets
    #  with the clinical df, but ideally that should only be done afterwards when they are combined
    if config.REMOVE_KNRAS:
        k_ras_cases = maf_df.loc[maf_df['Hugo_Symbol'] == 'KRAS']['case_id']
        n_ras_cases = maf_df.loc[maf_df['Hugo_Symbol'] == 'NRAS']['case_id']
        kn_ras_cases_set = set(k_ras_cases.tolist() + n_ras_cases.tolist())
        entries_of_cases_with_kn_ras_mutations = maf_df.loc[maf_df[
            'case_id'].isin(kn_ras_cases_set)].index
        maf_df.drop(entries_of_cases_with_kn_ras_mutations, inplace=True)

    if config.REMOVE_SYNONYMOUS_VARIANTS:
        synonymous_variants = maf_df.loc[
            maf_df['Consequence'] == 'synonymous_variant'].index
        maf_df.drop(synonymous_variants, inplace=True)

    # Unique Variant Identifier
    # create unique variant ID, store in column 'uvi'
    # note, there are some NaN in the data set
    maf_df.loc[:, 'uvi'] = pd.Series(maf_df['Hugo_Symbol'] + ':' + maf_df['HGVSc'], index=maf_df.index)
    case_to_mut_df = pd.DataFrame(False, index=maf_df['case_id'].unique(), columns=maf_df.uvi.unique())

    if config.REMOVE_HYPERMUTATED:
        for case in case_to_mut_df.index:
            logging.debug('value_count: {}'.format(case_to_mut_df.loc[case].value_counts()[0]))
            if case_to_mut_df.loc[case].value_counts()[0] > config.HYPERMUTATION_CUTOFF:
                case_to_mut_df.drop(case, inplace=True)

    return case_to_mut_df


def munge_transcriptome() -> pd.DataFrame:
    trans_c_to_f = get_case_association(TRANSCRIPTOME_PAYLOAD)
    trans_uid_to_fn = extract_uuid_and_filenames_from_manifest(
        'transcriptome_profiling',
        '../manifest/gdc_manifest.transcriptome_profiling.txt')
    trans_file_df = build_association_df(trans_c_to_f, trans_uid_to_fn)
    # TODO: extract transcriptome data

    return trans_file_df


