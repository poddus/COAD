import gzip
import logging

import pandas as pd

from src import config
from src.cache import read_df_from_cache, write_df_to_cache
from src.munge_common import extract_uuid_and_filenames_from_manifest


def munge_genome() -> pd.DataFrame:
    if config.USE_CACHED_DATA:
        return read_df_from_cache('case_to_mut_df')

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

    # remove NaNs
    log_n_hgvsc_null = len(maf_df[maf_df['HGVSc'].isnull()])
    maf_df.drop(maf_df[maf_df['HGVSc'].isnull()].index, inplace=True)
    logging.info('{} mutations without HGVSc descriptor were removed'.format(log_n_hgvsc_null))

    if config.MUT_REMOVE_KNRAS:
        k_ras_cases = maf_df.loc[maf_df['Hugo_Symbol'] == 'KRAS']['case_id']
        n_ras_cases = maf_df.loc[maf_df['Hugo_Symbol'] == 'NRAS']['case_id']
        kn_ras_cases_set = set(k_ras_cases.tolist() + n_ras_cases.tolist())
        entries_of_cases_with_kn_ras_mutations = maf_df.loc[maf_df['case_id'].isin(kn_ras_cases_set)].index
        maf_df.drop(entries_of_cases_with_kn_ras_mutations, inplace=True)
        logging.info('{} cases containing k- or n-RAS mutations were removed'.format(len(kn_ras_cases_set)))

    if config.MUT_REMOVE_SYNONYMOUS_VARIANTS:
        synonymous_variants = maf_df.loc[maf_df['Consequence'] == 'synonymous_variant'].index
        maf_df.drop(synonymous_variants, inplace=True)
        logging.info('{} synonymous variants were removed'.format(len(synonymous_variants)))

    if config.MUT_USE_UVI:
        # TODO: this introduces the problem that it inflates the possibility space and results in mutations which
        #  only occur once in the dataset.
        # create unique variant ID, store in column 'uvi'
        # note, there are some NaN in the data set
        maf_df.loc[:, 'uvi'] = pd.Series(maf_df['Hugo_Symbol'] + ':' + maf_df['HGVSc'], index=maf_df.index)
        case_to_mut_df = pd.DataFrame(False, index=maf_df['case_id'].unique(), columns=maf_df.uvi.unique())
        # this following computation is intense
        logging.debug('generating case_to_mut DataFrame (with uvi). this may take a while...')
        for case, mutations in maf_df.groupby('case_id')['uvi']:
            for mut in mutations:
                case_to_mut_df.at[case, mut] = True
    else:
        case_to_mut_df = pd.DataFrame(False, index=maf_df['case_id'].unique(), columns=maf_df.Hugo_Symbol.unique())
        logging.debug('generating case_to_mut DataFrame (without uvi). this may take a while...')
        for case, mutations in maf_df.groupby('case_id')['Hugo_Symbol']:
            for mut in mutations:
                case_to_mut_df.at[case, mut] = True

    logging.info('counting mutations that only occur once in all cases...')
    log_count_mutations = 0
    for column in case_to_mut_df.columns:
        if case_to_mut_df[column].value_counts()[1] > 1:
            log_count_mutations += 1
    logging.info('case_to_mut_df has {} of {} total mutations with more than one occurrence'.format(
        log_count_mutations, len(case_to_mut_df.columns)))

    log_n_hypermutated = 0
    if config.MUT_REMOVE_HYPERMUTATED:
        for case in case_to_mut_df.index:
            logging.debug('case: {}\nvalue_count: {}\n'.format(case, case_to_mut_df.loc[case].value_counts()[0]))
            if case_to_mut_df.loc[case].value_counts()[0] > config.MUT_HYPERMUTATION_CUTOFF:
                case_to_mut_df.drop(case, inplace=True)
                log_n_hypermutated += 1
    logging.info('{} hypermutated cases were removed'.format(log_n_hypermutated))

    if config.UPDATE_CACHE:
        write_df_to_cache(case_to_mut_df, 'case_to_mut_df')

    return case_to_mut_df