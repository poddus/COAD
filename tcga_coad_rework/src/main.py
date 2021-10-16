import logging
import pandas as pd

from src import config
from src.analysis import classify_logit, classify_rf
from src.munge_clinical import munge_clinical
from src.munge_genome import munge_genome
from src.munge_transcriptome import munge_transcriptome


def find_monotone_mutations(df):
    monotone_mutations = []
    for col in df:
        vals = df[col].value_counts().tolist()
        if len(vals) <= 1:
            monotone_mutations.append(col)
    logging.debug('following mutations are monotone (all true/false): {}'.format(monotone_mutations))
    return monotone_mutations


def join_clin_df(df: pd.DataFrame, clin_df: pd.DataFrame, datatype: type):
    """
    joining DataFrames of disparate shape generates NaN values, which are of dtype `float`. this causes the dtype of the
    DataFrame to change to `object`. after dropping the NaN, the dtype must be corrected for the
    classifiers to handle the data correctly
    """

    df = df.join(clin_df, how='inner')
    # inner join causes some mutations with very few cases to lose exactly these
    # this results in monotonously False columns, which carry no information an must be pruned
    monotone_muts = find_monotone_mutations(df)
    df.drop(monotone_muts, axis=1, inplace=True)

    # any change in dtype during joining is corrected here
    # TODO: upon changing join to 'inner', this may no longer be necessary
    if datatype is bool:
        df = df.astype(bool)
    else:
        df['tumor_left'] = df['tumor_left'].astype(bool)
        df.loc[:, df.columns != 'tumor_left'] = df.loc[:, df.columns != 'tumor_left'].astype(datatype)

    return df


def main():
    clin_df = munge_clinical()

    if config.GENOME:
        mut_df = munge_genome()
        mut_df = join_clin_df(mut_df, clin_df, bool)

        # find_collinearity(mut_df)
        if config.LOGIT:
            classify_logit(mut_df, 'mutation')
        if config.GENOME:
            classify_rf(mut_df, 'mutation')

    if config.TRANSCRIPTOME:
        trans_df = munge_transcriptome()
        trans_df = join_clin_df(trans_df, clin_df, float)

        if config.LOGIT:
            classify_logit(trans_df, 'transcriptome')
        if config.GENOME:
            classify_rf(trans_df, 'transcriptome')

    pass


main()
