import pandas as pd

from src import config
from src.analysis import analysis
from src.munge_clinical import munge_clinical
from src.munge_genome import munge_genome
from src.munge_transcriptome import munge_transcriptome


def join_clin_df(df: pd.DataFrame, clin_df: pd.DataFrame, datatype: type):
    """
    joining DataFrames of disparate shape generates NaN values, which are of dtype `float`. this causes the dtype of the
    DataFrame to change to `object`. after dropping the NaN, the dtype must be corrected for the
    classifiers to handle the data correctly
    """

    df = df.join(clin_df)
    df.dropna(inplace=True)
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
        analysis(mut_df, 'mut')

    if config.TRANSCRIPTOME:
        trans_df = munge_transcriptome()
        trans_df = join_clin_df(trans_df, clin_df, float)
        analysis(trans_df, 'trans')

    pass


main()
