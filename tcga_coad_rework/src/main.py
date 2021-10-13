from src import config
from src.munge_clinical import munge_clinical
from src.munge_genome import munge_genome
from src.munge_transcriptome import munge_transcriptome


def main():
    clin_df = munge_clinical()

    if config.GENOME_ANALYSIS:
        mut_df = munge_genome()
        mut_df = mut_df.join(clin_df)
        mut_df.dropna(inplace=True)

    if config.TRANSCRIPTOME_ANALYSIS:
        trans_df = munge_transcriptome()
        trans_df = trans_df.join(clin_df)
        trans_df.dropna(inplace=True)

    pass


main()
