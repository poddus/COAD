from src import config
from src.munge_transcriptome import munge_transcriptome
from src.munge_genome import munge_genome
from src.munge_clinical import munge_clinical


def main():
    # clin_df = munge_clinical()

    # if config.GENOME_ANALYSIS:
    # mut_df = munge_genome()

    # if config.TRANSCRIPTOME_ANALYSIS:
    trans_df = munge_transcriptome()
    pass


main()
