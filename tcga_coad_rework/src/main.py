from src import config
from src.munge import munge_clinical, munge_genome, munge_transcriptome


def main():
    # clin_df = munge_clinical()
    mut_df = munge_genome()
    # trans_df = munge_transcriptome()
    pass


main()
