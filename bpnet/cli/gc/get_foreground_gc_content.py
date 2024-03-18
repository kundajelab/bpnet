import pysam
from tqdm import tqdm
import pandas as pd


def get_foreground_gc_content(ref_fasta: str,
                              peaks_bed: str,
                              out_file: str,
                              flank_size: int) -> None:
    """
    Calculate gc content for foreground bed file (contains peaks)
    Args:
        ref_fasta: path to reference fasta file
        peaks_bed: path to peaks bed file
        flank_size: number of bases on either side of peak summit by which to define a peak
        out_file: where to store output
    """
    ref = pysam.FastaFile(ref_fasta)
    data = pd.read_csv(peaks_bed, header=None, sep='\t')

    with open(out_file, 'w') as outf:
        for index, row in tqdm(data.iterrows()):
            chrom = row[0]
            peak_start = row[1]
            summit = peak_start + row[9]

            # calculate new start and end based on fixed distnce from peak summit
            start = summit - flank_size
            end = summit + flank_size

            # calculate gc content when centered at summit
            seq = ref.fetch(chrom, start, end).upper()
            g = seq.count('G')
            c = seq.count('C')
            gc = g + c
            gc_fract = round(gc / len(seq), 2)
            outf.write(chrom + '\t' + str(start) + '\t' + str(end) + '\t' + str(gc_fract) + "\n")
