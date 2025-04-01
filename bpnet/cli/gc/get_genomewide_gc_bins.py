import pandas as pd
import pysam
from tqdm import tqdm
from bpnet.cli.argparsers import genomewide_gc_bins_argparser


def get_genomewide_gc_bins_main() -> None:
    """
    Calculate GC content of genome-wide bins, offset from each other by # of bases specified in `stride`
    Args (via arg parser):
        ref_fasta: reference genome file
        chrom_sizes: path to tab-separated file with chrom and size for genome corresponding to ref_fasta
        stride: number of bases to shift bin start positions between bins
        inputlen: bin size
        out_prefix: where to save output
    """
    parser = genomewide_gc_bins_argparser()
    args = parser.parse_args()

    ref = pysam.FastaFile(args.ref_fasta)
    chrom_sizes = pd.read_csv(args.chrom_sizes, header=None, sep='\t')
    regions_str: str = ''
    for index, row in chrom_sizes.iterrows():
        chrom = row[0]
        print(chrom)
        chrom_size = row[1]
        for bin_start in tqdm(range(0, chrom_size, args.stride)):
            bin_end = bin_start + args.inputlen
            seq = ref.fetch(chrom, bin_start, bin_end).upper()
            g = seq.count('G')
            c = seq.count('C')
            gc = g + c
            fract = round(gc / (args.inputlen), 2)
            regions_str += '\t'.join([chrom, str(bin_start), str(bin_end), str(fract)])+"\n"

    print("Writing GC content file...")
    out_bed = args.output_prefix + ".bed"
    with open(out_bed, 'w') as outfile:
        outfile.write(regions_str)


if __name__ == "__main__":
    get_genomewide_gc_bins_main()
