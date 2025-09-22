from bpnet.cli.argparsers import background_gc_argparser
from bpnet.cli.gc.get_foreground_gc_content import get_foreground_gc_content
from bpnet.cli.gc.get_gc_matched_negatives import get_gc_matched_negatives
from pathlib import Path
import subprocess


def get_candidate_negatives(peaks_gc_bed: str,
                            reference_gc_bed: str,
                            out_file: str,
                            ) -> None:
    """
    Find candidate negative regions in genome, or regions that do not overlap with peaks,
    using bedtools intersect
    Args:
        peaks_gc_bed: path to bed file with peaks locations
        reference_gc_bed: path to bed file with average gc content for bins across genome
        out_file: path to save out
    """
    command = f"bedtools intersect -v -a {reference_gc_bed} -b {peaks_gc_bed} > {out_file}"
    subprocess.run(command, shell=True)
    return None


def get_gc_background_main() -> None:
    """
    Wrapper around scripts to calculate gc background bins. These will be included in the
    background track to improve training accuracy and reduce model bias.

    Steps include:
    1. calculate gc content fraction in bins with peaks with `get_foreground_gc_content`
    2. identify genomic bins which do not overlap with peaks with `get_candidate_negatives`
    3. from the candidate negatives (i.e. background bins), select an appropriate number
    of bins with gc content matching that of the foreground with `get_gc_matched_negatives`

    """
    parser = background_gc_argparser()
    args = parser.parse_args()

    # all output files will be saved in this directory
    out_dir: Path = Path(args.out_dir)

    # additional arguments; see bpnet.cli.argparsers for default values
    ref_fasta: str = args.ref_fasta
    ref_gc_bed: str = args.ref_gc_bed
    peaks_bed: str = args.peaks_bed
    foreground_gc_out: str = str(out_dir / args.foreground_gc_bed)
    candidate_negatives: str = str(out_dir / args.candidate_negatives)
    flank_size: int = args.flank_size
    neg_to_pos_ratio: int = args.neg_to_pos_ratio_train
    gc_matched_negatives_out: str = str(out_dir / args.output_prefix)

    print("Calculating foreground GC content...")
    get_foreground_gc_content(ref_fasta=ref_fasta,
                              peaks_bed=peaks_bed,
                              out_file=foreground_gc_out,
                              flank_size=flank_size)

    print("Generating candidate negatives...")
    #TODO: should this be peaks_gc.bed rather than peaks_bed?
    get_candidate_negatives(peaks_gc_bed=peaks_bed,
                            reference_gc_bed=ref_gc_bed,
                            out_file=candidate_negatives)

    print("Generating GC matched negatives...")
    get_gc_matched_negatives(candidate_negatives=candidate_negatives,
                             neg_to_pos_ratio_train=neg_to_pos_ratio,
                             foreground_gc_bed=foreground_gc_out,
                             output_prefix=gc_matched_negatives_out)


if __name__ == '__main__':
    get_gc_background_main()

