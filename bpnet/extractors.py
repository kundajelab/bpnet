import numpy as np
from tqdm import tqdm
import attr
from concise.preprocessing import encodeDNA
from joblib import Parallel, delayed
from collections import OrderedDict


# utility functions
def _chrom_names(fasta_file):
    """Get the list of chromosome names from a fasta file
    """
    from pysam import FastaFile
    with FastaFile(fasta_file) as fa:
        chroms = list(fa.references)
    return chroms


def _chrom_sizes(fasta_file):
    """Get the chromosome sizes for a fasta file
    """
    from pysam import FastaFile
    fa = FastaFile(fasta_file)
    chrom_lens = {name: l for name, l in zip(fa.references, fa.lengths)}
    if len(chrom_lens) == 0:
        raise ValueError(f"no chromosomes found in fasta file: {fasta_file}. "
                         "Make sure the file path is correct and that the fasta index "
                         "file {fasta_file}.fai is up to date")
    fa.close()
    return chrom_lens


@attr.s
class Variant:
    chr = attr.ib(type=str)
    pos = attr.ib(converter=int, type=int)
    ref = attr.ib(type=str)
    alt = attr.ib(type=str)


# TODO - put this class to kipoiseq
@attr.s
class Interval:
    chrom = attr.ib(type=str)
    start = attr.ib(converter=int, type=int)
    stop = attr.ib(converter=int, type=int)
    name = attr.ib(type=str, default='')
    score = attr.ib(type=float, default=0)
    strand = attr.ib(type=str, default='.')

    @classmethod
    def from_pybedtools(cls, interval):
        return cls(chrom=interval.chrom,
                   start=interval.start,
                   stop=interval.stop,
                   strand=interval.strand,
                   )

    @property
    def end(self):
        return self.stop

    @end.setter
    def end(self, value):
        self.stop = value

    def to_pybedtools(self):
        import pybedtools
        return pybedtools.create_interval_from_list([self.chrom,
                                                     self.start,
                                                     self.stop,
                                                     self.name,
                                                     self.score,
                                                     self.strand])


def extract_seq(interval, variant, fasta_file, one_hot=False):
    """
    Note: in case the variant is an indel, the anchorpoint at the beginning is used

    Args:
      interval: pybedtools.Interval where to extract the sequence from
      variant: Variant class with attributes: chr, pos, ref, alt
      fasta_file: file path or pysam.FastaFile instance
      one_hot: if True, one-hot-encode the output sequence

    Returns:
      sequence
    """
    if isinstance(fasta_file, str):
        from pysam import FastaFile
        fasta_file = FastaFile(fasta_file)
    if variant is not None and variant.pos - 1 >= interval.start and variant.pos <= interval.stop:
        inside = True
        lendiff = len(variant.alt) - len(variant.ref)
    else:
        inside = False
        lendiff = 0
    seq = fasta_file.fetch(str(interval.chrom),
                           interval.start,
                           interval.stop - lendiff)

    if not inside:
        out = seq
    else:
        # now, mutate the sequence
        pos = variant.pos - interval.start - 1
        expect_ref = seq[pos:(pos + len(variant.ref))]
        if expect_ref != variant.ref:
            raise ValueError(f"Expected reference: {expect_ref}, observed reference: {variant.ref}")
        # Anchor at the beginning
        out = seq[:pos] + variant.alt + seq[(pos + len(variant.ref)):]
    assert len(out) == interval.stop - interval.start   # sequece length has to be correct at the end
    if one_hot:
        out = encodeDNA([out.upper()])[0]
    return out


# -------------------------------------

# TODO - add bigwigExtractor

class StrandedBigWigExtractor:
    """Big-wig file extractor

    NOTE: The extractor is not thread-save.
    If you with to use it with multiprocessing,
    create a new extractor object in each process.

    # Arguments
      bigwig_file: path to the bigwig file
    """

    def __init__(self, bigwig_file,
                 interval_transform=None,
                 use_strand=False,
                 nan_as_zero=True):
        self.nan_as_zero = nan_as_zero
        self.use_strand = use_strand
        self.bigwig_file = bigwig_file
        self.interval_transform = interval_transform
        self.batch_extractor = None

    def extract_single(self, interval):
        if self.batch_extractor is None:
            from genomelake.extractors import BigwigExtractor
            self.batch_extractor = BigwigExtractor(self.bigwig_file)

        if self.interval_transform is not None:
            interval = self.interval_transform(interval)
        arr = self.batch_extractor([interval], nan_as_zero=self.nan_as_zero)[0]
        if self.use_strand and interval.strand == '-':
            arr = arr[::-1]
        return arr

    def extract(self, intervals, progbar=False):
        return np.stack([self.extract_single(interval)
                         for interval in tqdm(intervals, disable=not progbar)])

    def close(self):
        return self.batch_extractor.close()

# class ExtractorListAdd:
#     def __init__(self, extractors):
#         self.extractors = extractors

#     def extract(self, intervals):
#         return sum([ex.extract(intervals)
#                     for ex in self.extractors])

# class ExtractorList:
#     def __init__(self, extractors):
#         self.extractors = extractors

#     def extract(self, intervals):
#         return [ex.extract(intervals)
#                 for ex in self.extractors]


def bw_extract(bw_file, intervals, interval_transform, use_strand=False):
    return StrandedBigWigExtractor(bw_file,
                                   interval_transform,
                                   use_strand).extract(intervals)
