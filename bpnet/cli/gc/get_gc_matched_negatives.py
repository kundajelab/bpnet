import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import csv
from operator import add, sub
from typing import Dict, Tuple
from pathlib import Path
import pysam
import subprocess
from bpnet.cli.gc.get_foreground_gc_content import get_foreground_gc_content

random.seed(1234)


def parse_args():
    parser = argparse.ArgumentParser(
        description="generate a bed file of non-peak regions that are gc-matched with foreground")
    parser.add_argument("-c", "--candidate_negatives",
                        help="candidate negatives bed file with gc content in 4th column rounded to 2 decimals")
    parser.add_argument("-f", "--foreground_gc_bed",
                        help="regions with their corresponding gc fractions for matching, 4th column has gc content value rounded to 2 decimals")
    parser.add_argument("-o", "--output_prefix", help="gc-matched non-peaks output file name")
    parser.add_argument("-npr", "--neg_to_pos_ratio_train", type=int, default=1,
                        help="Ratio of negatives to positives to sample for training")
    return parser.parse_args()


def make_gc_dict(candidate_negatives: str) -> Dict:
    """
    Imports the candidate negatives into a nested dictionary structure.
    The top level key is the chromosome. The nested key is the gc content fraction,
    and the `values` are a list containing the (chrom,start,end) of a region
    with the corresponding gc content fraction.

    Args:
        candidate_negatives: path to bed file with genomic regions not overlapping
        with peaks and associated GC content
    """
    # first determine number of lines for tqdm progress bar
    with open(candidate_negatives, 'r') as in_csv:
        num_lines = sum(1 for line in in_csv)

    with open(candidate_negatives, 'r') as in_csv:
        data = csv.DictReader(in_csv, delimiter='\t', fieldnames=['chr', 'start', 'end', 'gc'])
        gc_dict = {}
        for line in tqdm(data, total=num_lines):
            chrom = line['chr']
            gc = float(line['gc'])
            start = line['start']
            end = line['end']

            if chrom not in gc_dict:
                gc_dict[chrom] = {}

            if gc not in gc_dict[chrom]:
                gc_dict[chrom][gc] = [(chrom, start, end)]
            else:
                gc_dict[chrom][gc].append((chrom, start, end))

    return gc_dict


def scale_gc(cur_gc: float,
             increment: float = 0.01) -> float:
    """
    Randomly increase/decrease the gc-fraction value by increment. Also, round gc value
     to two digits.

     Args:
         cur_gc: GC content to be updated
         increment: amount by which to increase or decrease cur_gc
    """
    operation = add if random.random() > 0.5 else sub
    cur_gc = operation(cur_gc, increment)

    if cur_gc <= 0:
        cur_gc += increment
    if cur_gc >= 1:
        cur_gc -= increment

    cur_gc = round(cur_gc, 2)

    return cur_gc


def adjust_gc(chrom: str, cur_gc: float, negatives: Dict, used_negatives: Dict) -> Tuple[float, Dict]:
    """
    Checks if (1) the given gc fraction value is available
    in the negative candidates or (2) if the given gc fraction value has
    candidates not already sampled. If either of the condition fails we
    sample the neighbouring gc_fraction value by randomly scaling with 0.01.

    Args:
        chrom: chromosome location of cur_gc
        cur_gc: gc fraction in question
        negatives: all candidate negative gc fractions
        used_negatives: candidate negative gc fractions already included in matched negatives
    """
    if chrom not in used_negatives:
        used_negatives[chrom] = {}

    if cur_gc not in used_negatives[chrom]:
        used_negatives[chrom][cur_gc] = []

    while (cur_gc not in negatives[chrom]) or (len(used_negatives[chrom][cur_gc]) >= len(negatives[chrom][cur_gc])):
        cur_gc = scale_gc(cur_gc)
        if cur_gc not in used_negatives[chrom]:
            used_negatives[chrom][cur_gc] = []

    return cur_gc, used_negatives


def get_gc_matched_negatives(candidate_negatives: str,
                             neg_to_pos_ratio_train: int,
                             foreground_gc_bed: str,
                             output_prefix: str) -> None:
    """
    Main function to generate a bed file of non-peak regions that are gc-matched with foreground
    Args:
        candidate_negatives: path to bed file with non-peak genomic regions and
        gc content in 4th column rounded to 2 decimals
        neg_to_pos_ratio_train: ratio of negatives to positives to sample for training
        foreground_gc_bed: regions with their corresponding gc fractions for matching,
        4th column has gc content value rounded to 2 decimals
        output_prefix: gc-matched non-peaks output file name
    """
    negatives = make_gc_dict(candidate_negatives)

    used_negatives: Dict = {}
    negatives_bed = []
    foreground_gc_vals = []
    output_gc_vals = []
    neg_to_pos_ratio = neg_to_pos_ratio_train

    # calculate number of lines for tqdm
    with open(foreground_gc_bed, 'r') as in_csv:
        num_lines = sum(1 for line in in_csv)

    with open(foreground_gc_bed, 'r') as in_csv:
        cur_peaks = csv.DictReader(in_csv, delimiter='\t', fieldnames=['chr', 'start', 'end', 'gc'])

        for row in tqdm(cur_peaks, total=num_lines):
            chrom = row['chr']
            start = row['start']
            end = row['end']
            gc_value = float(row['gc'])
            # for every gc value in positive how many negatives to find
            # we will keep the ratio of positives to negatives in the test set same
            for rep in range(neg_to_pos_ratio):
                cur_gc, used_negatives = adjust_gc(chrom, gc_value, negatives, used_negatives)
                num_candidates = len(negatives[chrom][cur_gc])
                rand_neg_index = random.randint(0, num_candidates - 1)
                while rand_neg_index in used_negatives[chrom][cur_gc]:
                    cur_gc, used_negatives = adjust_gc(chrom, cur_gc, negatives, used_negatives)
                    num_candidates = len(negatives[chrom][cur_gc])
                    rand_neg_index = random.randint(0, num_candidates - 1)

                used_negatives[chrom][cur_gc].append(rand_neg_index)
                neg_chrom, neg_start, neg_end = negatives[chrom][cur_gc][rand_neg_index]
                negatives_bed.append([neg_chrom, int(neg_start), int(neg_end), ".", 0, ".", 0, 0, 0,
                                      (int(neg_end) - int(neg_start)) // 2])
                output_gc_vals.append(cur_gc)
                foreground_gc_vals.append(gc_value)

    negatives_bed = pd.DataFrame(negatives_bed)
    negatives_bed.to_csv(output_prefix + ".bed", sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)

    # checking how far the true distribution of foreground is compared to the backgrounds generated
    bins = np.linspace(0, 1, 100)
    plt.hist([output_gc_vals, foreground_gc_vals], bins, density=True,
             label=['negatives gc distribution', "foreground gc distribution"])
    plt.xlabel("GC content")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.savefig(output_prefix + "_compared_with_foreground.png")


if __name__ == "__main__":
    args = parse_args()
    get_gc_matched_negatives(candidate_negatives=args.candidate_negatives,
                             neg_to_pos_ratio_train=args.neg_to_pos_ratio_train,
                             foreground_gc_bed=args.foreground_gc_bed,
                             output_prefix=args.output_prefix
                             )

