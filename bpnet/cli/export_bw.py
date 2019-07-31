"""
Export prediction to a bigWig file
"""

from pybedtools import BedTool
import pandas as pd
import os
from bpnet.BPNet import BPNet
from bpnet.utils import read_pkl, create_tf_session
from bpnet.cli.schemas import DataSpec
from bpnet.preproc import resize_interval
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def export_bw_workflow(model_dir,
                       bed_file,
                       output_dir,
                       contrib_method='grad',
                       # pred_summary='weighted',
                       batch_size=512,
                       scale_contribution=False,
                       gpu=0):
    """
    Export model predictions to big-wig files

    Args:
      model_dir: model directory path
      output_dir: output directory path
      bed_file: file path to a bed-file containing
        the intervals of interest

    """
    # pred_summary: 'mean' or 'max', summary function name for the profile gradients
    os.makedirs(output_dir, exist_ok=True)
    if gpu is not None:
        create_tf_session(gpu)

    logger.info("Load model, preprocessor and data specs")
    bp = BPNet.from_mdir(model_dir)

    seqlen = bp.input_seqlen()
    logger.info(f"Resizing intervals (fix=center) to model's input width of: {seqlen}")
    intervals = list(BedTool(bed_file))
    intervals = [resize_interval(interval, seqlen) for interval in intervals]
    logger.info("Sort the bed file")
    intervals = list(BedTool(intervals).sort())

    bp.export_bw(intervals=intervals,
                 output_dir=output_dir,
                 # pred_summary=pred_summary,
                 contrib_method=contrib_method,
                 batch_size=batch_size,
                 scale_contribution=scale_contribution,
                 chromosomes=None)  # infer chromosomes from the fasta file
