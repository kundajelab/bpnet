"""
Export prediction to a bigWig file
"""
import pandas as pd
import os
from bpnet.seqmodel import SeqModel
from bpnet.BPNet import BPNetSeqModel
from bpnet.utils import add_file_logging, read_pkl, create_tf_session
from bpnet.dataspecs import DataSpec
from bpnet.preproc import resize_interval
from argh.decorators import named, arg
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@named("export-bw")
@arg('model_dir',
     help='Path to the trained model directory (specified in `bpnet train <output_dir>`')
@arg('output_dir',
     help='if True, the output directory will be overwritten')
@arg('--regions',
     help='Path to the interval bed file. If not specified, files specified in dataspec.yml will be used')
@arg('--contrib-method',
     help='Contribution score method to use. Available: grad, deeplift')
@arg("--contrib-wildcard",
     help="Wildcard of the contribution scores to compute. For example, */profile/wn computes"
     "the profile contribution scores for all the tasks (*) using the wn normalization (see bpnet.heads.py)."
     "*/counts/pre-act computes the total count contribution scores for all tasks w.r.t. the pre-activation output "
     "of prediction heads. Multiple wildcards can be by comma-separating them.")
@arg("--scale-contribution",
     help='if True, multiple the contribution scores by the predicted count value')
@arg("--batch-size",
     help='Batch size for computing the predictions and contribution scores')
@arg('--gpu',
     help='which gpu to use. Example: gpu=1')
@arg('--memfrac-gpu',
     help='what fraction of the GPU memory to use')
def bpnet_export_bw(model_dir,
                    output_dir,  # TODO - migrate to prefix
                    fasta_file=None,  # TODO - also use the fasta file
                    regions=None,
                    contrib_method='grad',
                    contrib_wildcard='*/profile/wn,*/counts/pre-act',  # specifies which contrib. scores to compute
                    batch_size=512,
                    scale_contribution=False,
                    gpu=0,
                    memfrac_gpu=0.45):
    """Export model predictions and contribution scores to big-wig files
    """
    from pybedtools import BedTool
    from bpnet.modisco.core import Seqlet
    add_file_logging(output_dir, logger, 'bpnet-export-bw')
    os.makedirs(output_dir, exist_ok=True)
    if gpu is not None:
        create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac_gpu)

    logger.info("Load model")

    bp = BPNetSeqModel.from_mdir(model_dir)

    if regions is not None:
        logger.info(f"Computing predictions and contribution scores for provided regions: {regions}")
        regions = list(BedTool(regions))
    else:
        logger.info("--regions not provided. Using regions from dataspec.yml")
        ds = DataSpec.load(os.path.join(model_dir, 'dataspec.yml'))
        regions = ds.get_all_regions()

    seqlen = bp.input_seqlen()
    logger.info(f"Resizing regions (fix=center) to model's input width of: {seqlen}")
    regions = [resize_interval(interval, seqlen) for interval in regions]
    logger.info("Sort the bed file")
    regions = list(BedTool(regions).sort())

    bp.export_bw(regions=regions,
                 output_dir=output_dir,
                 contrib_method=contrib_method,
                 # TODO - use also contrib_wildcard there
                 pred_summaries=contrib_wildcard.replace("*/", "").split(","),
                 batch_size=batch_size,
                 scale_contribution=scale_contribution,
                 chromosomes=None)  # infer chromosomes from the fasta file


def contrib2bw(contrib_file,
               output_dir,
               batch_size=512,
               scale_contribution=False):
    """Convert the contribution file to bigwigs
    """
    pass
