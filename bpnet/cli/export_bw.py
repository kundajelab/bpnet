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
@arg('output_prefix',
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
                    output_prefix,
                    fasta_file=None,
                    regions=None,
                    contrib_method='grad',
                    contrib_wildcard='*/profile/wn,*/counts/pre-act',  # specifies which contrib. scores to compute
                    batch_size=256,
                    scale_contribution=False,
                    gpu=0,
                    memfrac_gpu=0.45):
    """Export model predictions and contribution scores to big-wig files
    """
    from pybedtools import BedTool
    from bpnet.modisco.core import Seqlet
    output_dir = os.path.dirname(output_prefix)
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
                 output_prefix=output_prefix,
                 contrib_method=contrib_method,
                 fasta_file=fasta_file,
                 # TODO - rename pred_summaries
                 pred_summaries=contrib_wildcard.replace("*/", "").split(","),
                 batch_size=batch_size,
                 scale_contribution=scale_contribution,
                 chromosomes=None)  # infer chromosomes from the fasta file


def contrib2bw(contrib_file,
               output_prefix):
    """Convert the contribution file to bigwigs
    """
    from kipoi.writers import BigWigWriter
    from bpnet.cli.contrib import ContribFile
    from bpnet.cli.modisco import get_nonredundant_example_idx
    output_dir = os.path.dirname(output_prefix)
    add_file_logging(output_dir, logger, 'contrib2bw')
    os.makedirs(output_dir, exist_ok=True)

    cf = ContribFile(contrib_file)

    # remove overlapping intervals
    ranges = cf.get_ranges()
    keep_idx = get_nonredundant_example_idx(ranges, width=None)
    cf.include_samples = keep_idx
    discarded = len(ranges) - len(keep_idx)
    logger.info(f"{discarded}/{len(ranges)} of ranges will be discarded due to overlapping intervals")

    contrib_scores = cf.available_contrib_scores()  # TODO - implement contrib_wildcard to filter them
    chrom_sizes = [(k, v) for k, v in cf.get_chrom_sizes().items()]
    ranges = cf.ranges()

    assert len(ranges) == len(keep_idx)

    for contrib_score in contrib_scores:
        contrib_dict = cf.get_contrib(contrib_score=contrib_score)
        contrib_score_name = contrib_score.replace("/", "_")

        for task, contrib in contrib_dict.items():
            output_file = output_prefix + f'.contrib.{contrib_score_name}.{task}.bw'
            logger.info(f"Genrating {output_file}")
            contrib_writer = BigWigWriter(output_file,
                                          chrom_sizes=chrom_sizes,
                                          is_sorted=False)

            for idx in range(len(ranges)):
                contrib_writer.region_write(region={"chr": ranges['chrom'].iloc[idx],
                                                    "start": ranges['start'].iloc[idx],
                                                    "end": ranges['end'].iloc[idx]},
                                            data=contrib[idx])
            contrib_writer.close()
    logger.info("Done!")
