import logging.config
import logging
import pkg_resources
import argh
import os
import sys
from tqdm import tqdm
import json

from bpnet.cli.train import train, preproc
from bpnet.cli.imp_score import imp_score, avail_imp_scores, imp_score_seqmodel
from bpnet.cli.modisco import (modisco_run, modisco_plot, modisco2bed,
                               modisco_score, modisco_score2,
                               modisco_report,
                               modisco_report_all,
                               modisco_export_patterns,
                               modisco_instances_to_bed, modisco_table, modisco_centroid_seqlet_matches)
from bpnet.cli.export_bw import export_bw_workflow
from genomelake.backend import extract_fasta_to_file, extract_bigwig_to_file

import csv
csv.register_dialect('tsv', delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')

# logging
logging.config.fileConfig(pkg_resources.resource_filename(__name__, "logging.conf"))
logger = logging.getLogger(__name__)


def wc_l(fname):
    """Get the number of lines of a text-file using unix `wc -l`
    """
    import os
    import subprocess
    return int((subprocess.Popen('wc -l {0}'.format(fname), shell=True,
                                 stdout=subprocess.PIPE).stdout).readlines()[0].split()[0])


def dict2hammock_json(d):
    """Convert a dictionary to a hammock json file
    """
    return json.dumps(d)[1:-1].replace(" ", "")


def tabix_file(outfile):
    """Add a tabix to the output file
    """
    logger.info("Sorting the output file")
    os.system('sort -k1,1 -k2,2n ' + outfile + ' > ' + outfile + '.srt')
    os.system('mv ' + outfile + '.srt' + ' ' + outfile)
    logger.info("Zipping and indexing")
    os.system('bgzip -f ' + outfile)
    os.system('tabix -f -p bed ' + outfile + '.gz')
    logger.info("Done! Output file: {}".format(outfile + '.gz'))


# def bed2hammock(infile, outfile=None):
#     """
#     Convert a bed file to a hammock file
#     """
#     import os

#     assert os.path.exists(infile)
#     if outfile is None:
#         outfile = infile + ".hammock"

#     id = 1
#     fout = open(outfile, 'w')
#     show_msg = True
#     with open(outfile, 'w') as fout:
#         with open(infile) as fin:
#             for line in fin:
#                 lst = line.rstrip().split('\t')
#                 if len(lst) < 10:
#                     if show_msg:
#                         print("bed file doesn't have 10 columns. Padding with 0's")
#                         show_msg = False
#                     lst += [0] * (10 - len(lst))
#                 fout.write('{0[0]}\t{0[1]}\t{0[2]}\tscorelst:[{0[6]},{0[7]},{0[8]}],id:{1},'.format(lst, id))
#                 id += 1
#                 # if not isinstance(lst[3], str) or len(lst[3])>1:
#                 fout.write('name:"' + str(lst[3]) + '",')
#                 if lst[5] != '.':
#                     fout.write('strand:"' + str(lst[5]) + '",')
#                 if lst[9] != '-1':
#                     fout.write('sbstroke:[' + str(lst[9]) + ']')
#                 fout.write('\n')
#     tabix_file(outfile)


def parquet2tsv(input_parquet, output_tsv):
    """Convert a parquet file into a tsv
    """
    import pandas as pd
    compression = 'gzip' if output_tsv.endswith(".gz") else None
    pd.read_parquet(input_parquet, engine='fastparquet').to_csv(output_tsv,
                                                                sep='\t',
                                                                index=False,
                                                                compression=compression)


def ipynb_render(input_ipynb, output_ipynb, params=""):
    from bpnet.utils import render_ipynb, kwargs_str2kwargs
    render_ipynb(input_ipynb, output_ipynb, kwargs_str2kwargs(params))


def main():
    # assembling:
    parser = argh.ArghParser()
    parser.add_commands([ipynb_render,
                         # bed2hammock,
                         parquet2tsv,
                         # evaluate,
                         export_bw_workflow,
                         imp_score,
                         avail_imp_scores, imp_score_seqmodel,
                         modisco_report_all,
                         modisco_instances_to_bed,
                         modisco_export_patterns,
                         modisco_run, modisco_plot,
                         modisco2bed, modisco_score2,
                         modisco_report,
                         modisco_centroid_seqlet_matches,
                         extract_fasta_to_file, extract_bigwig_to_file, modisco_table,
                         modisco_score])
    argh.dispatch(parser)
