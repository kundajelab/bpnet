import argh

from bpnet.cli.train import bpnet_train, dataspec_stats
from bpnet.cli.contrib import bpnet_contrib, list_contrib
from bpnet.cli.export_bw import bpnet_export_bw
from bpnet.cli.modisco import (bpnet_modisco_run, cwm_scan,
                               chip_nexus_analysis, modisco_export_seqlets)

# logging
import pkg_resources
import logging
import logging.config
logging.config.fileConfig(pkg_resources.resource_filename(__name__, "logging.conf"))
logger = logging.getLogger(__name__)


def ipynb_render(input_ipynb, output_ipynb, params=""):
    from bpnet.utils import render_ipynb, kwargs_str2kwargs
    render_ipynb(input_ipynb, output_ipynb, kwargs_str2kwargs(params))


def main():
    parser = argh.ArghParser()
    parser.add_commands([
        # available commands
        bpnet_train,
        dataspec_stats,
        bpnet_contrib,
        list_contrib,
        bpnet_export_bw,
        bpnet_modisco_run,
        cwm_scan,
        modisco_export_seqlets,
        chip_nexus_analysis,
        ipynb_render,
    ])
    argh.dispatch(parser)
