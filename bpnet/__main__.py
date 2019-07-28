import argh

from bpnet.cli.train import bpnet_train

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
        ipynb_render,
    ])
    argh.dispatch(parser)
