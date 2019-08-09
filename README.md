# BPNet

<a href='https://circleci.com/gh/kundajelab/bpnet>
	<img alt='CircleCI' src='https://circleci.com/gh/kundajelab/bpnet.svg?style=svg&circle-token=f55c1cf580b05df76e260993f7645e35d5302e76 style="max-height:20px;width:auto">
</a>

## Getting started

Main documentation of the bpnet package and an end-to-end example higlighting the main features are contained in the following colab notebook <https://colab.research.google.com/drive/1VNsNBfugPJfJ02LBgvPwj-gPK0L_djsD>. You can run this notebook yourself by clicking on '**Open in playground**'. Individual cells of this notebook can be executed by pressing the Shift+Enter keyboard shortcut. To learn more about colab, visit <https://colab.research.google.com> and follow the 'Welcome To Colaboratory' notebook.

<img src="./docs/theme_dir/bpnet/colab-header.png" alt="BPNet" style="width: 300px;"/>

<!-- <img src="./notebooks/figs/bpnet-arch.png" alt="BPNet" style="width: 300px;"/> -->

## Main commands

### Train BPNet

Train a model using an existing architecture [bpnet9](bpnet/premade/bpnet9.gin)

```bash
bpnet train --premade=bpnet9 dataspec.yml output_dir
```

Train a model by overriding some of the hyper-parameters:

```bash
bpnet train --premade=bpnet9 dataspec.yml --override='seq_width=200;n_dil_layers=3' output_dir
```

### Compute contribution scores

```bash
bpnet contrib $model_dir --method=deeplift $model_dir/imp.scores.h5
```

### Discover motifs with TF-MoDISco

```bash
bpnet modisco-run $model_dir/imp.scores.h5 --premade=modisco-50k $modisco_dir
```

### Determine motif instances with CWM scanning

```bash
bpnet cwm-scan $modisco_dir $modisco_dir/motif-instances.tsv.gz
```

## Installation

Supported python version is 3.6. After installing anaconda ([download page](https://www.anaconda.com/download/)) or miniconda ([download page](https://conda.io/miniconda.html)), create a new bpnet environment by executing the following code:

```bash
# Clone this repository
git clone git@github.com:kundajelab/bpnet.git
cd bpnet

# create 'bpnet' conda environment
conda env create -f conda-env.yml

# Disable HDF5 file locking to prevent issues with Keras (https://github.com/h5py/h5py/issues/1082)
echo 'export HDF5_USE_FILE_LOCKING=FALSE' >> ~/.bashrc

# Activate the conda environment
source activate bpnet
```

Alternatively, you could also start a fresh conda environment by running the following

```bash
conda env create -n bpnet python=3.6
source activate bpnet
conda install -c bioconda pybedtools bedtools pybigwig pysam genomelake
pip install git+https://github.com/kundajelab/DeepExplain.git
pip install tensorflow # or tensorflow-gpu if you are using a GPU
pip install bpnet
echo 'export HDF5_USE_FILE_LOCKING=FALSE' >> ~/.bashrc
```

When using bpnet from the command line, don't forget to activate the `bpnet` conda environment before:

```bash
# activate the bpnet conda environment
source activate bpnet

# run bpnet
bpnet <command> ...
```

### (Optional) Install `vmtouch` to use with `bpnet train --vmtouch`

To use the `--vmtouch` in `bpnet train` command and thereby speed-up data-loading, install [vmtouch](https://hoytech.com/vmtouch/). vmtouch is used to load the bigWig files into system memory cache which allows multiple processes to access
the bigWigs loaded into memory. 

Here's how to build and install vmtouch:

```bash
# ~/bin = directory for localy compiled binaries
mkdir -p ~/bin
cd ~/bin
# Clone and build
git clone https://github.com/hoytech/vmtouch.git vmtouch_src
cd vmtouch_src
make
# Move the binary to ~/bin
cp vmtouch ../
# Add ~/bin to $PATH
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
```
