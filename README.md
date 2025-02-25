# BPNet

BPNet is a python package with a CLI to train and interpret base-resolution deep neural networks trained on functional genomics data such as ChIP-nexus or ChIP-seq. This is the improved version of the repo associated the BPNet paper. We suggest using this going forward instead of the original repo.

## Table of contents

[Installation](https://github.com/kundajelab/bpnet-refactor?tab=readme-ov-file#installation) </br>
[Tutorial](https://github.com/kundajelab/bpnet-refactor?tab=readme-ov-file#tutorial) </br>
[How to Cite](https://github.com/kundajelab/bpnet-refactor?tab=readme-ov-file#how-to-cite) </br>
[Full Documentation](https://github.com/kundajelab/bpnet-refactor/wiki) </br>

## Installation

This section will discuss the packages needed to train and interpret the BPNet models. Firstly, it is recommended that you use a GPU for model training and have the necessary NVIDIA drivers and CUDA already installed. You can verify that your machine is set up to use GPU's properly by executing the nvidia-smi command and ensuring that the command returns information about your system GPU(s) (rather than an error). Secondly there are three ways to ensure you have the necessary packages which we detail below:


Instead of installing the BPNet repo by yourself, you can also try the Docker or Anvil options to train/use BPNet models.

### 1. Docker 

Download and install the latest version of Docker for your platform. Here is the link for the installers -[Docker Installers](https://docs.docker.com/get-docker/). Run the docker run command below to open an environment with all the packages installed.

```
docker pull vivekramalingam/tf-atlas:gcp-modeling_v2.1.0-rc.1

docker run -it --rm --cpus=10 --memory=32g --gpus device=1 --mount src=/mnt/bpnet-models/,target=/mydata,type=bind vivekramalingam/tf-atlas:gcp-modeling_v2.1.0-rc.1

```

### 2. Anvil

The NHGRI's AnVIL (Genomic Data Science Analysis, Visualization, and Informatics Lab-space) platform allows researchers with no/minimal computation skills to run analysis tools with just a few mouse clicks. Using our BPNet workflow on the AnVIL platform you can train state-of-art deep learning BPNet models for ChIP-seq data without any programming experience. We highly recommand this option for biologists. It is also highly scalable as GPU provisioning on the cloud etc., is automatically handled by AnVIL. We trained our Atlas scale models on the entire ENCODE compendium using the AnVIL platform. 

<a href="https://anvil.terra.bio/#workspaces/terra-billing-vir/tf-atlas/workflows">Anvil/Terra </a> 

It is also available for training models for <a href="https://anvil.terra.bio/#workspaces/terra-billing-vir/chromatin-atlas"> chromatin accessibility </a> experiments.  


A video turorial for training BPNet models on AnVIL is available here:

<a href="https://www.youtube.com/watch?v=wQ4xeLIGozM&list=PL6aYJ_0zJ4uA6ydpiPQwAH-v4BfVO0hcw&index=5">Video Tutorial</a>

***TODO - Anvil Tutorial


### 3. Local Installation

#### 3.1. Install Miniconda

Download and install the latest version of Miniconda for your platform. Here is the link for the installers - <a href="https://docs.conda.io/en/latest/miniconda.html">Miniconda Installers</a>

#### 3.2. Create new virtual environment

Create a new virtual environment and activate it as shown below

```
conda create --name bpnet python=3.8
conda activate bpnet
```

#### 3.3. Install basepairmodels

```
pip install git+https://github.com/kundajelab/bpnet-refactor.git

```
#### 3.4 Additional tools to install

For the following steps you will need `samtools` `bamtools` and `bedGraphToBigWig`, which are not 
installed as part of this repository. 

The tools can be installed via the links below or using `conda`.

<a href="http://www.htslib.org/download/">samtools</a>

<a href="https://anaconda.org/bioconda/bamtools">bamtools</a>

<a href="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/">bedGraphToBigWig (Linux 64-bit)</a>

<a href="http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/">bedGraphToBigWig (Mac OSX 10.14.6)</a>

```
conda install -y -c bioconda samtools=1.1 bedtools ucsc-bedgraphtobigwig

```


## Tutorial

### 0. Optional additional walk through for downloading and preprocessing an example data:

[Download-test-data](https://github.com/kundajelab/bpnet-refactor/wiki/Download-test-data)

[Preprocessing](https://github.com/kundajelab/bpnet-refactor/wiki/Preprocessing)

[Outlier_removal](https://github.com/kundajelab/bpnet-refactor/wiki/Outlier_removal)

[Background_generation](https://github.com/kundajelab/bpnet-refactor/wiki/Background_generation)

### 1. Train a model!

Before we start training, we need to compile a json file that contains information about the input data. We will call this file `input_data.json`. Here is a sample json file that shows how to specify the input data information for the data. The data is organized into tasks and tracks. In this example we have one task and two tracks, the plus and the minus strand. Each task has 4 required keys, with values corresponding to tracks calculated in the preprocessing steps:

`signal`: the `plus` and `minus` bigwig tracks 

`loci`: the bed file with the filtered peaks

`background_loci` (optional): the bed file with gc-matched background regions (`source`), and the ratio of positive-to-negative regions used when generating the gc-matched negatives file, expressed as a decimal (`ratio`) 

`bias` (optional): the `plus` and `minus` control bigwig tracks

Note that the `input_data.json` file is used for multiple downstream steps.

```
{
    "0": {
        "signal": {
            "source": ["ENCSR000EGM/data/plus.bw", 
                       "ENCSR000EGM/data/minus.bw"]
        },
        "loci": {
            "source": ["ENCSR000EGM/data/peaks_inliers.bed"]
        },
        "background_loci": {
            "source": ["ENCSR000EGM/data/gc_negatives.bed"],
            "ratio": [0.25]
        },
        "bias": {
            "source": ["ENCSR000EGM/data/control_plus.bw",
                       "ENCSR000EGM/data/control_minus.bw"],
            "smoothing": [null, null]
        }
    }
}
```

Additionally, we need a json file to specify parameters for the BPNet architecture. Let's call this json file
`bpnet_params.json` 

```
{
    "input_len": 2114,
    "output_profile_len": 1000,
    "motif_module_params": {
        "filters": [64],
        "kernel_sizes": [21],
        "padding": "valid"
    },
    "syntax_module_params": {
        "num_dilation_layers": 8,
        "filters": 64,
        "kernel_size": 3,
        "padding": "valid",
        "pre_activation_residual_unit": true
    },
    "profile_head_params": {
        "filters": 1,
        "kernel_size":  75,
        "padding": "valid"
    },
    "counts_head_params": {
        "units": [1],
        "dropouts": [0.0],
        "activations": ["linear"]
    },
    "profile_bias_module_params": {
        "kernel_sizes": [1]
    },
    "counts_bias_module_params": {
    },
    "use_attribution_prior": false,
    "attribution_prior_params": {
        "frequency_limit": 150,
        "limit_softness": 0.2,
        "grad_smooth_sigma": 3,
        "profile_grad_loss_weight": 200,
        "counts_grad_loss_weight": 100        
    },
    "loss_weights": [1, 42],
    "counts_loss": "MSE"
}
```

The `loss_weights` field has two values: the `profile` loss weight and the 
`counts` loss weight. Profile loss weight is always set to 1. Optimal counts loss weight can be automatically generated using the following command:

```
bpnet-counts-loss-weight --input-data input_data.json
```

Finally, we will make the `splits.json` file, which contains information about the chromosomes that are used for 
validation and test. Here is a sample that contains one split.

```
{
    "0": {
        "test":
            ["chr7", "chr13", "chr17", "chr19", "chr21", "chrX"],
        "val":
            ["chr10", "chr18"],
        "train":
            ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr8", "chr9", "chr11", "chr12", "chr14", "chr15", "chr16", "chr21", "chr22", "chrY"]
    }
}
```

Now that we have our data prepped, we can train our first model!!

The command to train a model is called `bpnet-train`. 

You can kick start the training process by executing this command in your shell

```
BASE_DIR=ENCSR000EGM
DATA_DIR=$BASE_DIR/data
MODEL_DIR=$BASE_DIR/models
REFERENCE_DIR=reference
CHROM_SIZES=$REFERENCE_DIR/hg38.chrom.sizes
REFERENCE_GENOME=$REFERENCE_DIR/hg38.genome.fa
CV_SPLITS=$BASE_DIR/splits.json
INPUT_DATA=$BASE_DIR/input_data.json
MODEL_PARAMS=$BASE_DIR/bpnet_params.json


mkdir $MODEL_DIR
bpnet-train \
        --input-data $INPUT_DATA \
        --output-dir $MODEL_DIR \
        --reference-genome $REFERENCE_GENOME \
        --chroms $(paste -s -d ' ' $REFERENCE_DIR/chroms.txt) \
        --chrom-sizes $CHROM_SIZES \
        --splits $CV_SPLITS \
        --model-arch-name BPNet \
        --model-arch-params-json $MODEL_PARAMS \
        --sequence-generator-name BPNet \
        --model-output-filename model \
        --input-seq-len 2114 \
        --output-len 1000 \
        --shuffle \
        --threads 10 \
        --epochs 100 \
	--batch-size 64 \
	--reverse-complement-augmentation \
	--early-stopping-patience 10 \
	--reduce-lr-on-plateau-patience 5 \
        --learning-rate 0.001
```

Note: It might take a few minutes for the training to begin once the above command has been issued, be patient and you should see the training eventually start. 

### 2. Model Prediction

Once the training is complete we can generate predictions on the test set chromosomes to understand how well the model performs on regions not seen during training. 

```
PREDICTIONS_DIR=$BASE_DIR/predictions_and_metrics
mkdir $PREDICTIONS_DIR
bpnet-predict \
        --model $MODEL_DIR/model_split000 \
        --chrom-sizes $CHROM_SIZES \
        --chroms chr7 chr13 chr17 chr19 chr21 chrX \
        --test-indices-file None \
        --reference-genome $REFERENCE_GENOME \
        --output-dir $PREDICTIONS_DIR \
        --input-data $INPUT_DATA \
        --sequence-generator-name BPNet \
        --input-seq-len 2114 \
        --output-len 1000 \
        --output-window-size 1000 \
        --batch-size 64 \
        --reverse-complement-average \
        --threads 2 \
        --generate-predicted-profile-bigWigs
```

This script will output test metrics and also output bigwig tracks if the 
`--generate-predicted-profile-bigWigs` is specified

It is usefull to also make predictions on all the peaks as the model predictions provide denoised version of the observed signal.
```
bpnet-predict \
    --model $MODEL_DIR/model_split000 \
    --chrom-sizes $CHROM_SIZES \
    --chroms chr1 chr2 chrX chr3 chr4 chr5 chr6 chr7 chr10 chr8 chr14 chr9 chr11 chr13 chr12 chr15 chr16 chr17 chrY chr18 chr19 chrM \
    --test-indices-file None \
    --reference-genome $REFERENCE_GENOME \
    --output-dir $PREDICTIONS_DIR \
    --input-data $INPUT_DATA \
    --sequence-generator-name BPNet \
    --input-seq-len 2114 \
    --output-len 1000 \
    --output-window-size 1000 \
    --batch-size 64 \
    --reverse-complement-average \
    --threads 2 \
    --generate-predicted-profile-bigWigs
```

### 3. Compute importance scores

"Understanding why a model makes a certain prediction can be as crucial as the prediction’s accuracy in many applications". [SHAP](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction. In our case we will assign each nucleotide an importance value or how much it contributed for the binding at each input region according to our model. A model that can predict the binding signal well at the test regions presumably must have learnt usefull features and thus the importance score values can shed light in to the driving sequence feature including the Transcription factor binding motifs. 

```
SHAP_DIR=$BASE_DIR/shap
mkdir $SHAP_DIR
bpnet-shap \
        --reference-genome $REFERENCE_GENOME \
        --model $MODEL_DIR/model_split000  \
        --bed-file $DATA_DIR/peaks_inliers.bed \
        --chroms chr1 \
        --output-dir $SHAP_DIR \
        --input-seq-len 2114 \
        --control-len 1000 \
        --task-id 0 \
        --input-data $INPUT_DATA \
        --generate-shap-bigWigs

```

This script will output shap bigwig tracks which are usefull for visualizing in a genome browser the bases/motif that are important for binding example if
`--generate-shap-bigWigs`  is specified.

UCSC, WashU browsers currently support visualizing the importance scores - use the [Dynseq](https://kundajelab.github.io/dynseq-pages/) option.



#### Please cite:

<a href=https://www.nature.com/articles/s41588-022-01194-w> The dynseq browser  track shows context-specific features at nucleotide resolution </a>

S Nair, A Barrett, D Li, BJ Raney, BT Lee, P Kerpedjiev, V Ramalingam, ...
Nature genetics 54 (11), 1581-1583


### 4. Discover motifs with TF-modisco

"TF-MoDISco is a biological motif discovery algorithm that differentiates itself by using attribution scores from a machine learning model, in addition to the sequence itself, to guide motif discovery. Using the attribution scores, as opposed to just the sequence, can be beneficial because the attributions fine-map the specific sequence drivers of biology. Although in many of our examples this model is BPNet and the attributions are from DeepLIFT/DeepSHAP, there is no limit on what attribution algorithm is used, or what model the attributions come from."

This step allows one to get a list of sequence motifs that drive the signal in the studied experiment.

Use the newer version of TF-modisco called modisco-lite to discover the motifs. Support to directly use modisco-lite from the BPNet repo will be added later. For now use: https://github.com/jmschrei/tfmodisco-lite


### 5. Discover location of the identified motifs with FiNeMo

While TF-modisco allows one to get a list of sequence motifs, FiNeMo allows one to map the location of these motifs in all the input regions. FiNeMO is a GPU-accelerated hit caller for retrieving TFMoDISCo motif occurences from machine-learning-model-generated contribution scores.

Support to directly use FiNeMO from the BPNet repo will be added later. For now use: https://github.com/austintwang/finemo_gpu

## How to Cite

If you're using BPNet in your work, please cite the original BPNet paper:

Avsec, Ž., Weilert, M., Shrikumar, A. et al. Base-resolution models of transcription-factor binding reveal soft motif syntax. Nat Genet 53, 354–366 (2021). 

And reach out to Vivekanandan Ramalingam and Anshul Kundaje to discuss how to cite this update version. An preprint associated with this new version will soon be released and will be pointed out here. 

