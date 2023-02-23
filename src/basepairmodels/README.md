# BasepairModels

BasepairModels is a python package with a CLI to train and interpret base-resolution deep neural networks trained on functional genomics data such as ChIP-nexus or ChIP-seq.

## Installation

### 1. Install Miniconda

Download and install the latest version of Miniconda for your platform. Here is the link for the installers - <a href="https://docs.conda.io/en/latest/miniconda.html">Miniconda Installers</a>

### 2. Create new virtual environment

Create a new virtual environment and activate it as shown below

```
conda create --name basepairmodels python=3.7
conda activate basepairmodels
```

### 3. Install basepairmodels

```
pip install git+https://github.com/kundajelab/basepairmodels.git
```

## Tutorial on how to use the command line interface

### 1. Experimental dataset

For this tutorial we'll use experimental CHIP-seq data, for the transcription factor CTCF obtained for the K562 cell line, which is available on the ENCODE data portal. There are 5 such experiments that we find in ENCODE, you can see them listed here <a href="https://www.encodeproject.org/search/?type=Experiment&status=released&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&biosample_ontology.term_name=K562&assay_title=Histone+ChIP-seq&assay_title=TF+ChIP-seq&target.label=CTCF">CHIP-seq CTCF K562 </a> . We'll restrict ourselves to one experiment 
<a href="https://www.encodeproject.org/experiments/ENCSR000EGM/">ENCSR000EGM</a>

Download the .bam files for the two replicates shown below in the image.

<img src="./docs-build/tutorial/bpnet/images/tutorial-data.png" alt="replicate bams"/>

The two replicates are isogenic replicates (biological). A more detailed explanation 
of the various types of replicates can be found <a href="https://www.encodeproject.org/data-standards/terms/">here</a>.

Links to the replicate bam files provided below.

<a href="https://www.encodeproject.org/files/ENCFF198CVB/@@download/ENCFF198CVB.bam">ENCFF198CVB</a>

<a href="https://www.encodeproject.org/files/ENCFF488CXC/@@download/ENCFF488CXC.bam">ENCFF488CXC</a>

```
wget https://www.encodeproject.org/files/ENCFF198CVB/@@download/ENCFF198CVB.bam -O rep1.bam
wget https://www.encodeproject.org/files/ENCFF488CXC/@@download/ENCFF488CXC.bam -O rep2.bam
```

Now download the control for the experiment, which is available here <a href="https://www.encodeproject.org/experiments/ENCSR000EHI/">ENCSR000EHI</a>

Download the bam file shown in the image below.

<img src="./docs-build/tutorial/bpnet/images/tutorial-control.png" alt="control bam"/>

Link provided below

<a href="https://www.encodeproject.org/files/ENCFF023NGN/@@download/ENCFF023NGN.bam">ENCFF023NGN</a>

```
wget https://www.encodeproject.org/files/ENCFF023NGN/@@download/ENCFF023NGN.bam -O control.bam
```

#### 1.1 Preprocessing steps to generate bigwig counts tracks

For the following steps you will need `samtools` `bamtools` and `bedGraphToBigWig`, which are not 
installed as part of this repository. 

Here are some links to help install those tools.

<a href="http://www.htslib.org/download/">samtools</a>

<a href="https://anaconda.org/bioconda/bamtools">bamtools</a>

<a href="http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/">bedGraphToBigWig (Linux 64-bit)</a>

<a href="http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/">bedGraphToBigWig (Mac OSX 10.14.6)</a>

##### 1.1.1 Merge the two replicates and create and index

```
samtools merge -f merged.bam rep1.bam rep2.bam
samtools index merged.bam
```

##### 1.1.2 Create bigwig files using bedtools via intermediate bedGraph files

**Experiment**
```
# get coverage of 5’ positions of the plus strand
bedtools genomecov -5 -bg -strand + \
        -g hg38.chrom.sizes -ibam merged.bam \
        | sort -k1,1 -k2,2n > plus.bedGraph

# get coverage of 5’ positions of the minus strand
bedtools genomecov -5 -bg -strand - \
        -g hg38.chrom.sizes -ibam merged.bam \
        | sort -k1,1 -k2,2n > minus.bedGraph

# Convert bedGraph files to bigWig files
bedGraphToBigWig plus.bedGraph hg38.chrom.sizes plus.bw
bedGraphToBigWig minus.bedGraph hg38.chrom.sizes minus.bw
```
**Control**
```
# get coverage of 5’ positions of the plus strand
bedtools genomecov -5 -bg -strand + \
        -g hg38.chrom.sizes -ibam control.bam \
        | sort -k1,1 -k2,2n > control_plus.bedGraph

bedtools genomecov -5 -bg -strand - \
        -g hg38.chrom.sizes -ibam control.bam \
         | sort -k1,1 -k2,2n > control_minus.bedGraph

# Convert bedGraph files to bigWig files
bedGraphToBigWig control_plus.bedGraph hg38.chrom.sizes control_plus.bw
bedGraphToBigWig control_minus.bedGraph hg38.chrom.sizes control_minus.bw
```

#### 1.2 Identify peaks

For the purposes of this tutorial we will use the optimal IDR thresholded peaks that are already available in the ENCODE data portal. We will use the the narrowPeak files that are in BED6+4 format. Explanation of what each of the 10 fields means can be found  <a href="http://genome.ucsc.edu/FAQ/FAQformat.html#format12">here</a>. Currently, only this format is supported but in the
future support for more formats will be added.

See image below that shows the file listed in the ENCODE data portal

<img src="./docs-build/tutorial/bpnet/images/tutorial-idrpeaks.png">

Link to download the file 
<a href="https://www.encodeproject.org/files/ENCFF396BZQ/@@download/ENCFF396BZQ.bed.gz">ENCFF396BZQ</a>

#### 1.3 Organize you data

We suggest creating a directory structure to store the data, models, predictions, metrics, importance scores, discovered motifs, plots & visualizations etc. that will make it easier for you to organize and maintain your work. Let's start by creating a parent directory for the experiment and moving the bigwig files and peaks file from section 1.1 & 1.2 to a data directory

```
mkdir ENCSR000EGM
mkdir ENCSR000EGM/data
mv *.bw ENCSR000EGM/data
mv peaks.bed ENCSR000EGM/data
```

Once this is done, your directory heirarchy should resemble this

<div align="left"><img src="./docs-build/tutorial/bpnet/images/directory-data.png"></div>

#### 1.4 Reference genome

For the sake of this tutorial let's assume we have a `reference` directory at the same level as the `ENCSR000EGM` experiment directory. In the `reference` directory we will place 4 files the hg38 fasta file, the index to the fasta file, chromosome sizes file and one text file that contains a list of chromosomes we care about (one per line - chr1-22, X, Y, M and exclude the rest). The directory structure looks like this.

<div align="left"><img src="./docs-build/tutorial/bpnet/images/directory-reference.png"></img>
</div>

### 2. Train a model!

Before we start training, we need to compile a json file that contains information about the input data. Here is a sample json file that shows how to specify the input data information for the data we organized in Section 1.3. The data is organized into tasks and tracks. In this example we have one task and two tracks, the plus and the minus strand. Each track has 4 required keys `signal`, `loci`, `background_loci`, & `bias`. 


```
{
    "0": {
        "signal": {
            "source": ["/users/john/ENCSR000EGM/data/plus.bw", 
                       "/users/john/ENCSR000EGM/data/minus.bw"]
        },
        "loci": {
            "source": ["/users/john/ENCSR000EGM/data/peaks.bed"]
        },
        "background_loci": {
            "source": [],
            "ratio": []
        },
        "bias": {
            "source": ["/users/john/ENCSR000EGM/data/control_plus.bw",
                       "/users/john/ENCSR000EGM/data/control_minus.bw"],
            "smoothing": [null, null]
        }
    }
}
```

We'll call the above json file `input_data.json`. Additionally we need a json
file to specify parameters for the BPNet architecture. Let's call this json file
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
    "loss_weights": [1, 42]
}
```

The `loss_weights` field has two values the `profile` loss weight and the 
`counts` loss weight. The counts loss weight can be automatically generated 
using the following command

```
counts_loss_weight --input-data $INPUT_DATA
```

Now that we have our data prepped, we can train our first model!!

The command to train a model is called `train`. 

You can kick start the training process by executing this command in your shell

```
BASE_DIR=/users/john/ENCSR000EGM
DATA_DIR=$BASE_DIR/data
MODEL_DIR=$BASE_DIR/models
REFERENCE_DIR=$BASE_DIR/reference
CHROM_SIZES=$REFERENCE_DIR/hg38.chrom.sizes
REFERENCE_GENOME=$REFERENCE_DIR/hg38.genome.fa
CV_SPLITS=$BASE_DIR/splits.json
INPUT_DATA=$BASE_DIR/input_data.json
MODEL_PARAMS=$BASE_DIR/bpnet_params.json


mkdir $MODEL_DIR
train \
    --input-data $INPUT_DATA \
    --output-dir $MODEL_DIR \
    --reference-genome $REFERENCE_GENOME \
    --chroms $(paste -s -d ' ' $REFERENCE_DIR/hg38_chroms.txt) \
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
    --learning-rate 0.004
```

The `splits.json` file contains information about the chromosomes that are used for validation and test. Here is a sample that contains one split.

```
{
    "0": {
        "val": ["chr10", "chr8"], 
        "test": ["chr1"]
    }
}
```

Note: It might take a few minutes for the training to begin once the above command has been issued, be patient and you should see the training eventually start. 

### 3. Predict on test set

Once the training is complete we can generate predictions on the test chromosome.

```
PREDICTIONS_DIR=$BASE_DIR/predictions_and_metrics
mkdir $PREDICTIONS_DIR
predict \
    --model $MODEL_DIR/model_split000.h5 \
    --chrom-sizes $REFERENCE_DIR/GRCh38_EBV.chrom.sizes \
    --chroms chr1 \
    --reference-genome $REFERENCE_DIR/hg38.genome.fa \
    --output-dir $PREDICTIONS_DIR \
    --input-data $BASE_DIR/input.json \
    --sequence-generator-name BPNet \
    --input-seq-len 2114 \
    --output-len 1000 \
    --output-window-size 1000 \
    --batch-size 64 \
    --threads 2 \
    --generate-predicted-profile-bigWigs
```

This script will output test metrics and also output bigwig tracks if the 
`--generate-predicted-profile-bigWigs` is specified

### 4. Compute importance scores

```
SHAP_DIR=$BASE_DIR/shap
mkdir $SHAP_DIR
shap_scores \
    --reference-genome $REFERENCE_GENOME \
    --model $MODEL_DIR/model_split000.h5  \
    --bed-file $DATA_DIR/peaks_med.bed \
    --chroms chr1 \
    --output-dir $SHAP_DIR \
    --input-seq-len 2114 \
    --control-len 1000 \
    --task-id 0 \
    --input-data $BASE_DIR/input.json 
```

### 5. Discover motifs with TF-modisco

```
MODISCO_PROFILE_DIR=$BASE_DIR/modisco_profile
mkdir $MODISCO_PROFILE_DIR
motif_discovery \
    --scores-path $SHAP_DIR/profile_scores.h5 \
    --output-directory $MODISCO_PROFILE_DIR

MODISCO_COUNTS_DIR=$BASE_DIR/modisco_counts
mkdir $MODISCO_COUNTS_DIR
motif_discovery \
    --scores-path $SHAP_DIR/counts_scores.h5 \
    --output-directory $MODISCO_COUNTS_DIR
```

### 6. Filter the peaks file for outliers

```
outliers \
    --input-data $INPUT_DATA  \
    --quantile 0.99 \
    --quantile-value-scale-factor 1.2 \
    --task 0 \
    --chrom-sizes $REFERENCE_DIR/hg38.chrom.sizes \
    --chroms $(paste -s -d ' ' $REFERENCE_DIR/hg38_chroms.txt) \
    --sequence-len 1000 \
    --blacklist $BASE_DIR/blacklist.bed \
    --global-sample-weight 1.0 \
    --output-bed $DATA_DIR/inliers.bed
```

### 7. Compute embeddings from intermediate layers

```
EMBEDDINGS_DIR=$BASE_DIR/embeddings
mkdir $EMBEDDINGS_DIR
embeddings \
    --model $MODEL_DIR/model_split000.h5 \
    --reference-genome $REFERENCE_GENOME \
    --embeddings-layer-name main_profile_head \
    --cropped-size 1000 \
    --input-layer-shape 2114 4 \
    --peaks $DATA_DIR/peaks_med.bed \
    --output-directory $EMBEDDINGS_DIR \
    --batch-size 256
```
