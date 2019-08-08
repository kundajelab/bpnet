# Quick start

## Train BPNet

Train a model using an existing architecture [bpnet9](bpnet/premade/bpnet9.gin)

```bash
bpnet train --premade=bpnet9 dataspec.yml output_dir
```

Train a model by overriding some of the hyper-parameters:

```bash
bpnet train --premade=bpnet9 dataspec.yml --override='seq_width=200;n_dil_layers=3' output_dir
```

## Compute contribution scores

```bash
bpnet contrib $model_dir --method=deeplift $model_dir/imp.scores.h5
```

## Discover motifs with TF-MoDISco

```bash
bpnet modisco-run $model_dir/imp.scores.h5 --premade=modisco-50k $modisco_dir
```

## Determine motif instances with CWM scanning

```bash
bpnet cwm-scan $modisco_dir $modisco_dir/motif-instances.tsv.gz
```

See the end-to-end tutorial https://colab.research.google.com/drive/1VNsNBfugPJfJ02LBgvPwj-gPK0L_djsD for more information.


# Installation


```bash
pip install bpnet
```

<!-- ## Dependencies -->

<!-- modisco -->
<!-- plotnine - hidden -->


<!-- argh -->
<!-- attr -->

<!-- concise -->
<!-- deepexplain -->

<!-- gin -->
<!-- h5py -->
<!-- joblib -->
<!-- keras -->
<!-- kipoi - can we use kipoi_utils? -->
<!-- kipoiseq -->
<!-- matplotlib -->
<!-- numpy -->
<!-- pandas -->

<!-- pyBigWig -->
<!-- related -->
<!-- scipy -->
<!-- sklearn -->
<!-- seaborn -->
<!-- statsmodels -->
<!-- tensorflow -->
<!-- tqdm -->

<!-- papermill -->
<!-- vdom -->
<!-- ipython - available by default with nbconvert? -->


<!-- ## Optional -->

<!-- torch -->
<!-- pprint -->
<!-- fastparquet -->

<!-- ipywidgets -->
<!-- genomelake -->

<!-- pygam - remove? -->

<!-- ## remove  -->

<!-- genomelake   # implement your own bigwigextractor -->
<!-- gin_train -->

<!-- pybedtools -> replace with pyranges -->
<!-- pysam  # replace to pyfaidx? -->


<!-- ## TODO -->

<!-- - [ ] try to get rid of pybedtools entirely -->
<!--   - [ ] replace with pyranges -->

