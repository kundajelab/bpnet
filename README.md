## TODO

- [ ] try to get rid of pybedtools entirely
  - [ ] replace with pyranges

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




## Dependencies

modisco
plotnine - hidden


argh
attr

concise
deepexplain

gin
h5py
joblib
keras
kipoi - can we use kipoi_utils?
kipoiseq
matplotlib
numpy
pandas

pyBigWig
related
scipy
sklearn
seaborn
statsmodels
tensorflow
tqdm

papermill
vdom
ipython - available by default with nbconvert?


## Optional

torch
pprint
fastparquet

ipywidgets
genomelake

pygam - remove?

## remove 

genomelake   # implement your own bigwigextractor
gin_train

pybedtools -> replace with pyranges
pysam  # replace to pyfaidx?
