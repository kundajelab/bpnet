## Reports

This directory contains Jupyter Notebooks and scripts needed to generate motif reports.

### Library dependencies
These are the libraries needed. You can probably get away with different versions for some of these, but these are the versions used for development:
- h5py 2.10.0
- Jupyter notebook 6.0.3
- NumPy 1.18.5
- MatPlotLib 3.3.1
- MEME 5.0.1 (specifically for TOMTOM)
- [MOODS](https://github.com/jhkorhonen/MOODS/wiki/Getting-started) 1.9.4.1
- Pandas 1.1.1
- [Pomegranate](https://pomegranate.readthedocs.io/en/latest/index.html) 0.14.2
- SciPy 1.5.2
- sklearn 0.23.2
- [TF-MoDISco](https://github.com/kundajelab/tfmodisco) 0.5.14.1
- tqdm 4.48.2
- [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) 0.5.1
- [VDOM](https://github.com/nteract/vdom) 0.6

### `view_tfmodisco_results.ipynb`
Visualizes the TF-MoDISco motifs, including:
- PFM, PWM, CWM, and hCWM of each discovered motif
- Average predicted/observed profile underlying motif seqlets
- Distance distribution of motif seqlets to peak summits
- TOMTOM matches of each motif
- Examples of seqlet importance scores for each motif

This notebook requires:
- TF-MoDISco result HDF5
- Peak predictions HDF5 of the following format:

		`coords`:
		    `coords_chrom`: N-array of chromosome (string)
		    `coords_start`: N-array
		    `coords_end`: N-array
		`predictions`:
		    `log_pred_profs`: N x T x O x 2 array of predicted log profile
		        probabilities
		    `log_pred_counts`: N x T x 2 array of log counts
		    `true_profs`: N x T x O x 2 array of true profile counts
		    `true_counts`: N x T x 2 array of true counts

- Importance scores HDF5 of the following format:

		`coords_chrom`: N-array of chromosome (string)
		`coords_start`: N-array
		`coords_end`: N-array
		`hyp_scores`: N x L x 4 array of hypothetical importance scores
		`input_seqs`: N x L x 4 array of one-hot encoded input sequences

- Set of all peaks as ENCODE NarrowPeak format (used for distance distribution of seqlets to peak summits)

Note that the N sequences in the importance scores must be precisely those that TF-MoDISco was run on (in the exact order). N is the number of peaks, T is the number of tasks (for single task models, that is just 1), O is the output profile length (e.g. 1000bp), L is the input sequence length (e.g. 2114bp), and 2 is for the two strands.

We also assume that TF-MoDISco wsa run only on the central 400bp of the importance scores.

### `showcase_motifs_and_profiles.ipynb`
For each TF-MoDISco motif, visualizes a sample of:
- Predicted/observed profile of that sequence
- Importance scores for that sequence
- The underlying seqlet

This notebook requires:
- TF-MoDISco result HDF5
- Peak predictions HDF5 (same format as above)
- Importance scores HDF5 (same format as above)

### `summarize_motif_hits.ipynb`
For the set of TF-MoDISco motifs, this notebook will run MOODS motif instance calling and analyze the resulting hits. This notebook will visualize:
- The distribution of how many motif hits are found per peak
- The proportion of peaks that have each type of motif
- Example importance score tracks with highlighted motif hits
- Co-occurrence of different motifs in peaks
- Homotypic motif densities in peaks
- Distribution of distances between strongly co-occurring motifs

This notebook requires:
- TF-MoDISco result HDF5
- Importance scores HDF5 (same format as above)
- Set of all peaks as ENCODE NarrowPeak format
- A location to store the MOODS results

MOODS calling is performed by `moods.py`.

### `cluster_motif_hits_and_peaks.ipynb`
From the set of TF-MoDISco motifs and the motif hits in peaks, this notebook will visualize:
- Subclustering structure within motifs themselves
- Clustering of peak embeddings based on which peaks contain which motifs

This notebook requires:
- TF-MoDISco result HDF5
- Importance scores HDF5 (same format as above)
- Location where MOODS results were stored
- Embeddings of peaks as an N x L x F array
	- N is the number of peaks, L is the length along the final dilated convolutional axis, and F is the number of filters
	- The peaks N must be in the same order as in the peaks BED file which was used to compute motif hits

### `model_performance.ipynb`
Plots the profile and counts performance of a model, including:
- CDFs of profile performance metrics over peaks (MNLL, cross entropy, and JSD are min-max-normalized)
- Scatter plot of predicted and true log counts, and their correlation

This notebook requires:
- Peak predictions HDF5 (same format as above)
- Path to metrics directory:
	- This directory must contain the subdirectories `plus/` and `minus/`, each with NumPy arrays of `{key}.npz` for each metric key
	- Any min-max-normalization must have already happened prior to saving these vectors
