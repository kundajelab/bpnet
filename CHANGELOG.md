Changes since Surag started cleanup.

1 March
- removed unused imports
- removed code corresponding to ChromBPNet and attribution priors.
- Removed CustomMeanSquaredError and MultichannelMultinomialNLL
- Removed api directory
- changed losses.py to remove duplication, moved functions from custommodel.py
- changed CLI options and added "bpnet-" prefix
- simplified organization, renamed directories to understandable names (e.g. mseqgen -> generators, genomicsarchsandlosses -> model)
- removed repetition of counts loss function (two functions for mse and poisson loss calculation with identical logic) in custommodel.py

9 March
- removed random genomewide negatives
- removed logic for background only training (releted to bias model training for ChromBPNet)

14 March
- removed getChromPositions from seqUtils
- started adding unit tests beginning with sequtils
- added some unit tests for generator 
- fixed issue where rev comp augmentation wasn't happening

23 March
- Upgrading to tf2.11 (py3.10), removing CustomScore since it's no longer required. Removing h5 extension so that files are saved in the SavedModel format instead of h5.
- Also requires a small change in writing the shap hdf5 (string compression with blosc causes seg fault).
- Add modisco-lite and shap to setup.py
- added reverse complement averaging for predict.py

26 March
- Test revert to tf2.4.1 (py3.7) since the newer version seems to give lower model performance for experiments with lower peaks
