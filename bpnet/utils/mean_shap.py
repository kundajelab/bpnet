import pandas as pd

import h5py
import hdf5plugin
import numpy as np

import argparse
import gc


parser = argparse.ArgumentParser(description="calculate mean shap over the given h5s")
parser.add_argument("--counts_shaps", type=str, default=None, help="counts shap h5s - counts_fold0.h5,counts_fold1.h5,counts_fold2.h5")
parser.add_argument("--profile_shaps", type=str, default=None, help="profile shap h5s - profile_fold0.h5,profile_fold1.h5,profile_fold2.h5")
parser.add_argument("--output_dir", type=str,default='/cromwell_root/', help="output directory for the mean_shap")

def mean_shap(shaps_list,output_path):
    hyp_scores_lst=[]
    chrom_lst=[]
    start_lst=[]
    end_lst=[]
    input_seqs_lst=[]
    for shap_h5 in shaps_list.split(','):
        try:
            f = h5py.File(shap_h5, 'r')
            hyp_scores_lst.append(f['hyp_scores'][()])
            chrom_lst.append(f['coords_chrom'][()])
            start_lst.append(f['coords_start'][()])
            end_lst.append(f['coords_end'][()])
            input_seqs_lst.append(f['input_seqs'][()])
            f.close()
        except:
            print(f'{shap_h5} does not exist')

    if(sum([all(all(element == chrom_lst[0]) for element in chrom_lst),
     all(all(element == start_lst[0]) for element in start_lst),
     all(all(element == end_lst[0]) for element in end_lst)])==3):

        hyp_scores_mean = np.nanmean(np.array(hyp_scores_lst),axis=0)

        num_examples = hyp_scores_mean.shape[0]

        f = h5py.File(output_path, "w")

        coords_chrom_dset = f.create_dataset(
            "coords_chrom", (num_examples,),
            dtype=h5py.string_dtype(encoding="ascii"), **hdf5plugin.Blosc()
        )
        coords_chrom_dset[:] = chrom_lst[0]

        coords_start_dset = f.create_dataset(
            "coords_start", (num_examples,), dtype=int, **hdf5plugin.Blosc()
        )
        coords_start_dset[:] = start_lst[0]

        coords_end_dset = f.create_dataset(
            "coords_end", (num_examples,), dtype=int, **hdf5plugin.Blosc()
        )
        coords_end_dset[:] = end_lst[0]

        hyp_scores_dset = f.create_dataset(
            "hyp_scores", (num_examples, hyp_scores_mean.shape[1], hyp_scores_mean.shape[2]), **hdf5plugin.Blosc()
        )
        hyp_scores_dset[:, :, :] = hyp_scores_mean

        input_seqs_dset = f.create_dataset(
            "input_seqs", (num_examples, hyp_scores_mean.shape[1], hyp_scores_mean.shape[2]), **hdf5plugin.Blosc()
        )
        input_seqs_dset[:, :, :] = input_seqs_lst[0]

        f.close()
        
        del(input_seqs_lst,hyp_scores_mean,chrom_lst,start_lst,end_lst,hyp_scores_lst)
        
        gc.collect()
        
        
args = parser.parse_args()

if args.counts_shaps:
    mean_shap(args.counts_shaps,f"{args.output_dir}/counts_mean_shap_scores.h5")

if args.profile_shaps:
    mean_shap(args.profile_shaps,f"{args.output_dir}/profile_mean_shap_scores.h5")