import pandas as pd
import h5py
import numpy as np
import argparse
import gc
import pyBigWig
import os

parser = argparse.ArgumentParser(description="Take a mean of prediction h5s and save the outputs as both hdf5 format and bigwigs. PROVIDE ABSOLUTE PATHS!")

parser.add_argument("--prediction_h5s", type=str, required=True, help="prediction_h5s")
parser.add_argument("--generate_bigwigs", action='store_true', help="whether to generate bigWigs from the merged prediction h5s")
parser.add_argument("-c", "--chrom_sizes", type=str, help="path to chromosome sizes 2 column file - first row is chromosome name and the second column is the chromosome size. Required if generate_bigwigs option is given ")
parser.add_argument("-o", "--output_dir", type=str, help="Output bigwig file",required=True)


args = parser.parse_args()
print(args)

if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
    
def mean_predictions(predictions_list_str,output_path):
    pred_logcounts_lst=[]
    pred_profs_lst=[]
    log_pred_prob_lst=[]
    chrom_lst=[]
    start_lst=[]
    end_lst=[]
    for predictions_h5 in predictions_list_str.split(','):
        try:
            f = h5py.File(predictions_h5, 'r')
            coord_chrom = f['coords']['coords_chrom'][()]
            coord_start = f['coords']['coords_start'][()]
            coord_end = f['coords']['coords_end'][()]
            order = np.argsort([str(coord_chrom[i])+"_"+str(coord_start[i])+"_"+str(coord_end[i]) for i in range(f['coords']['coords_chrom'][()].shape[0])])
            pred_logcounts = f['predictions']['pred_logcounts'][()][order]
            pred_logcounts_lst.append(pred_logcounts)
            chrom_lst.append(coord_chrom[order])
            start_lst.append(coord_start[order])
            end_lst.append(coord_end[order])
            pred_prof = f['predictions']['pred_profs'][()][order]
            pred_profs_lst.append(pred_prof)
            pred_sum = np.sum(pred_prof,axis=1)
            pred_prob = np.swapaxes(pred_prof,1,2)/np.reshape(pred_sum,(pred_sum.shape[0],pred_sum.shape[1],1))
            pred_prob = np.swapaxes(pred_prob,1,2)
            log_pred_prob = np.log(pred_prob)
            log_pred_prob_lst.append(log_pred_prob)
            f.close()
        except:
            print(f'{predictions_h5} does not exist')

    if(sum([all(all(element == chrom_lst[0]) for element in chrom_lst),
     all(all(element == start_lst[0]) for element in start_lst),
     all(all(element == end_lst[0]) for element in end_lst)])==3):

        pred_logcounts_mean = np.nanmean(np.array(pred_logcounts_lst),axis=0)

        num_examples = pred_logcounts_mean.shape[0]
        
        log_pred_prob_mean = np.nanmean(np.array(log_pred_prob_lst),axis=0)
        
        exp_log_pred_prob_mean = np.exp(log_pred_prob_mean)
        
        pred_prob_mean = np.swapaxes(exp_log_pred_prob_mean,1,2)*np.reshape(np.exp(pred_logcounts_mean),
                                                                            (pred_logcounts_mean.shape[0], pred_logcounts_mean.shape[1],1))

        pred_prob_mean = np.swapaxes(pred_prob_mean,1,2)
                

        f = h5py.File(output_path, "w")

        coords_chrom_dset = f.create_dataset(
            "coords_chrom", (num_examples,),
            dtype=h5py.string_dtype(encoding="ascii")
        )
        coords_chrom_dset[:] = chrom_lst[0]

        coords_start_dset = f.create_dataset(
            "coords_start", (num_examples,), dtype=int
        )
        coords_start_dset[:] = start_lst[0]

        coords_end_dset = f.create_dataset(
            "coords_end", (num_examples,), dtype=int
        )
        coords_end_dset[:] = end_lst[0]

        pred_logcounts_dset = f.create_dataset(
            "pred_logcounts", (num_examples, pred_logcounts_mean.shape[1])
        )
        pred_logcounts_dset[:, :] = pred_logcounts_mean
        
        pred_prof_dset = f.create_dataset(
            "pred_prof", (num_examples, pred_prob_mean.shape[1], pred_prob_mean.shape[2])
        )
        pred_prof_dset[:, :, :] = pred_prob_mean

        f.close()
        
        del(log_pred_prob_mean,pred_prob_mean, exp_log_pred_prob_mean, pred_logcounts_mean, chrom_lst,start_lst,end_lst,pred_logcounts_lst,pred_profs_lst,log_pred_prob_lst)
        
        gc.collect()
        
mean_predictions(args.prediction_h5s,f"{args.output_dir}/mean_predictions.h5")


if args.generate_bigwigs:
    if args.chrom_sizes:

        chrom_sizes = args.chrom_sizes

        with open(chrom_sizes) as f:
            gs = [x.strip().split('\t') for x in f]
        gs = [(x[0], int(x[1])) for x in gs]

        chr_to_idx = {}
        for i,x in enumerate(gs):
            chr_to_idx[x[0]] = i


        predictions = h5py.File(f"{args.output_dir}/mean_predictions.h5", 'r')

        pred_prof = predictions['pred_prof'][()]
        coords_chrom = predictions['coords_chrom'][()].astype("U8")
        coords_start = predictions['coords_start'][()]
        coords_end = predictions['coords_end'][()]

        regions = [[coords_chrom[i],coords_start[i],coords_end[i],(coords_start[i]+coords_end[i])//2]  for i in range(coords_chrom.shape[0])]
        predictions.close()


        # regions may not be sorted, so get their sorted order
        order_of_regs = sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))

        # regions may overlap but as we go in sorted order, the value from the entry whose summit is closest if choosen for each position in the overlapping region

        bw = pyBigWig.open(f"{args.output_dir}/mean_predictions_plus.bw", 'w')
        bw.addHeader(gs)
        cur_chr = ""
        cur_end = 0
        # if args.tqdm:
        #     from tqdm import tqdm
        #     iterator = tqdm(order_of_regs)
        # else:
        #     iterator = order_of_regs

        iterator = range(len(order_of_regs))

        for itr in iterator:
            # subset to chromosome (debugging)
            #if regions[i][0]!="chr12":
            #    continue
            i = order_of_regs[itr]
            i_chr, i_start, i_end, i_mid = regions[i]
            

            if i_chr != cur_chr: 
                cur_chr = i_chr
                cur_end = 0

            # bring current end to at least start of current region
            if cur_end < i_start:
                cur_end = i_start

            assert(regions[i][2]>=cur_end)

            # figure out where to stop for this region, get next region
            # which may partially overlap with this one
            next_end = i_end

            if itr+1 != len(order_of_regs):
                n = order_of_regs[itr+1]
                next_chr, next_start, _, next_mid = regions[n]

                if next_chr == i_chr and next_start < i_end:
                    # if next region overlaps with this, end between their midpoints
                    next_end = (i_mid+next_mid)//2
                    
            vals = pred_prof[i,:,0][cur_end - i_start:next_end - i_start]
            
            bw.addEntries([i_chr]*(next_end-cur_end), 
                           list(range(cur_end,next_end)), 
                           ends = list(range(cur_end+1, next_end+1)), 
                           values=[float(x) for x in vals])   
        bw.close()

        bw = pyBigWig.open(f"{args.output_dir}/mean_predictions_minus.bw", 'w')
        bw.addHeader(gs)
        cur_chr = ""
        cur_end = 0


        iterator = range(len(order_of_regs))

        for itr in iterator:
            # subset to chromosome (debugging)
            #if regions[i][0]!="chr12":
            #    continue
            i = order_of_regs[itr]
            i_chr, i_start, i_end, i_mid = regions[i]

            if i_chr != cur_chr: 
                cur_chr = i_chr
                cur_end = 0

            # bring current end to at least start of current region
            if cur_end < i_start:
                cur_end = i_start

            assert(regions[i][2]>=cur_end)

            # figure out where to stop for this region, get next region
            # which may partially overlap with this one
            next_end = i_end

            if itr+1 != len(order_of_regs):
                n = order_of_regs[itr+1]
                next_chr, next_start, _, next_mid = regions[n]

                if next_chr == i_chr and next_start < i_end:
                    # if next region overlaps with this, end between their midpoints
                    next_end = (i_mid+next_mid)//2

            vals = pred_prof[i,:,1][cur_end - i_start:next_end - i_start]
            bw.addEntries([i_chr]*(next_end-cur_end), 
                           list(range(cur_end,next_end)), 
                           ends = list(range(cur_end+1, next_end+1)), 
                           values=[float(x) for x in vals])   
            
        bw.close()
    else:
        print("chrom_sizes file. It is required to output the bigwigs")