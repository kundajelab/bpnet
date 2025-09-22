import argparse
import pyBigWig
import numpy as np
import h5py
import gzip
import hdf5plugin

def importance_hdf5_to_bigwig(hdf5_path: str, 
                              regions_path: str,
                              outfile: str,
                              outstats: str,
                              chrom_sizes: str,
                              debug_chr: str = None,
                              gzipped: bool = False) -> None:
    '''
    Convert importance scores in hdf5 format to bigwig
    
    Args:
    
        hdf5_path (str): required=True, path to HDF5 file f such that f['hyp_scores'] has (N x 4 x seqlen) shape with importance score * sequence so that at each f['input_seqs'] has 3 zeros and 1 non-zero value
        
        regions_path (str): required=True, 10 column BED file of length N which matches f['input_seqs'].shape[0]. The ith region in the imBED file corresponds to ith entry in importance matrix. If start=2nd col, summit=10th col, then the importance scores are for [start+summit-(seqlen/2):start+summit+(seqlen/2)]
        
        chrom_sizes (str): required=True, Chromosome sizes 2 column file
        
        outfile (str): required=True, Output bigwig file
        
        outstats (str): required=True, Output file with stats of low and high quantiles
        
        debug_chr (str): default=None, Run for one chromosome only (e.g. chr12) for debugging
        
        gzipped (bool): default=False, whether the peak file is gzipped
    
    Returns:
        None
    
    '''


    with open(chrom_sizes) as f:
        gs = [x.strip().split('\t') for x in f]
    gs = [(x[0], int(x[1])) for x in gs]

    chr_to_idx = {}
    for i,x in enumerate(gs):
        chr_to_idx[x[0]] = i


    scores = h5py.File(hdf5_path, 'r')

    shap_scores = scores['hyp_scores']
    one_hot_seqs = scores['input_seqs']

    print("Computing projected shap scores")
    proj_shap_scores = np.multiply(one_hot_seqs, shap_scores)
    print("Done computing projected shap scores")

    scores.close()

    proj_shap_scores = proj_shap_scores.transpose((0,2,1))

    SEQLEN = proj_shap_scores.shape[2]
    assert(SEQLEN%2==0)

    if gzipped:
        with gzip.open(regions_path) as f:
            regions = [x.decode('utf8').strip().split('\t') for x in f]
    else:
        with open(regions_path) as r:
            regions = [x.strip().split('\t') for x in r]

    regions = [[x[0], int(x[1])+int(x[9])-int(SEQLEN/2), int(x[1])+int(x[9])+int(SEQLEN/2), int(x[1])+int(x[9])] for x in regions]

    # regions may not be sorted, so get their sorted order
    order_of_regs = sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))

    # regions may overlap but as we go in sorted order, for regions that overlap values are repeated used
    # from the nearest peak 

    bw = pyBigWig.open(outfile, 'w')
    bw.addHeader(gs)
    all_entries = []
    cur_chr = ""
    cur_end = 0

    iterator = range(len(order_of_regs))
    
    for itr in iterator:
        # subset to chromosome (debugging)
        if debug_chr and regions[i][0]!=debug_chr:
            continue

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

        vals = np.sum(proj_shap_scores[i], axis=0)[cur_end - i_start:next_end - i_start]

        bw.addEntries([i_chr]*(next_end-cur_end), 
                       list(range(cur_end,next_end)), 
                       ends = list(range(cur_end+1, next_end+1)), 
                       values=[float(x) for x in vals])

        all_entries.append(vals)

        cur_end = next_end

    bw.close()

    all_entries = np.hstack(all_entries)
    with open(outstats, 'w') as f:
        f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
        f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
        f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
        f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
        f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
        f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
        f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
        f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
        f.write("Max\t{:.6f}\n".format(np.max(all_entries)))
