import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import modisco.visualization
from modisco.visualization import viz_sequence
import h5py

import numpy as np
import modisco
import modisco.backend
import modisco.nearest_neighbors
import modisco.affinitymat
import modisco.tfmodisco_workflow.seqlets_to_patterns
import modisco.tfmodisco_workflow.workflow
import modisco.aggregator
import modisco.cluster
import modisco.core
import modisco.coordproducers
import modisco.metaclusterers
import modisco.util

from bpnet.cli.argparsers import motif_discovery_argsparser
from bpnet.utils.exceptionhandler import NoTracebackException
from modisco.tfmodisco_workflow.seqlets_to_patterns \
    import TfModiscoSeqletsToPatternsFactory
from modisco.tfmodisco_workflow.workflow import TfModiscoWorkflow
from modisco.visualization import viz_sequence


import hdf5plugin

def save_plot(weights, dst_fname):
    """
    
    """
    print(dst_fname)
    colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
    plot_funcs = {0: viz_sequence.plot_a, 1: viz_sequence.plot_c, 
                  2: viz_sequence.plot_g, 3: viz_sequence.plot_t}

    fig = plt.figure(figsize=(20, 2))
    ax = fig.add_subplot(111) 
    viz_sequence.plot_weights_given_ax(ax=ax, array=weights, 
                                       height_padding_factor=0.2,
                                       length_padding=1.0, 
                                       subticks_frequency=1.0, 
                                       colors=colors, plot_funcs=plot_funcs, 
                                       highlight={}, ylabel="")

    plt.savefig(dst_fname)


def motif_discovery_main():
    parser = motif_discovery_argsparser()
    args = parser.parse_args()

    if not os.path.exists(args.scores_path):
        raise NoTracebackException(
            "Score file {} does not exist".format(args.scores_path))

    if not os.path.exists(args.output_directory):
        raise NoTracebackException(
            "Output directiry {} does not exist".format(args.output_directory))

    # Load the scores
    scores = h5py.File(args.scores_path, 'r')
    
    # window start and end based on modisco_window_size
    center = scores['hyp_scores'].shape[1] // 2
    start = center - args.modisco_window_size // 2
    end = center + args.modisco_window_size // 2
    
    print("Shap scores shape - {}".format(scores['hyp_scores'].shape))
    
    shap_scores = scores['hyp_scores'][:,start:end,:]
    one_hot_seqs = scores['input_seqs'][:,start:end,:]
    print("Done slicing shap scores and one hot seqs")
    
    proj_shap_scores = np.multiply(one_hot_seqs, shap_scores)
    print("Done computing projected shap scores")

    scores.close()
    
    tasks = ['task0']
    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()
    task_to_scores['task0']  = proj_shap_scores
    task_to_hyp_scores['task0']  = shap_scores

    tfmodisco_workflow = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        min_metacluster_size_frac=0.0001,
        max_seqlets_per_metacluster=args.max_seqlets,
        sliding_window_size=20,
        flank_size=5,
        target_seqlet_fdr=0.05, 
        seqlets_to_patterns_factory=\
        modisco.tfmodisco_workflow.seqlets_to_patterns
            .TfModiscoSeqletsToPatternsFactory(
                n_cores=10,n_cores_mainclustering=4,
                trim_to_window_size=20, 
                initial_flank_to_add=5, 
                final_min_cluster_size=20))

    tfmodisco_results = tfmodisco_workflow(
        task_names=["task0"], contrib_scores=task_to_scores, 
        hypothetical_contribs=task_to_hyp_scores, one_hot=one_hot_seqs)

    modisco_results_path = '{}/modisco_results.h5'.format(
        args.output_directory)
        
    tfmodisco_results.save_hdf5(h5py.File(modisco_results_path, 'w'))
    print("Saved modisco results to file {}".format(str(modisco_results_path)))
    

    seqlet_path = '{}/seqlets.txt'.format(args.output_directory)
    print("Saving seqlets to %s" % seqlet_path)
    seqlets = \
        tfmodisco_results.metacluster_idx_to_submetacluster_results[0].seqlets
    bases = np.array(["A", "C", "G", "T"])
    with open(seqlet_path, "w") as f:
        for seqlet in seqlets:
            sequence = "".join(
                bases[np.argmax(seqlet["sequence"].fwd, axis=-1)]
            )
            example_index = seqlet.coor.example_idx
            start, end = seqlet.coor.start, seqlet.coor.end
            f.write(">example%d:%d-%d\n" % (example_index, start, end))
            f.write(sequence + "\n")

    print("Saving pattern visualizations")

    patterns = (tfmodisco_results
                .metacluster_idx_to_submetacluster_results[0]
                .seqlets_to_patterns_result.patterns)

    # generate .pngs of each motif and write motif seqlet to
    # individual files
    for idx,pattern in enumerate(patterns):
        print(pattern)
        print("pattern idx",idx)
        print(len(pattern.seqlets))
        
        pattern_seqlet_path = os.path.join(args.output_directory,
                                           'pattern{}_seqlets.txt'.format(idx))
        with open(pattern_seqlet_path, "w") as f: 
            for seqlet in pattern.seqlets:
                sequence = "".join(
                    bases[np.argmax(seqlet["sequence"].fwd, axis=-1)]
                )
                example_index = seqlet.coor.example_idx
                start, end = seqlet.coor.start, seqlet.coor.end
                f.write(">example%d:%d-%d\n" % (example_index, start, end))
                f.write(sequence + "\n")

        save_plot(pattern["task0_contrib_scores"].fwd, 
                  '{}/contrib_{}.png'.format(args.output_directory, idx))
        save_plot(pattern["sequence"].fwd,
                  '{}/sequence_{}.png'.format(args.output_directory, idx))


if __name__ == '__main__':
    motif_discovery_main()
