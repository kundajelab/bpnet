import os
import subprocess
import numpy as np
import pandas as pd
import tempfile

BACKGROUND_FREQS = np.array([0.25, 0.25, 0.25, 0.25])
DATABASE_PATH = "/users/amtseng/tfmodisco/data/processed/motif_databases/HOCOMOCO_JASPAR_motifs.txt"

def import_database_pfms(database_path):
    """
    Imports the database of PFMs by reading through the entire database and
    constructing a dictionary mapping motif IDs to NumPy arrays of PFMs.
    """
    motif_dict = {}
    with open(database_path, "r") as f:
        try:
            while True:
                line = next(f)
                if line.startswith("MOTIF"):
                    key = line.strip().split()[1]
                    header = next(f)
                    motif_width = int(header.split()[5])
                    motif = np.empty((motif_width, 4))
                    for i in range(motif_width):
                        motif[i] = np.array([
                            float(x) for x in next(f).strip().split()
                        ])
                    motif_dict[key] = motif
        except StopIteration:
            pass
    return motif_dict


def export_pfms_to_meme_format(
    pfms, outfile, background_freqs=None, names=None
):
    """
    Exports a set of PFMs to MEME motif format. Includes the background
    frequencies `BACKGROUND_FREQS`.
    Arguments:
        `pfms`: a list of L x 4 PFMs (where L can be different for each PFM)
        `outfile`: path to file to output the MEME-format PFMs
        `background_freqs`: background frequencies of A, C, G, T as a length-4
            NumPy array; defaults to `BACKGROUND_FREQS`
        `names`: if specified, a list of unique names to give to each PFM, must
            be parallel to `pfms`
    """
    if names is None:
        names = [str(i) for i in range(len(pfms))]
    else:
        assert len(names) == pfms
        assert len(names) == len(np.unique(names))
    if background_freqs is None:
        background_freqs = BACKGROUND_FREQS

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        f.write("MEME version 5\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("Background letter frequencies\n")
        f.write("A %f C %f G %f T %f\n\n" % tuple(background_freqs))
        for i in range(len(pfms)):
            pfm, name = pfms[i], names[i]
            f.write("MOTIF %s\n" % name)
            f.write("letter-probability matrix:\n")
            for row in pfm:
                f.write(" ".join([str(freq) for freq in row]) + "\n")
            f.write("\n")


def run_tomtom(
    query_motif_file, target_motif_file, outdir, show_output=True
):
    """
    Runs TOMTOM given the target and query motif files. The default threshold
    of q < 0.5 is used to filter for matches.
    Arguments:
        `query_motif_file`: file containing motifs in MEME format, which will
            be the query motifs for which matches are found
        `target_motif_file`: file containing motifs in MEME format, which will
            be used to search for matches
        `outdir`: path to directory to store results
        `show_output`: whether or not to show TOMTOM output
    """
    comm = ["tomtom"]
    comm += [query_motif_file, target_motif_file]
    comm += ["-oc", outdir]
    comm += ["-no-ssc"]
    comm += ["-dist", "pearson"]
    comm += ["-min-overlap", "5"]
    proc = subprocess.run(comm, capture_output=(not show_output))


def import_tomtom_results(tomtom_dir):
    """
    Imports the TOMTOM output directory as a Pandas DataFrame.
    Arguments:
        `tomtom_dir`: TOMTOM output directory, which contains the output file
            "tomtom.tsv"
    Returns a Pandas DataFrame.
    """
    return pd.read_csv(
        os.path.join(tomtom_dir, "tomtom.tsv"), sep="\t", header=0,
        index_col=False, comment="#"
    )


def match_motifs_to_targets(
    query_pfms, target_pfms, temp_dir=None, show_tomtom_output=False
):
    """
    For each motif in the query PFMs, finds the best match to the target PFMs,
    based on TOMTOM q-value.
    Arguments:
        `query_pfms`: list of L x 4 PFMs to look for matches for
        `target_pfms`: list of L x 4 PFMs to match to
        `temp_dir`: a temporary directory to store intermediates; defaults to
            a randomly created directory
        `show_tomtom_output`: whether to show TOMTOM output when running
    Returns an array of indices parallel to `query_pfms`, where each index is
    denotes the best PFM within `target_pfms` that matches the query PFM. If
    a good match is not found (i.e. based on TOMTOM's threshold), the index will
    be -1.
    """
    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
    else:
        temp_dir_obj = None

    # Convert motifs to MEME format
    query_motif_file = os.path.join(temp_dir, "query_motifs.txt")
    target_motif_file = os.path.join(temp_dir, "target_motifs.txt")
    export_pfms_to_meme_format(query_pfms, query_motif_file)
    export_pfms_to_meme_format(target_pfms, target_motif_file)

    # Run TOMTOM
    tomtom_dir = os.path.join(temp_dir, "tomtom")
    run_tomtom(
        query_motif_file, target_motif_file, tomtom_dir,
        show_output=show_tomtom_output
    )

    # Find results, mapping each query motif to target index
    # The query/target IDs are the indices
    tomtom_table = import_tomtom_results(tomtom_dir)
    match_inds = []
    for i in range(len(query_pfms)):
        rows = tomtom_table[tomtom_table["Query_ID"] == i]
        if rows.empty:
            match_inds.append(-1)
            continue
        target_id = rows.loc[rows["q-value"].idxmin()]["Target_ID"]
        match_inds.append(target_id)

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return np.array(match_inds)
        

def match_motifs_to_database(
    query_pfms, top_k=5, temp_dir=None, database_path=DATABASE_PATH,
    show_tomtom_output=False
):
    """
    For each motif in the query PFMs, finds the best matches to the TOMTOM
    database, ranked by TOMTOM q-value.
    Arguments:
        `query_pfms`: list of L x 4 PFMs to look for matches for
        `top_k`: the number of motifs to return based on q-value
        `temp_dir`: a temporary directory to store intermediates; defaults to
            a randomly created directory
        `database_path`: the path to a TOMTOM motif database; defaults to
            DATABASE_PATH
        `show_tomtom_output`: whether to show TOMTOM output when running
    Returns a list of lists of (motif name, motif PFM, q-value) tuples
    parallel to `query_pfms`, where each sublist of tuples is the set of motif
    names, motif PFMs (as NumPy arrays), and q-values for the corresponding
    query motif. Each sublit is sorted in ascending order by q-value. If fewer
    than `top_k` matches are found (based on TOMTOM's threshold), the returned
    sublist will be shorter (and may even be empty).
    """
    # First, import the database PFMs
    database_pfms = import_database_pfms(database_path)

    if temp_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
    else:
        temp_dir_obj = None

    # Convert motifs to MEME format
    query_motif_file = os.path.join(temp_dir, "query_motifs.txt")
    export_pfms_to_meme_format(query_pfms, query_motif_file)

    # Run TOMTOM
    tomtom_dir = os.path.join(temp_dir, "tomtom")
    run_tomtom(
        query_motif_file, database_path, tomtom_dir,
        show_output=show_tomtom_output
    )

    # Find results, mapping each query motif to target index
    # The query/target IDs are the indices
    tomtom_table = import_tomtom_results(tomtom_dir)
    matches = []
    for i in range(len(query_pfms)):
        rows = tomtom_table[tomtom_table["Query_ID"] == i]
        if rows.empty:
            matches.append([])
            continue
        rows = rows.sort_values("q-value").head(top_k)
        tups = list(zip(rows["Target_ID"], rows["q-value"]))
        tups = [
            (tup[0], database_pfms[tup[0]], tup[1]) for tup in tups
        ]
        matches.append(tups)

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    return matches
