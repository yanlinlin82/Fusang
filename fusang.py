import argparse
import datetime
import json
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import warnings
from io import StringIO
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from Bio import AlignIO
from ete3 import Tree, PhyloTree

# Parse arguments early to set logging level before TensorFlow import
_parser = argparse.ArgumentParser('get_msa_dir', add_help=False)
_parser.add_argument('-q', '--quiet', action='store_true', help='Suppress warning messages')
_parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output including stderr messages')
_early_args, _ = _parser.parse_known_args()

# Constants
WINDOW_SIZE_SHORT = 240
WINDOW_SIZE_LONG = 1200
MSA_LENGTH_THRESHOLD = 1210
BATCH_SIZE = 50000
MIN_TAXA_FOR_MASKING = 10
MIN_EDGES_FOR_MASKING = 3
MAX_PROCESSES = 64
MIN_PROCESSES = 6
MAX_MSA_LENGTH = 10000
SEQUENCE_PADDING = 14000
EPSILON = 1e-200


class StderrRedirector:
    """
    Context manager for redirecting stderr to suppress output.
    Handles both Python-level and file descriptor level redirection.
    Supports idempotent enter/exit (safe to call multiple times).
    """

    def __init__(self, redirect=True):
        """
        Initialize the stderr redirector.

        Args:
            redirect: If True, redirect stderr; if False, do nothing
        """
        self.redirect = redirect
        self.original_stderr = None
        self.stderr_fd = None
        self.original_stderr_fd = None
        self._enter_count = 0  # Track nesting level

    def __enter__(self):
        """Enter the context and redirect stderr if needed."""
        if not self.redirect:
            return self

        self._enter_count += 1
        # Only redirect on first enter
        if self._enter_count == 1:
            # Redirect at Python level
            self.original_stderr = sys.stderr
            sys.stderr = StringIO()

            # Also redirect at file descriptor level to catch C++ messages
            try:
                self.original_stderr_fd = os.dup(2)  # Save original stderr fd
                self.stderr_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(self.stderr_fd, 2)  # Redirect stderr fd to /dev/null
            except (OSError, AttributeError):
                # If file descriptor redirection fails, continue with Python-level redirection
                self.stderr_fd = None
                self.original_stderr_fd = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore stderr."""
        if not self.redirect or self._enter_count == 0:
            return False

        self._enter_count -= 1
        # Only restore on last exit
        if self._enter_count == 0:
            # Restore Python-level stderr
            if self.original_stderr is not None:
                sys.stderr = self.original_stderr
                self.original_stderr = None

            # Restore file descriptor level redirection
            if self.original_stderr_fd is not None:
                try:
                    os.dup2(self.original_stderr_fd, 2)
                    os.close(self.original_stderr_fd)
                    if self.stderr_fd is not None:
                        os.close(self.stderr_fd)
                except (OSError, AttributeError):
                    pass
                finally:
                    self.original_stderr_fd = None
                    self.stderr_fd = None

        return False  # Don't suppress exceptions


# Set TensorFlow and NumPy warning levels based on verbosity (before TensorFlow import)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create stderr redirector based on early args
_stderr_redirector = StderrRedirector(redirect=_early_args.quiet)

if _early_args.quiet:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all messages including ERROR
    warnings.filterwarnings('ignore')
    # Enter the redirector context early (before TensorFlow import)
    # This will be properly exited in main() using 'with' statement
    _stderr_redirector.__enter__()
elif _early_args.verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Show all messages
    warnings.filterwarnings('default')
else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages (default)
    warnings.filterwarnings('ignore', category=FutureWarning)

# TensorFlow and Keras are imported lazily when needed (see _import_tensorflow function)


def _print_stderr(message='', end='\n'):
    """Write a status line to stderr without polluting stdout."""
    print(message, end=end, flush=True, file=sys.stderr)


# ============================================================================
# Core Utility Functions
# ============================================================================

def comb_math(n, m):
    """Calculate combination C(n, m) = n! / (m! * (n-m)!)."""
    return math.factorial(n) // (math.factorial(n - m) * math.factorial(m))


def nlargest_indices(arr, n):
    uniques = np.unique(arr)
    threshold = uniques[-n]
    return np.where(arr >= threshold)


# ============================================================================
# Quartet and Tree Operations
# ============================================================================

def get_quartet_ID(quartet):
    return "".join(sorted(quartet))


def tree_from_quartet(quartet):
    root = Tree()
    root.name = "internal_node_0"
    left = root.add_child(name="internal_node_1")
    left.add_child(name=quartet[0])
    left.add_child(name=quartet[1])
    right = root.add_child(name="internal_node_2")
    right.add_child(name=quartet[2])
    right.add_child(name=quartet[3])
    for desc in root.iter_descendants():
        desc.dist = 0
    return root


def get_topology_ID(quartet):
    return get_quartet_ID(quartet[0:2]) + get_quartet_ID(quartet[2:4])


def get_current_topology_id(quart_key, cluster_1, cluster_2):
    """Determine topology ID based on quartet key and cluster positions."""
    a1 = quart_key.index(cluster_1)
    a2 = quart_key.index(cluster_2)
    ans = {str(a1), str(a2)}
    if ans == {'0', '1'} or ans == {'2', '3'}:
        return 0
    elif ans == {'0', '2'} or ans == {'1', '3'}:
        return 1
    elif ans == {'0', '3'} or ans == {'1', '2'}:
        return 2
    else:
        _print_stderr('Error of function get_current_topology_id, exit the program')
        sys.exit(1)


def judge_tree_score(tree, quart_distribution, new_addition_taxa, dic_for_leave_node_comb_name):
    """
    Calculate tree score based on quartet distribution.

    Parameters:
    tree: a candidate tree, can be any taxas
    quart_distribution: the prob distribution of the topology of every 4-taxa
    new_addition_taxa: newly added taxon name
    dic_for_leave_node_comb_name: dictionary mapping quartet keys to indices
    """
    crt_tree = tree.copy("newick")
    leaves = [ele.name for ele in crt_tree.get_leaves()]
    total_quarts = list(combinations(leaves, 4))
    quarts = [ele for ele in total_quarts if new_addition_taxa in ele]

    total_quart_score = 0

    for quart in quarts:
        crt_tree = tree.copy("newick")
        try:
            crt_tree.prune(list(quart))
        except Exception:
            _print_stderr('Error of pruning 4 taxa from current tree, the current tree is:')
            _print_stderr(str(crt_tree))
            sys.exit(1)

        quart_key = "".join(sorted(list(quart)))
        quart_topo_id = dic_for_leave_node_comb_name[quart_key]
        quart_topo_distribution = quart_distribution[quart_topo_id]

        # judge current tree belongs to which topology
        tmp = re.findall(r"\([\s\S]\,[\s\S]\)", crt_tree.write(format=9))[0]
        topology_id = get_current_topology_id(quart_key, tmp[1], tmp[3])

        total_quart_score += np.log(quart_topo_distribution[topology_id] + EPSILON)

    return total_quart_score


def get_modify_tree(tmp_tree, edge_0, edge_1, new_add_node_name):
    """
    Add a new leave node between edge_0 and edge_1.
    Default: edge_0 is the parent node of edge_1.
    """
    modify_tree = tmp_tree.copy("newick")
    if edge_0 != edge_1:
        new_node = Tree()
        new_node.add_child(name=new_add_node_name)
        detached_node = modify_tree & edge_1
        detached_node.detach()
        inserted_node = modify_tree & edge_0
        inserted_node.add_child(new_node)
        new_node.add_child(detached_node)
    else:
        modify_tree.add_child(name=new_add_node_name)

    return modify_tree


def search_this_branch(tmp_tree, edge_0, edge_1, current_quartets, current_leave_node_name, queue, dic_for_leave_node_comb_name):
    modify_tree = get_modify_tree(tmp_tree, edge_0, edge_1, current_leave_node_name)
    modify_tree.resolve_polytomy(recursive=True)
    modify_tree.unroot()
    tmp_tree_score = judge_tree_score(modify_tree, current_quartets, current_leave_node_name, dic_for_leave_node_comb_name)

    dic = {}
    dic['tree'] = modify_tree
    dic['score'] = tmp_tree_score
    queue.put(dic)


# ============================================================================
# Tree Search and Masking Functions
# ============================================================================

def select_mask_node_pair(dl_predict, new_add_taxa, start_end_list, comb_of_id):
    """Select node pairs to mask based on prediction confidence."""
    if new_add_taxa <= MIN_TAXA_FOR_MASKING - 1:
        return None

    mask_node_pair = []
    current_start = start_end_list[new_add_taxa][0]
    current_end = start_end_list[new_add_taxa][1]
    select_distribution = dl_predict[current_start:current_end + 1]
    if np.max(select_distribution) < 0.90:
        return None

    x, y = nlargest_indices(select_distribution, int(max(10, 0.01 * len(select_distribution))))

    for i in range(len(x)):
        idx = x[i]
        topology_value = y[i]
        quartet_comb = comb_of_id[current_start + idx]

        if topology_value == 0:
            mask_node_pair.append((quartet_comb[0], quartet_comb[1]))
        elif topology_value == 1:
            mask_node_pair.append((quartet_comb[0], quartet_comb[2]))
        elif topology_value == 2:
            mask_node_pair.append((quartet_comb[1], quartet_comb[2]))

    return mask_node_pair


def mask_edge(tree, node1, node2, edge_list):
    """Mask edge between node1 and node2."""
    if len(edge_list) <= MIN_EDGES_FOR_MASKING:
        return edge_list

    ancestor_name = tree.get_common_ancestor(node1, node2).name
    remove_edge = []

    for node_name in [node1, node2]:
        node = tree.search_nodes(name=node_name)[0]
        while node:
            if node.name == ancestor_name:
                break

            edge_0 = node.up.name
            edge_1 = node.name

            if len(remove_edge) >= len(edge_list) - MIN_EDGES_FOR_MASKING:
                break
            remove_edge.append((edge_0, edge_1))
            node = node.up

    for ele in remove_edge:
        if ele in edge_list:
            edge_list.remove(ele)

    return edge_list


def gen_phylogenetic_tree(current_quartets, beam_size, taxa_num, leave_node_comb_name,
                          dic_for_leave_node_comb_name, internal_node_name_pool,
                          use_masking=False, start_end_list=None, comb_of_id=None):
    """
    Search the phylogenetic tree having highest score using beam search.

    Parameters:
    current_quartets: quartet distribution predictions
    beam_size: size of beam for search
    taxa_num: number of taxa
    leave_node_comb_name: list of quartet combinations
    dic_for_leave_node_comb_name: dictionary mapping quartet keys to indices
    internal_node_name_pool: pool of names for internal nodes
    use_masking: whether to use edge masking optimization
    start_end_list: list of start/end indices for masking
    comb_of_id: list of quartet ID combinations
    """
    current_leave_node_name = [chr(ord(u'\u4e00') + i) for i in range(taxa_num)]
    candidate_tree_beam = []
    quartet_id = leave_node_comb_name[0]

    # Initialize with three possible topologies for first quartet
    for _label in [0, 1, 2]:
        if _label == 0:
            label_id = "".join([quartet_id[0], quartet_id[1], quartet_id[2], quartet_id[3]])
        elif _label == 1:
            label_id = "".join([quartet_id[0], quartet_id[2], quartet_id[1], quartet_id[3]])
        elif _label == 2:
            label_id = "".join([quartet_id[0], quartet_id[3], quartet_id[1], quartet_id[2]])

        _tree = tree_from_quartet(label_id)
        _tree.unroot()
        _tree_score = current_quartets[0, _label]
        tmp_tree_dict = {'Tree': _tree, 'tree_score': _tree_score}
        candidate_tree_beam.append(tmp_tree_dict)
        candidate_tree_beam.sort(key=lambda k: -k['tree_score'])

    idx_for_internal_node_name_pool = 0
    current_tree_score_beam = []
    optim_tree_beam = []

    # In the start point set beam size equal to 3
    for i in range(3):
        current_tree_score_beam.append(candidate_tree_beam[i]['tree_score'])
        optim_tree_beam.append(candidate_tree_beam[i]['Tree'])

    # Add remaining taxa one by one
    for i in range(4, len(current_leave_node_name)):
        candidate_tree_beam = []

        for j in range(len(optim_tree_beam)):
            ele = optim_tree_beam[j]
            if ele is None:
                continue

            optim_tree = ele.copy("newick")
            edge_0_list = []
            edge_1_list = []

            # Collect all edges
            for node in optim_tree.iter_descendants():
                edge_0 = node.up.name
                edge_1 = node.name
                if edge_0 == '' or edge_1 == '':
                    continue
                edge_0_list.append(edge_0)
                edge_1_list.append(edge_1)

            # Apply masking if enabled
            if use_masking and start_end_list is not None and comb_of_id is not None:
                edge_list = [(edge_0_list[k], edge_1_list[k]) for k in range(len(edge_0_list))]
                mask_node_pairs = select_mask_node_pair(current_quartets, i, start_end_list, comb_of_id)

                if mask_node_pairs is not None:
                    mask_node_pairs = list(set(mask_node_pairs))
                    for node_pairs in mask_node_pairs:
                        node1 = chr(ord(u'\u4e00') + node_pairs[0])
                        node2 = chr(ord(u'\u4e00') + node_pairs[1])
                        edge_list = mask_edge(ele.copy("deepcopy"), node1, node2, edge_list)
                        if len(edge_list) <= MIN_EDGES_FOR_MASKING:
                            break

                edge_0_list = [ele[0] for ele in edge_list]
                edge_1_list = [ele[1] for ele in edge_list]

            # Search all branches in parallel
            queue = multiprocessing.Manager().Queue()
            para_list = [(optim_tree, edge_0_list[k], edge_1_list[k], current_quartets,
                         current_leave_node_name[i], queue, dic_for_leave_node_comb_name)
                        for k in range(len(edge_0_list))]

            if use_masking:
                process_num = min(MAX_PROCESSES, len(edge_0_list))
            else:
                process_num = min(MAX_PROCESSES, 2 * i + MIN_PROCESSES)
            pool = Pool(process_num)
            pool.starmap(search_this_branch, para_list)
            pool.close()
            pool.join()

            # Collect results
            while not queue.empty():
                tmp_dic = queue.get()
                tmp_tree_dict = {'Tree': tmp_dic['tree'],
                               'tree_score': tmp_dic['score'] + current_tree_score_beam[j]}
                candidate_tree_beam.append(tmp_tree_dict)

        # Keep top beam_size candidates
        candidate_tree_beam.sort(key=lambda k: -k['tree_score'])
        candidate_tree_beam = candidate_tree_beam[0:beam_size]

        # Update beam for next iteration
        optim_tree_beam = []
        current_tree_score_beam = []
        for ele in candidate_tree_beam:
            crt_tree = ele['Tree'].copy("newick")
            for node in crt_tree.traverse("preorder"):
                if node.name == '':
                    node.name = str(internal_node_name_pool[idx_for_internal_node_name_pool])
                    idx_for_internal_node_name_pool += 1
            optim_tree_beam.append(crt_tree)
            current_tree_score_beam.append(ele['tree_score'])

    return optim_tree_beam[0].write(format=9)


def transform_str(str_a, taxa_name, taxa_num):
    """Transform internal taxon IDs to actual taxon names."""
    id_to_name = {chr(ord(u'\u4e00') + i): taxa_name[i] for i in range(taxa_num)}
    return ''.join(id_to_name.get(char, char) for char in str_a)


# ============================================================================
# MSA Processing Functions
# ============================================================================

def get_numpy(aln_file):
    '''
    current version only supports the total length of msa less than 10K
    '''
    # Handle both file paths and file-like objects (e.g., StringIO)
    if isinstance(aln_file, str):
        # Use context manager for automatic file closing
        with open(aln_file) as aln:
            return _process_alignment(aln)
    else:
        # File-like object (e.g., StringIO), no need to close
        return _process_alignment(aln_file)


def _process_alignment(aln):
    """Process alignment file-like object and return numpy array."""
    dic = {'A': '0', 'T': '1', 'C': '2', 'G': '3', '-': '4', 'N': '4'}

    # For masking other unknown bases
    other_base = ['R', 'Y', 'K', 'M', 'U', 'S', 'W', 'B', 'D', 'H', 'V', 'X']
    for ele in other_base:
        dic[ele] = '4'

    fasta_dic = {}
    for line in aln:
        if line[0] == ">":
            header = line[1:].rstrip('\n').strip()
            fasta_dic[header] = []
        elif line[0].isalpha() or line[0] == '-':
            processed_line = line[:].rstrip('\n').strip().upper()
            for base, num in dic.items():
                processed_line = processed_line.replace(base, num)
            line_list = [int(n) for n in list(processed_line)]
            fasta_dic[header] += line_list + [4] * (SEQUENCE_PADDING - len(line_list))

    taxa_block = []
    for taxa in sorted(fasta_dic.keys(), key=lambda x: int(x.strip())):
        taxa_block.append(fasta_dic[taxa.strip()])

    return np.array([taxa_block])


# ============================================================================
# Deep Learning Model Definitions
# ============================================================================

def _import_tensorflow():
    """Lazy import TensorFlow and Keras to avoid slow startup when not needed."""
    global tf, layers, models, optimizers
    if 'tf' not in globals():
        _print_stderr("Initializing TensorFlow (this may take a moment)...")
        import tensorflow as tf
        _print_stderr("TensorFlow initialized. Importing Keras...")
        from keras import layers, models, optimizers
        globals()['tf'] = tf
        globals()['layers'] = layers
        globals()['models'] = models
        globals()['optimizers'] = optimizers
        _print_stderr("TensorFlow and Keras ready.", end='\n')
    return tf, layers, models, optimizers


def get_dl_model(window_size):
    """
    Get the definition of DL model for the given window size.
    Supported window sizes: 240 and 1200.
    """
    if window_size not in (WINDOW_SIZE_SHORT, WINDOW_SIZE_LONG):
        raise ValueError(f"Unsupported window size: {window_size}. Must be {WINDOW_SIZE_SHORT} or {WINDOW_SIZE_LONG}.")

    _print_stderr("Importing TensorFlow...")
    _, layers, models, _ = _import_tensorflow()
    _print_stderr(f"TensorFlow imported. Creating model architecture ({window_size})...")

    conv_x = [4, 1, 1, 1, 1, 1, 1, 1]
    conv_y = [1, 2, 2, 2, 2, 2, 2, 2]
    filter_s = [1024, 1024, 128, 128, 128, 128, 128, 128]

    if window_size == WINDOW_SIZE_SHORT:
        pool = [1, 2, 2, 2, 2, 2, 2, 2]
    else:
        pool = [1, 4, 4, 4, 2, 2, 2, 1]

    visible = layers.Input(shape=(4, window_size, 1))
    x = visible

    for l in range(8):
        x = layers.ZeroPadding2D(padding=((0, 0), (0, conv_y[l] - 1)))(x)
        x = layers.Conv2D(filters=filter_s[l], kernel_size=(conv_x[l], conv_y[l]), strides=1, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.AveragePooling2D(pool_size=(1, pool[l]))(x)

    flat = layers.Flatten()(x)

    y = layers.Reshape((4, window_size))(visible)
    y = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(y)
    y = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(y)
    y = layers.Bidirectional(layers.LSTM(128))(y)
    flat = layers.concatenate([flat, y], axis=-1)

    hidden1 = layers.Dense(1024, activation='relu')(flat)
    drop1 = layers.Dropout(rate=0.2)(hidden1)
    output = layers.Dense(3, activation='softmax')(drop1)
    model = models.Model(inputs=visible, outputs=output)

    return model


def fill_dl_predict_each_slide_window(len_idx_1, len_idx_2, window_size, comb_of_id, org_seq,
                                      dl_model, dl_predict, verbose=0):
    """Process predictions for a single sliding window."""
    iters = len(comb_of_id) // BATCH_SIZE
    for i in range(iters):
        batch_seq = np.zeros((BATCH_SIZE, 4, window_size))

        for j in range(BATCH_SIZE):
            idx = np.array(comb_of_id[i * BATCH_SIZE + j])
            batch_seq[j] = org_seq[0, idx[:], len_idx_1:len_idx_2]

        tf, _, _, _ = _import_tensorflow()
        test_seq = tf.expand_dims(batch_seq.astype(np.float32), axis=-1)
        predicted = dl_model.predict(x=test_seq, verbose=verbose)

        for j in range(BATCH_SIZE):
            dl_predict[i * BATCH_SIZE + j, :] += predicted[j, :]

    # Process remaining items
    last_batch_size = len(comb_of_id) % BATCH_SIZE
    if last_batch_size > 0:
        batch_seq = np.zeros((last_batch_size, 4, window_size))

        for j in range(last_batch_size):
            idx = np.array(comb_of_id[iters * BATCH_SIZE + j])
            batch_seq[j] = org_seq[0, idx[:], len_idx_1:len_idx_2]

        tf, _, _, _ = _import_tensorflow()
        test_seq = tf.expand_dims(batch_seq.astype(np.float32), axis=-1)
        predicted = dl_model.predict(x=test_seq, verbose=verbose)

        for j in range(last_batch_size):
            dl_predict[iters * BATCH_SIZE + j, :] += predicted[j, :]


def fill_dl_predict(window_number, window_size, len_of_msa, comb_of_id, org_seq,
                    dl_model, dl_predict, verbose=0):
    """Fill predictions using sliding windows."""
    if len_of_msa > window_size:
        step = (len_of_msa - window_size) // window_number
    else:
        # For very short sequences, use a minimal step
        step = max(1, (window_size + 10 - window_size) // window_number)

    start_idx = 0
    for i in range(window_number):
        end_idx = start_idx + window_size
        fill_dl_predict_each_slide_window(start_idx, end_idx, window_size, comb_of_id,
                                         org_seq, dl_model, dl_predict, verbose=verbose)
        start_idx += step


def parse_msa_file(msa_file):
    """Parse MSA file and return alignment data and taxa names."""
    support_format = ['.fas', '.phy', '.fasta', '.phylip']
    bio_format = ['fasta', 'phylip', 'fasta', 'phylip']

    taxa_name = {}
    for i in range(len(support_format)):
        if msa_file.endswith(support_format[i]):
            try:
                with open(msa_file, 'r') as f:
                    alignment = AlignIO.read(f, bio_format[i])
                len_of_msa = len(alignment[0].seq)
                taxa_num = len(alignment)

                # Create in-memory file-like object instead of writing to disk
                save_alignment = StringIO()
                for record in alignment:
                    taxa_name[len(taxa_name)] = record.id
                    save_alignment.write('>' + str(len(taxa_name) - 1) + '\n')
                    save_alignment.write(str(record.seq) + '\n')
                save_alignment.seek(0)  # Reset to beginning for reading
                return save_alignment, taxa_name, len_of_msa, taxa_num
            except Exception:
                _print_stderr('Something wrong about your msa file, please check your msa file')
                sys.exit(1)

    _print_stderr('we do not support this format of msa')
    sys.exit(1)


def initialize_quartet_data(taxa_num):
    """Initialize quartet combinations and related data structures."""
    start_end_list = [None, None, None]
    end = -1
    for i in range(3, 100):
        start = end + 1
        end = start + int(comb_math(i, 3)) - 1
        start_end_list.append((start, end))

    id_for_taxa = list(range(taxa_num))
    comb_of_id = list(combinations(id_for_taxa, 4))
    comb_of_id.sort(key=lambda ele: ele[-1])

    leave_node_name = [chr(ord(u'\u4e00') + i) for i in range(taxa_num)]
    leave_node_comb_name = []
    dic_for_leave_node_comb_name = {}

    for ele in comb_of_id:
        term = [chr(ord(u'\u4e00') + id) for id in ele]
        quartet_key = "".join(term)
        dic_for_leave_node_comb_name[quartet_key] = len(dic_for_leave_node_comb_name)
        leave_node_comb_name.append(quartet_key)

    internal_node_name_pool = ['internal_node_' + str(i) for i in range(3, 3000)]

    return (start_end_list, comb_of_id, leave_node_name, leave_node_comb_name,
            dic_for_leave_node_comb_name, internal_node_name_pool)


def write_output(searched_tree, output_path=None):
    """Write output tree to file or stdout."""
    if output_path is not None:
        output_file = Path(output_path)
    else:
        print(searched_tree)
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'a') as f:
        f.write(searched_tree + '\n')


def load_dl_model(len_of_msa, sequence_type, branch_model):
    """
    Load deep learning model based on MSA length and parameters.
    Returns the loaded model and window size (240 or 1200).
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent

    # Map parameters to model directory codes
    seq_type_map = {'standard': 'S', 'coding': 'C', 'noncoding': 'N'}
    branch_model_map = {'gamma': 'G', 'uniform': 'U'}

    # Select model based on MSA length
    if len_of_msa <= MSA_LENGTH_THRESHOLD:
        model_num = '1'
        window_size = WINDOW_SIZE_SHORT
    else:
        model_num = '2'
        window_size = WINDOW_SIZE_LONG

    dl_model = get_dl_model(window_size)

    # Build and load model path: model/{S|C|N}{1|2}{G|U}.h5
    seq_prefix = seq_type_map.get(sequence_type, 'S')
    branch_suffix = branch_model_map.get(branch_model, 'G')
    model_filename = f'{seq_prefix}{model_num}{branch_suffix}.h5'
    model_path = script_dir / 'model' / model_filename
    dl_model.load_weights(filepath=str(model_path))

    return dl_model, window_size


# ============================================================================
# Simulation Data Generation Functions
# ============================================================================

def _get_extremes(tree):
    """Get the nearest and farthest nodes from root."""
    longest_distance = float('-inf')
    shortest_distance = float('+inf')
    nearest = None
    farthest = None
    for node in tree.get_leaves():
        distance = node.get_distance(tree)
        if distance > longest_distance:
            longest_distance = distance
            farthest = node
        if distance < shortest_distance:
            shortest_distance = distance
            nearest = node
    return (nearest, farthest), (shortest_distance, longest_distance)


def _find_lba_branches(tree):
    """Find long branch attraction branches."""
    min_branch_ratio = float('+inf')
    leaves = []
    for node in tree.traverse('preorder'):
        if node.is_leaf() or node.is_root():
            continue
        t = tree.copy('newick')
        t.set_outgroup(t & node.name)
        if t.children[0].is_leaf() or t.children[1].is_leaf():
            continue
        (short1, long1), (sdis1, ldis1) = _get_extremes(t.children[0])
        if short1 is None or long1 is None:
            continue
        (short2, long2), (sdis2, ldis2) = _get_extremes(t.children[1])
        if short2 is None or long2 is None:
            continue
        internal_distance = t.children[0].dist + t.children[1].dist
        branch_ratio = ((internal_distance + max(sdis1, sdis2)) / min(ldis1, ldis2))
        if branch_ratio < min_branch_ratio:
            leaves = [short1, long1, short2, long2]
            min_branch_ratio = branch_ratio
    leaves = [tree & leaf.name for leaf in leaves]
    return min_branch_ratio, leaves


def _gen_newick(args):
    """Generate a single Newick tree topology."""
    q, seed, taxa_num, range_of_taxa_num, distribution_of_internal_branch_length, \
        distribution_of_external_branch_length, range_of_mean_pairwise_divergence = args

    random.seed(seed)
    np.random.seed(seed)

    taxon_count_model = scipy.stats.uniform(range_of_taxa_num[0], range_of_taxa_num[1])
    tree = PhyloTree()
    raw_taxa_count = int(round(taxon_count_model.rvs()))
    taxa_count = max(taxa_num, raw_taxa_count, 2)
    tree.populate(taxa_count, random_branches=True)
    current_internal_index = 0
    current_leaf_index = 0

    if distribution_of_internal_branch_length[0] == 1:
        internal_branch_model = scipy.stats.gamma(
            a=distribution_of_internal_branch_length[1],
            scale=distribution_of_internal_branch_length[2]
        )
    elif distribution_of_internal_branch_length[0] == 0:
        internal_branch_model = scipy.stats.uniform(
            distribution_of_internal_branch_length[1],
            distribution_of_internal_branch_length[2]
        )

    if distribution_of_external_branch_length[0] == 1:
        external_branch_model = scipy.stats.gamma(
            a=distribution_of_external_branch_length[1],
            scale=distribution_of_external_branch_length[2]
        )
    elif distribution_of_external_branch_length[0] == 0:
        external_branch_model = scipy.stats.uniform(
            distribution_of_external_branch_length[1],
            distribution_of_external_branch_length[2]
        )

    expected_mean_pairwise_divergence = scipy.stats.uniform(
        range_of_mean_pairwise_divergence[0],
        range_of_mean_pairwise_divergence[1]
    ).rvs()

    for node in tree.traverse("preorder"):
        if node.is_leaf():
            node.name = f"taxon{current_leaf_index:d}"
            node.dist = external_branch_model.rvs()
            current_leaf_index += 1
        else:
            node.name = f"node{current_internal_index:d}"
            node.dist = internal_branch_model.rvs()
            current_internal_index += 1

    total_tree_leaves = [ele.name for ele in tree.get_leaves()]

    pairwise_divergence_list = []
    for i in range(0, len(total_tree_leaves) - 1):
        for j in range(i + 1, len(total_tree_leaves)):
            tmp_dist = tree.get_distance(total_tree_leaves[i], total_tree_leaves[j])
            if tmp_dist > 1e-4:
                pairwise_divergence_list.append(tmp_dist)

    mean_pairwise_divergence = np.mean(np.array(pairwise_divergence_list))

    scale_ratio = expected_mean_pairwise_divergence / mean_pairwise_divergence

    for node in tree.traverse("preorder"):
        node.dist = max(0.0001, float(node.dist) * scale_ratio)
        node.dist = format(node.dist, '.4f')

    if range_of_taxa_num[0] == range_of_taxa_num[1] == 4:
        lba_ratio = 0.15
    else:
        lba_ratio = -1

    if random.random() < lba_ratio:
        __, leaves = _find_lba_branches(tree)
        if len(tree.get_leaves()) != 4:
            leaves = random.sample(tree.get_leaves(), 4)
    else:
        leaves = random.sample(tree.get_leaves(), taxa_num)

    tree.prune(leaves, preserve_branch_length=True)
    tree.unroot()

    ans = tree.write(format=5)
    match = re.findall(r'taxon\d+:', ans)
    idx = ['0']
    number_set = [i for i in range(1, taxa_num)]

    for i in range(0, taxa_num - 1):
        sample_number = random.choice(number_set)
        number_set.remove(sample_number)
        idx.append(str(sample_number))

    for i in range(0, len(match)):
        ans = ans.replace(match[i], idx[i] + ':')

    q.put(ans)


def run_simulation_topology(simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
                           num_of_process, distribution_of_internal_branch_length,
                           distribution_of_external_branch_length, range_of_mean_pairwise_divergence,
                           seed=42, verbose=False, logger=None):
    """
    Generate phylogenetic tree topologies.

    Parameters:
    simulation_dir: Path to simulation directory
    num_of_topology: Number of MSAs to simulate
    taxa_num: Number of taxa in final tree
    range_of_taxa_num: Range of taxa numbers [lower, upper] (as string like '[5, 40]' or list)
    num_of_process: Number of processes for parallel execution
    distribution_of_internal_branch_length: Distribution parameters [type, param1, param2] (as string or list)
    distribution_of_external_branch_length: Distribution parameters [type, param1, param2] (as string or list)
    range_of_mean_pairwise_divergence: Range [min, max] for mean pairwise divergence (as string or list)
    seed: Random seed for reproducibility (default: 42)
    verbose: Whether to show verbose output
    """
    # Parse parameters
    if isinstance(range_of_taxa_num, str):
        range_of_taxa_num = list(eval(range_of_taxa_num))
    if isinstance(distribution_of_internal_branch_length, str):
        distribution_of_internal_branch_length = list(eval(distribution_of_internal_branch_length))
    if isinstance(distribution_of_external_branch_length, str):
        distribution_of_external_branch_length = list(eval(distribution_of_external_branch_length))
    if isinstance(range_of_mean_pairwise_divergence, str):
        range_of_mean_pairwise_divergence = list(eval(range_of_mean_pairwise_divergence))

    # Create label_file directory and set output path
    label_file_dir = Path(simulation_dir) / 'label_file'
    label_file_dir.mkdir(parents=True, exist_ok=True)
    output_newick = label_file_dir / 'newick.csv'

    if verbose:
        _print_stderr(f"Generating {num_of_topology} topologies...")

    # Set up multiprocessing
    if num_of_process <= 0:
        num_of_process = max(1, cpu_count() - 1)

    q = multiprocessing.Manager().Queue()
    para_list = [
        (q, seed + i, taxa_num, range_of_taxa_num, distribution_of_internal_branch_length,
         distribution_of_external_branch_length, range_of_mean_pairwise_divergence)
        for i in range(0, num_of_topology)
    ]

    pool = Pool(num_of_process)
    pool.map(_gen_newick, para_list)
    pool.close()
    pool.join()

    # Collect results
    csv_list = []
    while not q.empty():
        tmp_topo = q.get()
        csv_list.append(tmp_topo)

    csv_list.sort()  # Keep the same order for reproducibility

    # Save to CSV
    dictionary = {"newick": csv_list}
    data = pd.DataFrame(dictionary)
    data.to_csv(output_newick, index=False)

    if logger:
        logger.log_detail(f"Generated {len(csv_list)} topologies and saved to '{output_newick}'.")
    elif verbose:
        _print_stderr(f"Generated {len(csv_list)} topologies and saved to '{output_newick}'.")

    return output_newick


def _format_num(x, digits=5):
    """Format number with fixed decimal places."""
    if hasattr(x, 'ndim') and x.ndim > 0:
        x = x.item()
    return f"{float(x):.{digits}f}"


def _process_single_model(args):
    """Process a single model configuration."""
    i, model, max_indel_length, indel_rate_bounds, seed = args

    # Set random seed
    np.random.seed(seed)

    model_orig = model
    model_name = f"{model}Model{i}"

    # Generate parameters
    I = np.random.uniform(0, 1, size=1)
    A = np.random.uniform(0, 5, size=1)
    Pi = np.random.dirichlet([5, 5, 5, 5], size=1)[0]  # Nucleotide proportions
    Pi = [_format_num(p, 5) for p in Pi]

    output_lines = [f'[MODEL] {model_name}']

    # Generate parameters based on different models
    model_params = {
        'HKY': 1, 'K80': 1,
        'TrN': 2,
        'TIM': 3, 'TIMef': 3,
        'TVM': 4,
        'SYM': 5, 'GTR': 5,
        'UNREST': 11
    }

    if model in model_params:
        n_params = model_params[model]
        params = [_format_num(np.random.uniform(0, 3), 2) for _ in range(n_params)]
        submodel = f"{model} {' '.join(params)}"
    else:
        submodel = model

    output_lines.extend([
        f' [submodel] {submodel}',
        f' [rates] {_format_num(I, 2)} {_format_num(A, 2)} 0',
        f' [indelmodel] POW 1.5 {max_indel_length}',
        f' [indelrate] {_format_num(np.random.uniform(*indel_rate_bounds), 5)}'
    ])

    if model_orig in ['F81', 'HKY', 'TrN', 'TIM', 'TVM', 'GTR']:
        output_lines.append(f' [statefreq] {" ".join(Pi)}')

    return {'lines': output_lines, 'name': model_name}


def _write_indelible_control_file(out_file, seed, results, model_names, newick, tree_ids, partition_names, msa_lengths):
    """Write INDELible control file."""
    with open(out_file, 'w') as f:
        f.write("[TYPE] NUCLEOTIDE 2\n")

        # Write basic settings
        f.write("\n")
        f.write("[SETTINGS]\n")
        f.write(" [output] FASTA\n")
        f.write(f" [randomseed] {seed}\n")

        f.write("\n")
        for result in results:
            f.write('\n'.join(result['lines']) + '\n')

        # Write TREE block
        f.write("\n")
        for tid, nw in zip(tree_ids, newick):
            f.write(f"[TREE] {tid} {nw}\n")

        # Write PARTITIONS block
        f.write("\n")
        for pname, tid, mname, length in zip(partition_names, tree_ids, model_names, msa_lengths):
            f.write(f"[PARTITIONS] {pname} [ {tid} {mname} {length} ]\n")

        # Write EVOLVE block
        f.write("\n[EVOLVE]\n")
        for pname, tid in zip(partition_names, tree_ids):
            f.write(f"{pname} 1 {tid}\n")


def _write_control_file_and_run_indelible(args):
    """Generate control file and run INDELible for a batch."""
    start_idx, end_idx, seed, results, model_names, newick, tree_ids, partition_names, msa_lengths, out_dir, indelible_path = args

    batch_out_dir = Path(out_dir) / f"batch_{start_idx}_{end_idx}"
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    out_file = batch_out_dir / "control.txt"

    _write_indelible_control_file(
        out_file, seed,
        results[start_idx:end_idx],
        model_names[start_idx:end_idx],
        newick[start_idx:end_idx],
        tree_ids[start_idx:end_idx],
        partition_names[start_idx:end_idx],
        msa_lengths[start_idx:end_idx]
    )

    # Run indelible
    try:
        subprocess.run([str(indelible_path), str(out_file)], cwd=str(batch_out_dir), check=True)
    except subprocess.CalledProcessError as e:
        _print_stderr(f"Command failed with return code {e.returncode}")
        raise


def _find_indelible_executable(simulation_dir=None, indelible_path=None):
    """
    Find INDELible executable. Search order:
    1. User-specified path
    2. simulation_dir/indelible
    3. fusang.py directory/simulation/indelible
    4. Current directory/simulation/indelible
    """
    if indelible_path is not None:
        path = Path(indelible_path)
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"INDELible executable not found at specified path: {indelible_path}")

    # Try simulation_dir/indelible
    if simulation_dir is not None:
        path = Path(simulation_dir) / 'indelible'
        if path.exists():
            return path.resolve()

    # Try fusang.py directory/simulation/indelible
    script_dir = Path(__file__).resolve().parent
    path = script_dir / 'simulation' / 'indelible'
    if path.exists():
        return path.resolve()

    # Try current directory/simulation/indelible
    path = Path('simulation') / 'indelible'
    if path.exists():
        return path.resolve()

    raise FileNotFoundError(
        f"INDELible executable not found. Please specify --indelible_path or place it at:\n"
        f"  - {simulation_dir}/indelible (if simulation_dir is specified)\n"
        f"  - {script_dir}/simulation/indelible\n"
        f"  - ./simulation/indelible"
    )


def run_simulation_sequence(simulation_dir, taxa_num, num_of_topology, num_of_process,
                           len_of_msa_lower_bound, len_of_msa_upper_bound,
                           range_of_indel_substitution_rate, max_indel_length,
                           seed=42, indelible_path=None, batch_size=1000, verbose=False, logger=None):
    """
    Generate sequences using INDELible. This generates INDELible control files and runs INDELible.

    Parameters:
    simulation_dir: Path to simulation directory
    taxa_num: Number of taxa in final tree
    num_of_topology: Number of MSAs to simulate
    num_of_process: Number of processes for parallel execution
    len_of_msa_lower_bound: Lower bound of MSA length
    len_of_msa_upper_bound: Upper bound of MSA length
    range_of_indel_substitution_rate: Range [min, max] for indel substitution rate
    max_indel_length: Maximum indel length
    seed: Random seed for reproducibility (default: 42)
    indelible_path: Path to INDELible executable (None to auto-detect)
    verbose: Whether to show verbose output
    """
    # Find INDELible executable
    indelible_path = _find_indelible_executable(simulation_dir, indelible_path)
    if not indelible_path.is_executable():
        raise PermissionError(f"INDELible found at {indelible_path} but is not executable. Please check permissions.")

    # Create simulate_data directory
    simulate_data_dir = Path(simulation_dir) / 'simulate_data'
    simulate_data_dir.mkdir(parents=True, exist_ok=True)

    # Get input newick file
    newick_file = Path(simulation_dir) / 'label_file' / 'newick.csv'
    if not newick_file.exists():
        raise FileNotFoundError(f"Newick file not found: {newick_file}. Run topology simulation first.")

    # Parse indel substitution rate range
    if isinstance(range_of_indel_substitution_rate, str):
        indel_rate_range = list(eval(range_of_indel_substitution_rate))
    else:
        indel_rate_range = range_of_indel_substitution_rate
    indel_rate_bounds = (indel_rate_range[0], indel_rate_range[1])

    if logger:
        logger.log_detail(f"Generating sequences for {num_of_topology} topologies...")
    elif verbose:
        _print_stderr(f"Generating sequences for {num_of_topology} topologies...")

    # Set random seed
    np.random.seed(seed)

    # Generate models
    possible_models = ['JC', 'TIM', 'TIMef', 'GTR', 'UNREST']
    modelset = np.random.choice(possible_models, size=num_of_topology, replace=True)

    # Generate model configurations
    if num_of_process <= 0:
        n_cores = max(1, cpu_count() - 1)
    else:
        n_cores = num_of_process

    model_args = [
        (i, model, max_indel_length, indel_rate_bounds, seed + i)
        for i, model in enumerate(modelset, 1)
    ]
    with Pool(n_cores) as pool:
        model_results = pool.map(_process_single_model, model_args)

    model_names = [result['name'] for result in model_results]

    # Read Newick file
    newick_data = pd.read_csv(newick_file)
    # The CSV has a 'newick' column, access it by name or index
    if 'newick' in newick_data.columns:
        newick = newick_data['newick'].tolist()
    else:
        # Fallback: if column name is different, use the last column
        newick = newick_data.iloc[:, -1].tolist()
    if len(newick) != num_of_topology:
        raise ValueError(
            f"The number of topologies in the newick file ({len(newick)}) "
            f"does not match the number of simulations ({num_of_topology})."
        )

    # Generate tree IDs and partition names
    tree_ids = [f"t_sim{i}" for i in range(1, num_of_topology + 1)]
    partition_names = [f"p{i}" for i in range(1, num_of_topology + 1)]

    # Generate MSA lengths
    if len_of_msa_lower_bound == len_of_msa_upper_bound:
        msa_lengths = np.repeat(len_of_msa_lower_bound, num_of_topology)
    else:
        msa_lengths = np.random.randint(len_of_msa_lower_bound, len_of_msa_upper_bound + 1, size=num_of_topology)

    # Process in batches
    # Use batch_size to control data amount per batch for parallel processing
    # Each batch uses a different seed to ensure reproducibility and consistency
    # The seed offset ensures that parallel results match serial results when using the same base seed
    batch_args = []
    for start_idx in range(0, num_of_topology, batch_size):
        end_idx = min(start_idx + batch_size, num_of_topology)
        # Use batch index to offset seed, ensuring each batch has unique but deterministic seed
        # This ensures parallel results are consistent with serial execution
        batch_seed = seed + (start_idx // batch_size)
        batch_args.append((
            start_idx, end_idx, batch_seed, model_results, model_names, newick,
            tree_ids, partition_names, msa_lengths, simulate_data_dir, indelible_path
        ))

    if logger:
        logger.log_detail(f"Processing {len(batch_args)} batches with batch size {batch_size}...")
    elif verbose:
        _print_stderr(f"Processing {len(batch_args)} batches with batch size {batch_size}...")

    with Pool(n_cores) as pool:
        pool.map(_write_control_file_and_run_indelible, batch_args)

    # Combine trees.txt files from all batches
    if logger:
        logger.log_detail("Combining trees.txt files from batches...")
    elif verbose:
        _print_stderr("Combining trees.txt files from batches...")

    trees_txt_path = simulate_data_dir / 'trees.txt'
    if trees_txt_path.exists():
        trees_txt_path.unlink()

    batch_dirs = sorted([d for d in simulate_data_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')])
    if batch_dirs:
        with open(trees_txt_path, 'w') as outfile:
            # Write header from first batch
            first_batch_trees = batch_dirs[0] / 'trees.txt'
            if first_batch_trees.exists():
                with open(first_batch_trees, 'r') as infile:
                    header_lines = [next(infile) for _ in range(6)]
                    outfile.writelines(header_lines)
            # Append data from all batches
            for batch_dir in batch_dirs:
                batch_trees = batch_dir / 'trees.txt'
                if batch_trees.exists():
                    with open(batch_trees, 'r') as infile:
                        lines = infile.readlines()
                        # Skip header (first 6 lines) and append rest
                        if len(lines) > 6:
                            outfile.writelines(lines[6:])

    if verbose:
        _print_stderr(f"Sequence generation completed. Results in {simulate_data_dir}")


def _get_msa_length(msa_file):
    """Get MSA length from a FASTA file."""
    with open(msa_file, 'r') as f:
        alignment = AlignIO.read(f, 'fasta')
    return len(alignment[0].seq)


def _extract_fasta_file(args):
    """Extract a single FASTA file."""
    file_path_str, in_dir, out_dir, max_length = args
    file_path = Path(in_dir) / file_path_str

    if file_path.is_file() and '.fas' in file_path.name and 'TRUE' in file_path.name:
        msa_length = _get_msa_length(file_path)
        if msa_length > max_length:
            out_fail_dir = Path(out_dir) / 'fail'
            out_fail_dir.mkdir(parents=True, exist_ok=True)
            dest_path = out_fail_dir / file_path.name
        else:
            dest_path = Path(out_dir) / file_path.name
        shutil.copy(file_path, dest_path)


def extract_fasta_data(simulation_dir, max_length=None, verbose=False, logger=None):
    """
    Extract FASTA data from simulation output.

    Parameters:
    simulation_dir: Path to simulation directory
    max_length: Maximum MSA length to keep (None for no limit, default: 1e10)
    verbose: Whether to show verbose output
    """
    simulate_data_dir = Path(simulation_dir) / 'simulate_data'
    fasta_data_dir = Path(simulation_dir) / 'fasta_data'
    fasta_data_dir.mkdir(parents=True, exist_ok=True)

    if max_length is None:
        max_length = 1e10

    if verbose:
        _print_stderr(f"Extracting FASTA files from {simulate_data_dir}...")

    # Get all files in simulate_data directory (including batch subdirectories)
    file_list = []
    for item in simulate_data_dir.iterdir():
        if item.is_file() and '.fas' in item.name and 'TRUE' in item.name:
            file_list.append(str(item.relative_to(simulate_data_dir)))
        elif item.is_dir() and item.name.startswith('batch_'):
            # Also check batch subdirectories
            for subitem in item.iterdir():
                if subitem.is_file() and '.fas' in subitem.name and 'TRUE' in subitem.name:
                    # Copy from batch directory
                    file_list.append(str(subitem.relative_to(simulate_data_dir)))

    if verbose:
        _print_stderr(f"Found {len(file_list)} FASTA files to extract")

    # Extract files in parallel
    extract_args = [(fname, simulate_data_dir, fasta_data_dir, max_length) for fname in file_list]
    with Pool(8) as pool:
        pool.map(_extract_fasta_file, extract_args)

    if verbose:
        _print_stderr(f"FASTA extraction completed. Results in {fasta_data_dir}")


def _assign_label(tree_str):
    """Assign label (0, 1, or 2) to a tree based on quartet topology."""
    t1 = Tree(tree_str, format=5)
    label_0 = Tree('((0,1),2,3);')
    label_1 = Tree('((0,2),1,3);')
    label_2 = Tree('((0,3),1,2);')
    if t1.robinson_foulds(label_0, unrooted_trees=True)[0] == 0:
        return 0
    elif t1.robinson_foulds(label_1, unrooted_trees=True)[0] == 0:
        return 1
    elif t1.robinson_foulds(label_2, unrooted_trees=True)[0] == 0:
        return 2
    else:
        raise Exception("A fatal error occurred.")


def _get_numpy(fasta_file_path):
    """Convert FASTA file to numpy array."""
    dic = {'A': '0', 'T': '1', 'C': '2', 'G': '3', '-': '4'}
    matrix_out = []
    fasta_dic = {}

    with open(fasta_file_path, 'r') as aln:
        for line in aln:
            if line[0] == ">":
                header = line[1:].rstrip('\n').strip()
                fasta_dic[header] = []
            elif line[0].isalpha() or line[0] == '-':
                processed_line = line[:].rstrip('\n').strip()
                for base, num in dic.items():
                    processed_line = processed_line.replace(base, num)
                line_list = [int(n) for n in list(processed_line)]
                tmp_line = line_list + [4] * (2000 - len(line_list))
                fasta_dic[header] += tmp_line[0:2000]

    taxa_block = []
    for taxa in sorted(list(fasta_dic.keys())):
        taxa_block.append(fasta_dic[taxa.strip()])
    matrix_out.append(taxa_block)

    return np.array(matrix_out)


def generate_numpy_data(simulation_dir, verbose=False, logger=None):
    """
    Generate numpy training data from FASTA files.

    Parameters:
    simulation_dir: Path to simulation directory
    verbose: Whether to show verbose output
    """
    # Copy trees.txt if it exists
    trees_src = Path(simulation_dir) / 'simulate_data' / 'trees.txt'
    trees_dest = Path(simulation_dir) / 'label_file' / 'trees.txt'
    if trees_src.exists():
        trees_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(trees_src, trees_dest)

    # Set up paths
    trees_txt_path = trees_dest
    fasta_data_dir = Path(simulation_dir) / 'fasta_data'
    numpy_data_dir = Path(simulation_dir) / 'numpy_data'
    numpy_seq_dir = numpy_data_dir / 'seq'
    numpy_label_dir = numpy_data_dir / 'label'
    numpy_seq_dir.mkdir(parents=True, exist_ok=True)
    numpy_label_dir.mkdir(parents=True, exist_ok=True)

    if not trees_txt_path.exists():
        raise FileNotFoundError(f"Trees file not found: {trees_txt_path}")

    if verbose:
        _print_stderr(f"Reading trees from {trees_txt_path}...")

    # Read trees.txt (skip first 5 lines, tab-separated, column 8 contains tree)
    csv_data = pd.read_table(trees_txt_path, skiprows=5, sep='\t', header=None)
    file_names = list(csv_data[0])
    topologies = list(csv_data[8])
    tree_dict = {}
    for i in range(len(file_names)):
        tree_dict[file_names[i]] = topologies[i]

    if verbose:
        _print_stderr(f"Processing FASTA files from {fasta_data_dir}...")

    # Process all FASTA files
    file_list = [f for f in fasta_data_dir.iterdir() if f.is_file() and '.fas' in f.name and 'TRUE' in f.name]
    random.shuffle(file_list)

    for fasta_file in file_list:
        if fasta_file.name == 'fail':
            continue

        # Extract file ID (e.g., "sim1" from "sim1_TRUE.fas")
        file_base = fasta_file.stem.replace('_TRUE', '')
        if file_base not in tree_dict:
            if verbose:
                _print_stderr(f"Warning: No tree found for {file_base}, skipping...")
            continue

        try:
            current_label = np.array(_assign_label(tree_dict[file_base]))
            current_seq = _get_numpy(fasta_file)

            # Generate output file names (e.g., "1.npy" from "sim1_TRUE.fas")
            file_id = file_base.replace('sim', '')
            seq_file = numpy_seq_dir / f'{file_id}.npy'
            label_file = numpy_label_dir / f'{file_id}.npy'

            np.save(seq_file, current_seq)
            np.save(label_file, current_label)

            if verbose:
                _print_stderr(f'[{current_label}] {current_seq.shape}')
        except Exception as e:
            if verbose:
                _print_stderr(f"Error processing {fasta_file.name}: {e}")
            continue

    if verbose:
        _print_stderr(f"Numpy data generation completed. Results in {numpy_data_dir}")


class SimulationLogger:
    """
    Logger for simulation process that writes detailed logs to file
    and shows summary on screen.
    """
    def __init__(self, log_file_path, verbose=True):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.log_file = None
        self._open_log_file()

    def _open_log_file(self):
        """Open log file for writing."""
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        self.log_file.write("=" * 80 + "\n")
        self.log_file.write(f"Fusang Simulation Log\n")
        self.log_file.write(f"Started at: {datetime.datetime.now().isoformat()}\n")
        self.log_file.write("=" * 80 + "\n\n")
        self.log_file.flush()

    def log_detail(self, message, timestamp=True):
        """Write detailed message to log file only."""
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{timestamp_str}] {message}\n")
        else:
            self.log_file.write(f"{message}\n")
        self.log_file.flush()

    def log_summary(self, message):
        """Write summary message to both screen and log file."""
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp_str}] {message}\n"
        self.log_file.write(log_message)
        self.log_file.flush()
        if self.verbose:
            _print_stderr(message)

    def log_both(self, message, timestamp=True):
        """Write message to both screen and log file."""
        if timestamp:
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp_str}] {message}\n"
        else:
            log_message = f"{message}\n"
        self.log_file.write(log_message)
        self.log_file.flush()
        if self.verbose:
            _print_stderr(message)

    def close(self):
        """Close log file."""
        if self.log_file:
            self.log_file.write("\n" + "=" * 80 + "\n")
            self.log_file.write(f"Log ended at: {datetime.datetime.now().isoformat()}\n")
            self.log_file.write("=" * 80 + "\n")
            self.log_file.close()
            self.log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _collect_simulation_statistics(simulation_dir):
    """
    Collect statistics about generated simulation data.

    Parameters:
    simulation_dir: Path to simulation directory

    Returns:
    Dictionary containing statistics
    """
    sim_path = Path(simulation_dir)
    stats = {}

    # Count numpy files
    numpy_seq_dir = sim_path / 'numpy_data' / 'seq'
    numpy_label_dir = sim_path / 'numpy_data' / 'label'
    if numpy_seq_dir.exists():
        seq_files = list(numpy_seq_dir.glob('*.npy'))
        stats['num_numpy_seq_files'] = len(seq_files)
        if seq_files:
            # Get shape info from first file
            try:
                sample_seq = np.load(str(seq_files[0]))
                stats['numpy_seq_shape'] = list(sample_seq.shape)
            except:
                stats['numpy_seq_shape'] = None
    else:
        stats['num_numpy_seq_files'] = 0

    if numpy_label_dir.exists():
        label_files = list(numpy_label_dir.glob('*.npy'))
        stats['num_numpy_label_files'] = len(label_files)
        if label_files:
            try:
                sample_label = np.load(str(label_files[0]))
                stats['numpy_label_shape'] = list(sample_label.shape) if hasattr(sample_label, 'shape') else None
            except:
                stats['numpy_label_shape'] = None
    else:
        stats['num_numpy_label_files'] = 0

    # Count FASTA files
    fasta_dir = sim_path / 'fasta_data'
    if fasta_dir.exists():
        fasta_files = [f for f in fasta_dir.iterdir() if f.is_file() and '.fas' in f.name and 'TRUE' in f.name]
        stats['num_fasta_files'] = len(fasta_files)

        # Calculate total FASTA file size
        total_fasta_size = sum(f.stat().st_size for f in fasta_files)
        stats['total_fasta_size_bytes'] = total_fasta_size
        stats['total_fasta_size_mb'] = round(total_fasta_size / (1024 * 1024), 2)

        # Get MSA length statistics from FASTA files
        msa_lengths = []
        for fasta_file in fasta_files[:min(100, len(fasta_files))]:  # Sample first 100 files
            try:
                with open(fasta_file, 'r') as f:
                    alignment = AlignIO.read(f, 'fasta')
                if len(alignment) > 0:
                    msa_lengths.append(len(alignment[0].seq))
            except:
                pass

        if msa_lengths:
            stats['msa_length_stats'] = {
                'min': int(min(msa_lengths)),
                'max': int(max(msa_lengths)),
                'mean': round(float(np.mean(msa_lengths)), 2),
                'median': int(np.median(msa_lengths)),
                'samples_measured': len(msa_lengths)
            }
    else:
        stats['num_fasta_files'] = 0

    # Count trees
    trees_file = sim_path / 'label_file' / 'trees.txt'
    if trees_file.exists():
        try:
            csv_data = pd.read_table(trees_file, skiprows=5, sep='\t', header=None)
            stats['num_trees'] = len(csv_data)
        except:
            stats['num_trees'] = 0
    else:
        stats['num_trees'] = 0

    # Count newick topologies
    newick_file = sim_path / 'label_file' / 'newick.csv'
    if newick_file.exists():
        try:
            newick_data = pd.read_csv(newick_file)
            stats['num_topologies'] = len(newick_data)
        except:
            stats['num_topologies'] = 0
    else:
        stats['num_topologies'] = 0

    return stats


def _generate_simulation_metadata(simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
                                 len_of_msa_upper_bound, len_of_msa_lower_bound, num_of_process,
                                 distribution_of_internal_branch_length, distribution_of_external_branch_length,
                                 range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
                                 max_indel_length, max_length, seed, batch_size, indelible_path,
                                 verbose=False, logger=None):
    """
    Generate metadata files (JSON and summary.txt) for simulation results.

    Parameters:
    simulation_dir: Path to simulation directory
    All other parameters: Simulation parameters
    verbose: Whether to show verbose output
    """
    sim_path = Path(simulation_dir)
    sim_path.mkdir(parents=True, exist_ok=True)

    # Collect statistics
    stats = _collect_simulation_statistics(simulation_dir)

    # Get version information
    try:
        import importlib.metadata
        try:
            fusang_version = importlib.metadata.version('fusang')
        except:
            fusang_version = '0.1.0'  # Fallback to pyproject.toml version
    except:
        fusang_version = '0.1.0'

    # Get Python and system information
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        scipy_version = scipy.__version__
    except:
        try:
            scipy_version = scipy.stats.__version__
        except:
            scipy_version = 'unknown'

    system_info = {
        'platform': platform.platform(),
        'python_version': python_version,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'scipy_version': scipy_version
    }

    # Prepare parameters for JSON (convert lists/tuples to lists for JSON serialization)
    def convert_for_json(obj):
        if isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    # Create metadata dictionary
    timestamp = datetime.datetime.now().isoformat()
    metadata = {
        'simulation_info': {
            'timestamp': timestamp,
            'simulation_dir': str(sim_path),
            'fusang_version': fusang_version,
        },
        'parameters': {
            'num_of_topology': num_of_topology,
            'taxa_num': taxa_num,
            'range_of_taxa_num': convert_for_json(range_of_taxa_num),
            'len_of_msa_upper_bound': len_of_msa_upper_bound,
            'len_of_msa_lower_bound': len_of_msa_lower_bound,
            'num_of_process': num_of_process,
            'distribution_of_internal_branch_length': convert_for_json(distribution_of_internal_branch_length),
            'distribution_of_external_branch_length': convert_for_json(distribution_of_external_branch_length),
            'range_of_mean_pairwise_divergence': convert_for_json(range_of_mean_pairwise_divergence),
            'range_of_indel_substitution_rate': convert_for_json(range_of_indel_substitution_rate),
            'max_indel_length': max_indel_length,
            'max_length': max_length,
            'seed': seed,
            'batch_size': batch_size,
            'indelible_path': str(indelible_path) if indelible_path else None,
        },
        'system_info': system_info,
        'statistics': convert_for_json(stats),
    }

    # Write JSON file
    json_file = sim_path / 'simulation_metadata.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Write summary.txt
    summary_file = sim_path / 'summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Fusang Simulation Dataset Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("Simulation Information:\n")
        f.write(f"  Timestamp: {timestamp}\n")
        f.write(f"  Simulation Directory: {sim_path}\n")
        f.write(f"  Fusang Version: {fusang_version}\n")
        f.write("\n")

        f.write("Parameters:\n")
        f.write(f"  Number of Topologies: {num_of_topology}\n")
        f.write(f"  Taxa Number: {taxa_num}\n")
        f.write(f"  Range of Taxa: {range_of_taxa_num}\n")
        f.write(f"  MSA Length Range: {len_of_msa_lower_bound} - {len_of_msa_upper_bound} bp\n")
        f.write(f"  Number of Processes: {num_of_process}\n")
        f.write(f"  Max Indel Length: {max_indel_length}\n")
        f.write(f"  Seed: {seed}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Internal Branch Length Distribution: {distribution_of_internal_branch_length}\n")
        f.write(f"  External Branch Length Distribution: {distribution_of_external_branch_length}\n")
        f.write(f"  Mean Pairwise Divergence Range: {range_of_mean_pairwise_divergence}\n")
        f.write(f"  Indel Substitution Rate Range: {range_of_indel_substitution_rate}\n")
        if max_length:
            f.write(f"  Max MSA Length Filter: {max_length} bp\n")
        f.write("\n")

        f.write("Output Statistics:\n")
        f.write(f"  Number of Topologies Generated: {stats.get('num_topologies', 0)}\n")
        f.write(f"  Number of Trees: {stats.get('num_trees', 0)}\n")
        f.write(f"  Number of FASTA Files: {stats.get('num_fasta_files', 0)}\n")
        if stats.get('total_fasta_size_mb'):
            f.write(f"  Total FASTA Size: {stats['total_fasta_size_mb']} MB\n")
        f.write(f"  Number of NumPy Sequence Files: {stats.get('num_numpy_seq_files', 0)}\n")
        f.write(f"  Number of NumPy Label Files: {stats.get('num_numpy_label_files', 0)}\n")

        if stats.get('numpy_seq_shape'):
            f.write(f"  NumPy Sequence Shape: {stats['numpy_seq_shape']}\n")
        if stats.get('numpy_label_shape'):
            f.write(f"  NumPy Label Shape: {stats['numpy_label_shape']}\n")

        if stats.get('msa_length_stats'):
            msa_stats = stats['msa_length_stats']
            f.write(f"\n  MSA Length Statistics (sampled from {msa_stats['samples_measured']} files):\n")
            f.write(f"    Min: {msa_stats['min']} bp\n")
            f.write(f"    Max: {msa_stats['max']} bp\n")
            f.write(f"    Mean: {msa_stats['mean']} bp\n")
            f.write(f"    Median: {msa_stats['median']} bp\n")
        f.write("\n")

        f.write("Training Data Information:\n")
        num_samples = stats.get('num_numpy_seq_files', 0)
        if num_samples > 0:
            f.write(f"  Total Training Samples: {num_samples}\n")
            f.write(f"  Recommended Train/Val/Test Split:\n")
            f.write(f"    Training Set (80%): ~{int(num_samples * 0.8)} samples\n")
            f.write(f"    Validation Set (10%): ~{int(num_samples * 0.1)} samples\n")
            f.write(f"    Test Set (10%): ~{int(num_samples * 0.1)} samples\n")
            f.write("\n")
            f.write(f"  Training Command Example:\n")
            f.write(f"    uv run fusang.py train \\\n")
            f.write(f"      -d {sim_path} \\\n")
            f.write(f"      -w {'240' if len_of_msa_upper_bound <= 1210 else '1200'} \\\n")
            f.write(f"      -o model/<MODEL_NAME>.weights.h5\n")
        f.write("\n")

        f.write("System Information:\n")
        f.write(f"  Platform: {system_info['platform']}\n")
        f.write(f"  Python Version: {system_info['python_version']}\n")
        f.write(f"  NumPy Version: {system_info['numpy_version']}\n")
        f.write(f"  Pandas Version: {system_info['pandas_version']}\n")
        f.write(f"  SciPy Version: {system_info['scipy_version']}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("For detailed information, see simulation_metadata.json\n")
        f.write("=" * 80 + "\n")

    if logger:
        logger.log_detail(f"Metadata files generated:")
        logger.log_detail(f"  - {json_file}")
        logger.log_detail(f"  - {summary_file}")
    elif verbose:
        _print_stderr(f"Metadata files generated:")
        _print_stderr(f"  - {json_file}")
        _print_stderr(f"  - {summary_file}")


def run_full_simulation(simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
                       len_of_msa_upper_bound, len_of_msa_lower_bound, num_of_process,
                       distribution_of_internal_branch_length, distribution_of_external_branch_length,
                       range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
                       max_indel_length, max_length=None, cleanup=True, verbose=False,
                       indelible_path=None, seed=42, batch_size=1000):
    """
    Run the complete simulation pipeline.

    Parameters:
    simulation_dir: Path to simulation directory
    num_of_topology: Number of MSAs to simulate
    taxa_num: Number of taxa in final tree
    range_of_taxa_num: Range of taxa numbers [lower, upper]
    len_of_msa_upper_bound: Upper bound of MSA length
    len_of_msa_lower_bound: Lower bound of MSA length
    num_of_process: Number of processes for parallel execution
    distribution_of_internal_branch_length: Distribution parameters [type, param1, param2]
    distribution_of_external_branch_length: Distribution parameters [type, param1, param2]
    range_of_mean_pairwise_divergence: Range [min, max] for mean pairwise divergence
    range_of_indel_substitution_rate: Range [min, max] for indel substitution rate
    max_indel_length: Maximum indel length
    max_length: Maximum MSA length to keep (None for no limit)
    cleanup: Whether to remove simulate_data directory after completion
    verbose: Whether to show verbose output
    """
    sim_path = Path(simulation_dir)
    sim_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    log_file_path = sim_path / 'simulation.log'
    logger = SimulationLogger(log_file_path, verbose=verbose)

    try:
        logger.log_summary("=" * 80)
        logger.log_summary("Starting Full Simulation Pipeline")
        logger.log_summary("=" * 80)
        logger.log_detail(f"Simulation directory: {sim_path}")
        logger.log_detail(f"Parameters:")
        logger.log_detail(f"  - Number of topologies: {num_of_topology}")
        logger.log_detail(f"  - Taxa number: {taxa_num}")
        logger.log_detail(f"  - Range of taxa: {range_of_taxa_num}")
        logger.log_detail(f"  - MSA length range: {len_of_msa_lower_bound} - {len_of_msa_upper_bound} bp")
        logger.log_detail(f"  - Number of processes: {num_of_process}")
        logger.log_detail(f"  - Max indel length: {max_indel_length}")
        logger.log_detail(f"  - Seed: {seed}")
        logger.log_detail(f"  - Batch size: {batch_size}")
        logger.log_detail(f"  - Internal branch distribution: {distribution_of_internal_branch_length}")
        logger.log_detail(f"  - External branch distribution: {distribution_of_external_branch_length}")
        logger.log_detail(f"  - Mean pairwise divergence range: {range_of_mean_pairwise_divergence}")
        logger.log_detail(f"  - Indel substitution rate range: {range_of_indel_substitution_rate}")
        if max_length:
            logger.log_detail(f"  - Max MSA length filter: {max_length} bp")
        logger.log_detail("")

        # Step 1: Generate topologies
        logger.log_summary("Step 1/6: Generating topologies...")
        logger.log_detail(f"Generating {num_of_topology} topologies with {num_of_process} processes")
        run_simulation_topology(
            simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
            num_of_process, distribution_of_internal_branch_length,
            distribution_of_external_branch_length, range_of_mean_pairwise_divergence,
            seed=seed, verbose=False, logger=logger
        )
        logger.log_summary("  ✓ Topology generation completed")

        # Step 2: Generate sequences using INDELible
        logger.log_summary("Step 2/6: Generating sequences with INDELible...")
        logger.log_detail(f"Generating sequences for {num_of_topology} topologies")
        run_simulation_sequence(
            simulation_dir, taxa_num, num_of_topology, num_of_process,
            len_of_msa_lower_bound, len_of_msa_upper_bound,
            range_of_indel_substitution_rate, max_indel_length,
            seed=seed, indelible_path=indelible_path, batch_size=batch_size, verbose=False, logger=logger
        )
        logger.log_summary("  ✓ Sequence generation completed")

        # Step 3: Copy trees.txt to label_file directory
        logger.log_detail("Copying trees.txt to label_file directory...")
        trees_src = Path(simulation_dir) / 'simulate_data' / 'trees.txt'
        trees_dest = Path(simulation_dir) / 'label_file' / 'trees.txt'
        if trees_src.exists():
            trees_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(trees_src, trees_dest)
            logger.log_detail(f"Copied {trees_src} to {trees_dest}")

        # Step 4: Extract FASTA data
        logger.log_summary("Step 3/6: Extracting FASTA data...")
        extract_fasta_data(simulation_dir, max_length=max_length, verbose=False, logger=logger)
        logger.log_summary("  ✓ FASTA extraction completed")

        # Step 5: Generate numpy data
        logger.log_summary("Step 4/6: Generating numpy training data...")
        generate_numpy_data(simulation_dir, verbose=False, logger=logger)
        logger.log_summary("  ✓ NumPy data generation completed")

        # Step 5: Cleanup if requested
        if cleanup:
            simulate_data_dir = Path(simulation_dir) / 'simulate_data'
            if simulate_data_dir.exists():
                logger.log_summary("Step 5/6: Cleaning up temporary files...")
                logger.log_detail(f"Removing directory: {simulate_data_dir}")
                shutil.rmtree(simulate_data_dir)
                logger.log_summary("  ✓ Cleanup completed")

        logger.log_summary("Step 6/6: Generating metadata files...")
        try:
            _generate_simulation_metadata(
                simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
                len_of_msa_upper_bound, len_of_msa_lower_bound, num_of_process,
                distribution_of_internal_branch_length, distribution_of_external_branch_length,
                range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
                max_indel_length, max_length, seed, batch_size, indelible_path,
                verbose=False, logger=logger
            )
            logger.log_summary("  ✓ Metadata files generated")
        except Exception as e:
            logger.log_summary(f"  ✗ Error generating metadata files: {e}")
            logger.log_detail(f"Error details: {type(e).__name__}: {e}")
            import traceback
            logger.log_detail(traceback.format_exc())
            # Re-raise to ensure the error is visible
            raise

        logger.log_summary("=" * 80)
        logger.log_summary("Simulation pipeline completed successfully!")
        logger.log_summary(f"Log file saved to: {log_file_path}")
        logger.log_summary("=" * 80)

        sim_path = Path(simulation_dir)
        result = {
            'numpy_seq_dir': str(sim_path / 'numpy_data' / 'seq'),
            'numpy_label_dir': str(sim_path / 'numpy_data' / 'label'),
            'fasta_dir': str(sim_path / 'fasta_data'),
            'trees_file': str(sim_path / 'label_file' / 'trees.txt'),
            'log_file': str(log_file_path)
        }

        return result
    finally:
        logger.close()


# ============================================================================
# Inference Functions
# ============================================================================


def infer_tree_from_msa(msa_file, sequence_type='standard', branch_model='gamma',
                        beam_size=1, window_coverage=1, verbose=False):
    """
    Infer phylogenetic tree from a single MSA file.

    Parameters:
    msa_file: Path to MSA file
    sequence_type: Sequence type for model selection
    branch_model: Branch length model for model selection
    beam_size: Beam search size
    window_coverage: Sliding window coverage factor
    verbose: Whether to show verbose output

    Returns:
    Inferred tree in Newick format (string)
    """
    # Parse MSA file
    save_alignment, taxa_name, len_of_msa, taxa_num = parse_msa_file(msa_file)

    # Initialize quartet data structures
    (start_end_list, comb_of_id, leave_node_name, leave_node_comb_name,
     dic_for_leave_node_comb_name, internal_node_name_pool) = initialize_quartet_data(taxa_num)

    # Convert alignment to numpy array
    org_seq = get_numpy(save_alignment)

    # Load deep learning model
    dl_model, window_size = load_dl_model(len_of_msa, sequence_type, branch_model)
    window_number = int(len_of_msa * float(window_coverage) // window_size + 1)

    # Generate predictions
    dl_predict = np.zeros((len(comb_of_id), 3))
    predict_verbose = 1 if verbose else 0
    fill_dl_predict(window_number, window_size, len_of_msa, comb_of_id, org_seq,
                   dl_model, dl_predict, verbose=predict_verbose)
    dl_predict /= window_number

    # Generate phylogenetic tree
    use_masking = taxa_num > MIN_TAXA_FOR_MASKING
    searched_tree_str = gen_phylogenetic_tree(
        dl_predict, int(beam_size), taxa_num, leave_node_comb_name,
        dic_for_leave_node_comb_name, internal_node_name_pool,
        use_masking=use_masking, start_end_list=start_end_list, comb_of_id=comb_of_id
    )
    searched_tree = transform_str(searched_tree_str, taxa_name, taxa_num)

    return searched_tree.strip()


def calculate_rf_distance(tree1_str, tree2_str, unrooted=True):
    """
    Calculate Robinson-Foulds distance between two trees.

    Parameters:
    tree1_str: First tree in Newick format (string)
    tree2_str: Second tree in Newick format (string)
    unrooted: Whether trees are unrooted (default: True)

    Returns:
    RF distance (int) and normalized RF distance (float)
    """
    try:
        tree1 = Tree(tree1_str)
        tree2 = Tree(tree2_str)

        result = tree1.robinson_foulds(tree2, unrooted_trees=unrooted)
        # robinson_foulds returns a list: [rf_distance, max_rf, ...]
        rf_distance = result[0]
        max_rf = result[1]
        normalized_rf = rf_distance / max_rf if max_rf > 0 else 0.0

        return rf_distance, normalized_rf, max_rf
    except Exception as e:
        return None, None, None


def load_model_from_path(model_path, window_size=None, verbose=False):
    """
    Load a model from a specified file path (read-only).

    Parameters:
    model_path: Path to the model .h5 file
    window_size: Window size (240 or 1200). If None, will be inferred from model path or data.
    verbose: Whether to show verbose output

    Returns:
    (model, window_size): Loaded model and window size
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    import sys

    if verbose:
        _print_stderr(f"Loading model from {model_path}...")

    # Determine window size from model path or parameter
    if window_size is None:
        # Try to infer from filename (e.g., S1G.h5 -> 240, S2G.h5 -> 1200)
        model_name = model_path.stem
        if len(model_name) >= 2 and model_name[1] == '1':
            window_size = WINDOW_SIZE_SHORT
        elif len(model_name) >= 2 and model_name[1] == '2':
            window_size = WINDOW_SIZE_LONG
        else:
            # Default to 240 if cannot determine
            window_size = WINDOW_SIZE_SHORT

    if verbose:
        _print_stderr(f"Creating model architecture (window size: {window_size})...")

    # Create appropriate model architecture
    dl_model = get_dl_model(window_size)
    if verbose:
        _print_stderr("Model architecture created.")

    # Load weights (read-only, no modification)
    if verbose:
        _print_stderr("Loading model weights...")
    dl_model.load_weights(filepath=str(model_path))

    if verbose:
        _print_stderr("Model loaded successfully.")

    return dl_model, window_size


def evaluate_numpy_data(model_path, numpy_seq_dir, numpy_label_dir,
                        window_size=None, batch_size=32, output_dir=None,
                        verbose=False):
    """
    Evaluate model accuracy and performance on numpy sequence and label data.

    Parameters:
    model_path: Path to the model .h5 file (read-only)
    numpy_seq_dir: Directory containing numpy sequence files (.npy)
    numpy_label_dir: Directory containing numpy label files (.npy)
    window_size: Window size (240 or 1200). If None, will be inferred from model path.
    batch_size: Batch size for prediction (default: 32)
    output_dir: Directory to save evaluation results (None to skip saving)
    verbose: Whether to show verbose output

    Returns:
    Dictionary containing evaluation results with accuracy and performance metrics
    """
    import time
    import sys
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # Load model
    if verbose:
        _print_stderr(f"Loading model from {model_path}...")
    dl_model, window_size = load_model_from_path(model_path, window_size, verbose=verbose)
    if verbose:
        _print_stderr(f"Model loaded. Window size: {window_size}")

    # Get all numpy files
    seq_dir = Path(numpy_seq_dir)
    label_dir = Path(numpy_label_dir)
    seq_files = sorted([f.name for f in seq_dir.glob('*.npy')])
    label_files = sorted([f.name for f in label_dir.glob('*.npy')])

    if len(seq_files) != len(label_files):
        raise ValueError(f"Mismatch: {len(seq_files)} sequence files but {len(label_files)} label files")

    if len(seq_files) == 0:
        raise ValueError(f"No numpy files found in {numpy_seq_dir}")

    if verbose:
        _print_stderr(f"Found {len(seq_files)} data files to process...")

    if verbose:
        # Process data in batches: load a batch, process it, then move to next batch
        _print_stderr(f"Processing {len(seq_files)} files (loading and predicting in batches of {batch_size})...")

    y_all = []
    y_pred_proba = []
    label_counts = np.zeros(3, dtype=np.int32)

    start_time = time.time()

    # Process files in batches
    n_batches = int(np.ceil(len(seq_files) / batch_size))

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(seq_files))
        batch_files = list(zip(seq_files[start_idx:end_idx], label_files[start_idx:end_idx]))

        # Load batch of data
        X_batch = []
        y_batch = []

        for seq_file, label_file in batch_files:
            seq_path = seq_dir / seq_file
            label_path = label_dir / label_file

            X = np.load(str(seq_path))
            y = np.load(str(label_path))

            # Handle different input shapes (same as in NumpyDataGenerator)
            if len(X.shape) == 3 and X.shape[0] == 1:
                X = X[0]  # Remove batch dimension

            # Handle taxa dimension: should be 4 taxa
            if X.shape[0] > 4:
                X = X[:4, :]
            elif X.shape[0] < 4:
                padding = np.zeros((4 - X.shape[0], X.shape[1]), dtype=X.dtype)
                X = np.vstack([X, padding])

            # Handle length dimension: should be window_size
            if X.shape[1] > window_size:
                X = X[:, :window_size]
            elif X.shape[1] < window_size:
                padding = np.full((4, window_size - X.shape[1]), 4, dtype=X.dtype)
                X = np.hstack([X, padding])

            # Add channel dimension: (4, window_size) -> (4, window_size, 1)
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=-1)

            X = X.astype(np.float32)

            # Handle label: ensure it's a scalar integer
            if isinstance(y, np.ndarray):
                if y.size == 1:
                    y = int(y.item())
                elif len(y.shape) == 0:
                    y = int(y)
                else:
                    raise ValueError(f"Unexpected label shape: {y.shape} for file {label_file}")

            X_batch.append(X)
            y_batch.append(y)

        # Convert batch to numpy array
        X_batch = np.array(X_batch)  # Shape: (batch_size, 4, window_size, 1)

        # Make predictions on this batch
        pred_proba = dl_model.predict(X_batch, verbose=0, batch_size=len(X_batch))
        y_pred_proba.append(pred_proba)

        # Store labels
        y_all.extend(y_batch)
        label_counts += np.bincount(y_batch, minlength=3)

        # Calculate current accuracy and loss on all processed data so far
        y_all_current = np.array(y_all, dtype=np.int32)
        y_pred_proba_current = np.vstack(y_pred_proba)
        y_pred_current = np.argmax(y_pred_proba_current, axis=1)

        # Calculate accuracy
        current_accuracy = np.mean(y_all_current == y_pred_current)

        # Calculate loss (sparse categorical crossentropy)
        # Convert labels to one-hot for loss calculation
        y_one_hot = np.zeros((len(y_all_current), 3), dtype=np.float32)
        y_one_hot[np.arange(len(y_all_current)), y_all_current] = 1.0

        # Calculate cross-entropy loss
        epsilon = 1e-7  # Small value to avoid log(0)
        pred_proba_clipped = np.clip(y_pred_proba_current, epsilon, 1.0 - epsilon)
        loss = -np.mean(np.sum(y_one_hot * np.log(pred_proba_clipped), axis=1))

        # Calculate performance metrics
        elapsed_time = time.time() - start_time
        samples_done = len(y_all)
        samples_per_second = samples_done / elapsed_time if elapsed_time > 0 else 0

        # Estimate remaining time
        remaining_samples = len(seq_files) - samples_done
        estimated_remaining_time = remaining_samples / samples_per_second if samples_per_second > 0 else 0

        # Show progress with performance metrics
        files_done = end_idx
        percent = (batch_idx + 1) * 100 // n_batches

        # Format time strings
        elapsed_str = f"{elapsed_time:.1f}s"
        if estimated_remaining_time > 0:
            if estimated_remaining_time < 60:
                remaining_str = f"{estimated_remaining_time:.1f}s"
            elif estimated_remaining_time < 3600:
                remaining_str = f"{estimated_remaining_time/60:.1f}m"
            else:
                remaining_str = f"{estimated_remaining_time/3600:.1f}h"
        else:
            remaining_str = "calculating..."

        if verbose:
            _print_stderr(f"  Processed: {files_done}/{len(seq_files)} files ({percent}%) | "
                  f"Accuracy: {current_accuracy:.4f} | "
                  f"Loss: {loss:.4f} | "
                  f"Speed: {samples_per_second:.1f} samples/s | "
                  f"Elapsed: {elapsed_str} | "
                  f"Remaining: ~{remaining_str} | "
                  f"Batch {batch_idx + 1}/{n_batches}")

    if verbose:
        _print_stderr()  # New line after progress

    inference_time = time.time() - start_time

    # Convert to numpy arrays
    y_all = np.array(y_all, dtype=np.int32)  # Shape: (n_samples,)
    y_pred_proba = np.vstack(y_pred_proba)  # Shape: (n_samples, 3)
    y_pred = np.argmax(y_pred_proba, axis=1)

    if verbose:
        # Show summary
        _print_stderr(f"Processing completed: {len(y_all)} samples")
        _print_stderr(f"Label distribution: Class 0: {label_counts[0]}, "
            f"Class 1: {label_counts[1]}, "
            f"Class 2: {label_counts[2]}")
        _print_stderr()

        _print_stderr("Calculating evaluation metrics...")

    # Calculate metrics
    accuracy = accuracy_score(y_all, y_pred)
    cm = confusion_matrix(y_all, y_pred)
    class_report = classification_report(y_all, y_pred, output_dict=True, zero_division=0)

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i in range(3):
        mask = y_all == i
        if np.sum(mask) > 0:
            per_class_accuracy[i] = accuracy_score(y_all[mask], y_pred[mask])
        else:
            per_class_accuracy[i] = None

    # Performance metrics
    samples_per_second = len(y_all) / inference_time
    avg_time_per_sample = inference_time / len(y_all)

    stats = {
        'total_samples': len(y_all),
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'inference_time_seconds': inference_time,
        'samples_per_second': samples_per_second,
        'avg_time_per_sample_ms': avg_time_per_sample * 1000,
        'window_size': window_size,
        'batch_size': batch_size
    }

    # Save results if output directory provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / 'evaluation_results.txt'
        detailed_results_file = output_path / 'detailed_results.txt'

        # Save summary results
        with open(results_file, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Window size: {window_size}\n")
            f.write(f"Batch size: {batch_size}\n\n")
            f.write(f"Total samples: {stats['total_samples']}\n")
            f.write(f"Overall accuracy: {stats['accuracy']:.4f}\n\n")

            f.write("Per-class accuracy:\n")
            for class_idx, acc in stats['per_class_accuracy'].items():
                if acc is not None:
                    f.write(f"  Class {class_idx}: {acc:.4f}\n")
                else:
                    f.write(f"  Class {class_idx}: N/A (no samples)\n")
            f.write("\n")

            f.write("Confusion Matrix:\n")
            f.write("      Predicted\n")
            f.write("       0    1    2\n")
            for i in range(3):
                f.write(f"{i}   {cm[i, 0]:4d} {cm[i, 1]:4d} {cm[i, 2]:4d}\n")
            f.write("\n")

            f.write("Performance Metrics:\n")
            f.write(f"  Total inference time: {stats['inference_time_seconds']:.2f} seconds\n")
            f.write(f"  Samples per second: {stats['samples_per_second']:.2f}\n")
            f.write(f"  Average time per sample: {stats['avg_time_per_sample_ms']:.4f} ms\n")
            f.write("\n")

            f.write("Classification Report:\n")
            f.write(f"  Precision (macro avg): {class_report['macro avg']['precision']:.4f}\n")
            f.write(f"  Recall (macro avg): {class_report['macro avg']['recall']:.4f}\n")
            f.write(f"  F1-score (macro avg): {class_report['macro avg']['f1-score']:.4f}\n")
            f.write(f"  Weighted avg precision: {class_report['weighted avg']['precision']:.4f}\n")
            f.write(f"  Weighted avg recall: {class_report['weighted avg']['recall']:.4f}\n")
            f.write(f"  Weighted avg F1-score: {class_report['weighted avg']['f1-score']:.4f}\n")

        # Save detailed results (per-sample predictions)
        with open(detailed_results_file, 'w') as f:
            f.write("Detailed Results (per sample)\n")
            f.write("=" * 50 + "\n")
            f.write("File\tTrue_Label\tPredicted_Label\tConfidence\n")
            for i, (seq_file, label_file) in enumerate(zip(seq_files, label_files)):
                true_label = y_all[i]
                pred_label = y_pred[i]
                confidence = y_pred_proba[i, pred_label]
                f.write(f"{seq_file}\t{true_label}\t{pred_label}\t{confidence:.4f}\n")

        if verbose:
            _print_stderr(f"\nResults saved to: {output_dir}")

    return {
        'statistics': stats,
        'predictions': y_pred,
        'true_labels': y_all,
        'prediction_probabilities': y_pred_proba
    }


# ============================================================================
# Model Training Functions
# ============================================================================

def _create_numpy_data_generator_class():
    """Create NumpyDataGenerator class that inherits from keras.utils.Sequence."""
    # Import TensorFlow/Keras to get Sequence class
    _, _, _, _ = _import_tensorflow()
    from keras.utils import Sequence

    class NumpyDataGenerator(Sequence):
        """
        Data generator for loading numpy files on-the-fly during training.
        This avoids loading all data into memory at once.
        Inherits from keras.utils.Sequence for compatibility with Keras.
        """
        def __init__(self, file_pairs, batch_size=32, shuffle=True, window_size=240):
            """
            Initialize the data generator.

            Parameters:
            file_pairs: List of (seq_file_path, label_file_path) tuples
            batch_size: Batch size for training
            shuffle: Whether to shuffle data each epoch
            window_size: Window size for sequences (240 or 1200)
            """
            super().__init__()
            self.file_pairs = file_pairs
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.window_size = window_size
            self.indices = np.arange(len(file_pairs))

        def __len__(self):
            """Return the number of batches per epoch."""
            return int(np.ceil(len(self.file_pairs) / self.batch_size))

        def __getitem__(self, idx):
            """Generate one batch of data."""
            batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_X = []
            batch_y = []

            for i in batch_indices:
                seq_path, label_path = self.file_pairs[i]

                # Load data
                X = np.load(str(seq_path))
                y = np.load(str(label_path))

                # Handle different input shapes
                # Expected formats:
                # - (1, num_taxa, length) -> convert to (4, window_size, 1)
                # - (num_taxa, length) -> convert to (4, window_size, 1)
                # - (4, length) -> convert to (4, window_size, 1)

                # Remove batch dimension if present
                if len(X.shape) == 3 and X.shape[0] == 1:
                    X = X[0]  # Now shape: (num_taxa, length)

                # Get window size from model (default 240)
                window_size = getattr(self, 'window_size', 240)

                # Handle taxa dimension: should be 4 taxa
                if X.shape[0] > 4:
                    # Take first 4 taxa
                    X = X[:4, :]
                elif X.shape[0] < 4:
                    # Pad with zeros if less than 4 taxa
                    padding = np.zeros((4 - X.shape[0], X.shape[1]), dtype=X.dtype)
                    X = np.vstack([X, padding])

                # Handle length dimension: should be window_size (240 or 1200)
                if X.shape[1] > window_size:
                    # Take first window_size positions
                    X = X[:, :window_size]
                elif X.shape[1] < window_size:
                    # Pad with 4 (gap/unknown) if shorter
                    padding = np.full((4, window_size - X.shape[1]), 4, dtype=X.dtype)
                    X = np.hstack([X, padding])

                # Add channel dimension: (4, window_size) -> (4, window_size, 1)
                if len(X.shape) == 2:
                    X = np.expand_dims(X, axis=-1)

                # Convert to float32 for model input
                X = X.astype(np.float32)

                # Handle label: ensure it's a scalar integer
                if isinstance(y, np.ndarray):
                    if y.size == 1:
                        y = int(y.item())
                    elif len(y.shape) == 0:
                        y = int(y)
                    else:
                        raise ValueError(f"Unexpected label shape: {y.shape}")

                batch_X.append(X)
                batch_y.append(y)

            # Convert to numpy arrays
            batch_X = np.array(batch_X)  # Shape: (batch_size, 4, window_size, 1)
            batch_y = np.array(batch_y, dtype=np.int32)  # Shape: (batch_size,)

            return batch_X, batch_y

        def on_epoch_end(self):
            """Shuffle indices at the end of each epoch."""
            if self.shuffle:
                np.random.shuffle(self.indices)

        def get_all_data(self):
            """
            Load all data into memory (for test set evaluation).
            Use sparingly as this defeats the purpose of the generator.
            """
            X_all = []
            y_all = []
            window_size = getattr(self, 'window_size', 240)

            for seq_path, label_path in self.file_pairs:
                X = np.load(str(seq_path))
                y = np.load(str(label_path))

                # Apply same shape transformation as __getitem__
                if len(X.shape) == 3 and X.shape[0] == 1:
                    X = X[0]

                # Handle taxa dimension
                if X.shape[0] > 4:
                    X = X[:4, :]
                elif X.shape[0] < 4:
                    padding = np.zeros((4 - X.shape[0], X.shape[1]), dtype=X.dtype)
                    X = np.vstack([X, padding])

                # Handle length dimension
                if X.shape[1] > window_size:
                    X = X[:, :window_size]
                elif X.shape[1] < window_size:
                    padding = np.full((4, window_size - X.shape[1]), 4, dtype=X.dtype)
                    X = np.hstack([X, padding])

                # Add channel dimension
                if len(X.shape) == 2:
                    X = np.expand_dims(X, axis=-1)

                X = X.astype(np.float32)

                # Handle label
                if isinstance(y, np.ndarray):
                    if y.size == 1:
                        y = int(y.item())
                    elif len(y.shape) == 0:
                        y = int(y)

                X_all.append(X)
                y_all.append(y)

            X_all = np.array(X_all)
            y_all = np.array(y_all, dtype=np.int32)
            return X_all, y_all

    return NumpyDataGenerator

# Create the class (will be created when first needed)
_NumpyDataGeneratorClass = None

def _get_numpy_data_generator_class():
    """Get or create the NumpyDataGenerator class."""
    global _NumpyDataGeneratorClass
    if _NumpyDataGeneratorClass is None:
        _NumpyDataGeneratorClass = _create_numpy_data_generator_class()
    return _NumpyDataGeneratorClass

# Alias for backward compatibility - this is now a factory function
def NumpyDataGenerator(*args, **kwargs):
    """Factory function to create NumpyDataGenerator instances."""
    cls = _get_numpy_data_generator_class()
    return cls(*args, **kwargs)


def get_training_file_lists(numpy_seq_dir, numpy_label_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, verbose=False):
    """
    Get file lists for training, validation, and test sets without loading data.

    Parameters:
    numpy_seq_dir: Directory containing sequence numpy files
    numpy_label_dir: Directory containing label numpy files
    train_ratio: Ratio of data for training
    val_ratio: Ratio of data for validation
    test_ratio: Ratio of data for testing
    verbose: Whether to show verbose output

    Returns:
    (train_file_pairs, val_file_pairs, test_file_pairs) where each is a list of (seq_path, label_path) tuples
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Get all numpy files
    seq_dir = Path(numpy_seq_dir)
    label_dir = Path(numpy_label_dir)
    seq_files = sorted([f.name for f in seq_dir.glob('*.npy')])
    label_files = sorted([f.name for f in label_dir.glob('*.npy')])

    if len(seq_files) != len(label_files):
        raise ValueError(f"Mismatch: {len(seq_files)} sequence files but {len(label_files)} label files")

    if verbose:
        _print_stderr(f"Found {len(seq_files)} data files...")

    # Create file pairs
    file_pairs = []
    for seq_file, label_file in zip(seq_files, label_files):
        seq_path = seq_dir / seq_file
        label_path = label_dir / label_file
        file_pairs.append((seq_path, label_path))

    # Shuffle
    np.random.shuffle(file_pairs)

    # Split into train/val/test
    n_total = len(file_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_file_pairs = file_pairs[:n_train]
    val_file_pairs = file_pairs[n_train:n_train + n_val]
    test_file_pairs = file_pairs[n_train + n_val:]

    if verbose:
        _print_stderr(f"Training set: {len(train_file_pairs)} samples")
        _print_stderr(f"Validation set: {len(val_file_pairs)} samples")
        _print_stderr(f"Test set: {len(test_file_pairs)} samples")

    return train_file_pairs, val_file_pairs, test_file_pairs


def load_training_data(numpy_seq_dir, numpy_label_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, verbose=False):
    """
    Load training data from numpy files.

    Parameters:
    numpy_seq_dir: Directory containing sequence numpy files
    numpy_label_dir: Directory containing label numpy files
    train_ratio: Ratio of data for training
    val_ratio: Ratio of data for validation
    test_ratio: Ratio of data for testing
    verbose: Whether to show verbose output

    Returns:
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Get all numpy files
    seq_dir = Path(numpy_seq_dir)
    label_dir = Path(numpy_label_dir)
    seq_files = sorted([f.name for f in seq_dir.glob('*.npy')])
    label_files = sorted([f.name for f in label_dir.glob('*.npy')])

    if len(seq_files) != len(label_files):
        raise ValueError(f"Mismatch: {len(seq_files)} sequence files but {len(label_files)} label files")

    if verbose:
        _print_stderr(f"Loading {len(seq_files)} data files...")

    # Load all data
    X_all = []
    y_all = []

    for seq_file, label_file in zip(seq_files, label_files):
        seq_path = seq_dir / seq_file
        label_path = label_dir / label_file

        X = np.load(str(seq_path))
        y = np.load(str(label_path))

        # Reshape if needed: X should be (1, 4, length) -> (4, length, 1)
        if len(X.shape) == 3 and X.shape[0] == 1:
            X = X[0]  # Remove batch dimension
        if X.shape[0] == 4 and len(X.shape) == 2:
            X = np.expand_dims(X, axis=-1)  # Add channel dimension

        # Handle label: ensure it's a scalar integer
        if isinstance(y, np.ndarray):
            if y.size == 1:
                y = int(y.item())
            elif len(y.shape) == 0:
                y = int(y)
            else:
                raise ValueError(f"Unexpected label shape: {y.shape}")

        X_all.append(X)
        y_all.append(y)

    # Convert to numpy arrays
    # X_all should be list of (4, length, 1) arrays - we'll stack them
    # y_all should be list of integers
    X_all = np.array(X_all)  # Shape: (n_samples, 4, length, 1)
    y_all = np.array(y_all, dtype=np.int32)  # Shape: (n_samples,)

    # Shuffle
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]

    # Split into train/val/test
    n_total = len(X_all)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    X_val = X_all[n_train:n_train + n_val]
    y_val = y_all[n_train:n_train + n_val]
    X_test = X_all[n_train + n_val:]
    y_test = y_all[n_train + n_val:]

    if verbose:
        _print_stderr(f"Training set: {len(X_train)} samples")
        _print_stderr(f"Validation set: {len(X_val)} samples")
        _print_stderr(f"Test set: {len(X_test)} samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _get_training_history_logger(tsv_path):
    """
    Create a Keras Callback to save training metrics to a TSV file.

    Parameters:
    tsv_path: Path to TSV file where training history will be saved

    Returns:
    A Keras Callback instance
    """
    from keras.callbacks import Callback

    class TrainingHistoryLogger(Callback):
        """
        Keras Callback to save training metrics (loss, accuracy, val_loss, val_accuracy)
        to a TSV file after each epoch.
        """
        def __init__(self, tsv_path):
            super().__init__()
            self.tsv_path = Path(tsv_path)
            self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
            self.epoch_data = []

            # Initialize TSV file with header if it doesn't exist or is empty
            if not self.tsv_path.exists() or self.tsv_path.stat().st_size == 0:
                header = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
                pd.DataFrame(columns=header).to_csv(self.tsv_path, sep='\t', index=False)
            else:
                # Check if file has correct header
                try:
                    df = pd.read_csv(self.tsv_path, sep='\t', nrows=0)
                    expected_columns = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
                    if list(df.columns) != expected_columns:
                        # Recreate file with correct header
                        pd.DataFrame(columns=expected_columns).to_csv(self.tsv_path, sep='\t', index=False)
                except Exception:
                    # If file is corrupted, recreate it
                    header = ['epoch', 'loss', 'accuracy', 'val_loss', 'val_accuracy']
                    pd.DataFrame(columns=header).to_csv(self.tsv_path, sep='\t', index=False)

        def on_epoch_end(self, epoch, logs=None):
            """Called at the end of each epoch to save metrics."""
            if logs is None:
                logs = {}

            # Extract metrics (handle both with and without 'val_' prefix)
            epoch_num = epoch + 1  # Keras uses 0-indexed epochs, we use 1-indexed
            loss = logs.get('loss', None)
            accuracy = logs.get('accuracy', None)
            val_loss = logs.get('val_loss', None)
            val_accuracy = logs.get('val_accuracy', None)

            # Append to list
            self.epoch_data.append({
                'epoch': epoch_num,
                'loss': loss,
                'accuracy': accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            # Append to TSV file
            new_row = pd.DataFrame([{
                'epoch': epoch_num,
                'loss': loss,
                'accuracy': accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }])
            new_row.to_csv(self.tsv_path, mode='a', header=False, sep='\t', index=False)

    return TrainingHistoryLogger(tsv_path)


def train_model(model, train_gen, val_gen=None, epochs=100, batch_size=32,
                learning_rate=0.001, model_save_path=None, verbose=1,
                monitor='val_loss', patience=10):
    """
    Train a deep learning model using data generators.

    This function automatically saves the best model based on validation performance.
    The best model is determined by monitoring a specified metric (default: val_loss).
    Early stopping is enabled by default to prevent overfitting.

    Parameters:
    model: Keras model to train
    train_gen: Training data generator (NumpyDataGenerator or tuple of (X, y))
    val_gen: Validation data generator (NumpyDataGenerator or tuple of (X, y)) or None
    epochs: Number of training epochs
    batch_size: Batch size for training (used if generators are not provided)
    learning_rate: Learning rate
    model_save_path: Path to save trained model (None to not save)
    verbose: Verbosity level (0=silent, 1=progress, 2=one line per epoch)
    monitor: Metric to monitor for checkpointing and early stopping.
             Options: 'val_loss', 'val_accuracy', 'loss', 'accuracy' (default: 'val_loss')
    patience: Number of epochs with no improvement before early stopping.
              Set to None to disable early stopping (default: 10)

    Returns:
    Training history
    """
    # Compile model
    _, _, _, optimizers = _import_tensorflow()
    from keras.callbacks import ModelCheckpoint, EarlyStopping

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Prepare validation data
    validation_data = None
    if val_gen is not None:
        if isinstance(val_gen, tuple):
            validation_data = val_gen
        else:
            validation_data = val_gen

    # Prepare callbacks
    callbacks = []

    # Add training history logger if model_save_path is provided
    if model_save_path is not None:
        model_path = Path(model_save_path)
        # Save training history TSV in the same directory as the model
        tsv_path = model_path.parent / 'training_history.tsv'
        history_logger = _get_training_history_logger(str(tsv_path))
        callbacks.append(history_logger)
        if verbose >= 1:
            _print_stderr(f"Training history will be saved to {tsv_path}")

        # Add ModelCheckpoint to save best model based on validation performance
        # This is always enabled when validation data is available
        if validation_data is not None:
            checkpoint_path = model_path.parent / 'best_model.weights.h5'
            # Determine mode: 'min' for loss, 'max' for accuracy
            mode = 'min' if 'loss' in monitor else 'max'
            checkpoint = ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                mode=mode,
                verbose=1 if verbose >= 1 else 0
            )
            callbacks.append(checkpoint)
            if verbose >= 1:
                _print_stderr(f"Model checkpoint: saving best model to {checkpoint_path} (monitoring {monitor}, mode={mode})")

        # Add EarlyStopping if patience is specified and validation data is available
        if patience is not None and validation_data is not None:
            # Determine mode: 'min' for loss, 'max' for accuracy
            mode = 'min' if 'loss' in monitor else 'max'
            early_stop = EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,  # Restore best weights when stopping
                mode=mode,
                verbose=1 if verbose >= 1 else 0
            )
            callbacks.append(early_stop)
            if verbose >= 1:
                _print_stderr(f"Early stopping: will stop if {monitor} doesn't improve for {patience} epochs (mode={mode})")

    # Train model
    # If using generators, don't pass batch_size (generator handles it)
    # If using arrays, pass batch_size
    fit_kwargs = {
        'epochs': epochs,
        'verbose': verbose,
        'callbacks': callbacks
    }

    # Check if train_gen is a generator (Sequence-like object with __len__ and __getitem__)
    # or if it's a tuple of arrays
    is_generator = (not isinstance(train_gen, tuple) and
                    hasattr(train_gen, '__len__') and
                    hasattr(train_gen, '__getitem__') and
                    hasattr(train_gen, 'on_epoch_end'))

    if is_generator:
        # Generator handles batching, so don't pass batch_size
        fit_kwargs['x'] = train_gen
        fit_kwargs['validation_data'] = validation_data
    else:
        # Arrays: need batch_size
        fit_kwargs['x'] = train_gen[0]
        fit_kwargs['y'] = train_gen[1]
        fit_kwargs['validation_data'] = validation_data
        fit_kwargs['batch_size'] = batch_size

    history = model.fit(**fit_kwargs)

    # Save final model if path provided
    # Always load and save the best model when validation data is available
    if model_save_path is not None:
        model_path = Path(model_save_path)
        # Keras requires .weights.h5 extension for save_weights
        if not str(model_path).endswith('.weights.h5'):
            # Auto-append .weights.h5 if extension is missing or different
            if model_path.suffix in ['.h5', '.hdf5']:
                # Replace existing extension
                model_path = model_path.with_suffix('.weights.h5')
            else:
                # Append .weights.h5
                model_path = model_path.with_suffix('.weights.h5')
            if verbose >= 1:
                _print_stderr(f"Note: Model path adjusted to {model_path} (Keras requires .weights.h5 extension)")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # If validation data was used, load the best model weights
        if validation_data is not None:
            checkpoint_path = model_path.parent / 'best_model.weights.h5'
            if checkpoint_path.exists():
                # Load best model weights
                model.load_weights(str(checkpoint_path))
                if verbose >= 1:
                    _print_stderr(f"Loaded best model weights (based on {monitor}) from {checkpoint_path}")
            else:
                # If no checkpoint was saved (shouldn't happen), use current weights
                if verbose >= 1:
                    _print_stderr(f"Warning: No checkpoint found, using final epoch weights")

        # Save the model (which is now the best model if validation was used)
        model.save_weights(str(model_path))
        if verbose >= 1:
            _print_stderr(f"Model saved to {model_save_path}")

    return history


def train_fusang_model(numpy_seq_dir, numpy_label_dir, window_size=240, epochs=100,
                       batch_size=32, learning_rate=0.001, train_ratio=0.8, val_ratio=0.1,
                       model_save_path=None, verbose=1, monitor='val_loss', patience=10):
    """
    Train a Fusang model from numpy data using data generators for memory efficiency.

    Note: Keras requires model_save_path to end with .weights.h5. If a different extension
    is provided, it will be automatically adjusted.

    This function automatically saves the best model based on validation performance.
    The best model is determined by monitoring a specified metric (default: val_loss).
    Early stopping is enabled by default to prevent overfitting.

    Parameters:
    numpy_seq_dir: Directory containing sequence numpy files
    numpy_label_dir: Directory containing label numpy files
    window_size: Window size (240 or 1200)
    epochs: Number of training epochs
    batch_size: Batch size for training
    learning_rate: Learning rate
    train_ratio: Ratio of data for training
    val_ratio: Ratio of data for validation
    model_save_path: Path to save trained model
    verbose: Verbosity level
    monitor: Metric to monitor for checkpointing and early stopping.
             Options: 'val_loss', 'val_accuracy', 'loss', 'accuracy' (default: 'val_loss')
    patience: Number of epochs with no improvement before early stopping.
              Set to None to disable early stopping (default: 10)

    Returns:
    Trained model and training history
    """
    # Configure GPU memory growth to avoid memory issues
    tf, _, _, _ = _import_tensorflow()

    # Disable layout optimizer to avoid warnings about layout optimization failures
    # This is safe and won't affect training performance significantly
    try:
        tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    except (AttributeError, ValueError):
        # Fallback for older TensorFlow versions or if option is not available
        pass

    try:
        # Try new API first (TensorFlow 2.x)
        gpus = tf.config.list_physical_devices('GPU')
    except AttributeError:
        # Fall back to experimental API for older versions
        gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if verbose >= 1:
                _print_stderr(f"Configured {len(gpus)} GPU(s) with memory growth enabled")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            if verbose >= 1:
                _print_stderr(f"Warning: Could not set GPU memory growth: {e}")

    # Get file lists instead of loading all data
    train_file_pairs, val_file_pairs, test_file_pairs = get_training_file_lists(
        numpy_seq_dir, numpy_label_dir, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio, verbose=verbose >= 1
    )

    # Create data generators
    train_gen = NumpyDataGenerator(train_file_pairs, batch_size=batch_size, shuffle=True, window_size=window_size)
    val_gen = NumpyDataGenerator(val_file_pairs, batch_size=batch_size, shuffle=False, window_size=window_size)
    test_gen = NumpyDataGenerator(test_file_pairs, batch_size=batch_size, shuffle=False, window_size=window_size)

    # Normalize model_save_path to ensure .weights.h5 extension (Keras requirement)
    if model_save_path is not None:
        model_path = Path(model_save_path)
        # Keras requires .weights.h5 extension for save_weights
        if not str(model_path).endswith('.weights.h5'):
            # Auto-append .weights.h5 if extension is missing or different
            if model_path.suffix in ['.h5', '.hdf5']:
                # Replace existing extension
                normalized_path = model_path.with_suffix('.weights.h5')
            else:
                # Append .weights.h5
                normalized_path = model_path.with_suffix('.weights.h5')
            if verbose >= 1:
                _print_stderr(f"Note: Model path adjusted from {model_save_path} to {normalized_path} (Keras requires .weights.h5 extension)")
            model_save_path = str(normalized_path)

    # Create model based on window size
    model = get_dl_model(window_size)

    # Check if model file exists and load it for continued training
    if model_save_path is not None and os.path.exists(model_save_path):
        # Backup existing model with timestamp suffix
        model_path = Path(model_save_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = model_path.parent / f"{model_path.stem}_{timestamp}{model_path.suffix}"
        shutil.copy2(model_save_path, backup_path)
        if verbose >= 1:
            _print_stderr(f"Existing model found at {model_save_path}")
            _print_stderr(f"Backed up to {backup_path}")
            _print_stderr(f"Loading existing model weights for continued training...")

        # Load existing model weights
        try:
            model.load_weights(model_save_path)
            if verbose >= 1:
                _print_stderr(f"Successfully loaded model weights from {model_save_path}")
        except Exception as e:
            if verbose >= 1:
                _print_stderr(f"Warning: Failed to load model weights: {e}")
                _print_stderr("Starting training from scratch...")

    # Train model using generators
    history = train_model(
        model, train_gen, val_gen=val_gen,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        model_save_path=model_save_path, verbose=verbose,
        monitor=monitor, patience=patience
    )

    # Evaluate on test set
    if verbose >= 1:
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        _print_stderr(f"Test accuracy: {test_acc:.4f}")

    return model, history


# ============================================================================
# Command Line Interface
# ============================================================================

def _add_inference_args(parser):
    """Add inference arguments to a parser.

    Avoids using argument groups to prevent deprecation warnings in newer argparse versions.
    """
    # Add directly to parser (avoids deprecation warning with nested argument groups)
    parser.add_argument("-i", "--input", dest="msa_file", type=str, required=True, metavar="MSA_FILE", help="Input MSA file path")
    parser.add_argument("-o", "--output", type=str, help="Output tree file path (Newick format)")
    parser.add_argument("-b", "--beam_size", type=str, default='1', help="Beam search size (default: 1)")
    parser.add_argument("-t", "--sequence_type", type=str, default='standard', choices=['standard', 'coding', 'noncoding'], help="Sequence type for model selection (default: standard)")
    parser.add_argument("-r", "--branch_model", type=str, default='gamma', choices=['gamma', 'uniform'], help="Branch length model (default: gamma)")
    parser.add_argument("-w", "--window_coverage", type=str, default='1', help="Sliding window coverage factor (default: 1)")


def _extract_inference_args(args):
    """Extract inference parameters from args namespace."""
    msa_file = getattr(args, 'msa_file', None) or getattr(args, 'input', None)
    output_path = getattr(args, 'output', None)
    beam_size = int(getattr(args, 'beam_size', '1'))
    sequence_type = getattr(args, 'sequence_type', 'standard')
    branch_model = getattr(args, 'branch_model', 'gamma')
    window_coverage = float(getattr(args, 'window_coverage', '1'))
    return msa_file, output_path, beam_size, sequence_type, branch_model, window_coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'fusang',
        description='Fusang: Deep learning-based phylogenetic tree inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Infer phylogenetic tree (using subcommand):
  fusang infer -i input.fasta -o output.tree

  # Generate simulation data:
  fusang simulate -o ./simulation -n 20 -t 5 ...

  # Train a model:
  fusang train -d ./simulation/out/S1U -o ./model.h5

  # Evaluate model on numpy data:
  fusang evaluate -m ./model.h5 -d ./simulation/out/S1U -o ./results

For detailed help on a subcommand, use: fusang <subcommand> --help
        '''
    )

    # Global arguments
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress warning messages")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output including stderr messages")

    # Always create subparsers to show them in help
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', metavar='{infer,simulate,train,evaluate}')
    subparsers.required = True

    # Inference mode
    infer_parser = subparsers.add_parser(
        'infer',
        help='Infer phylogenetic tree from MSA',
        description='Infer phylogenetic tree from multiple sequence alignment (MSA) using deep learning.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_inference_args(infer_parser)

    # Simulation mode
    sim_parser = subparsers.add_parser(
        'simulate',
        help='Generate simulation data for training',
        description='Generate simulated phylogenetic data for model training using INDELible.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sim_parser.add_argument("-o", "--output", dest="simulation_dir", type=str, required=True, help="Path to simulation output directory")
    sim_parser.add_argument("-n", "--num_of_topology", type=int, required=True, help="Number of MSAs to simulate")
    sim_parser.add_argument("-t", "--taxa_num", type=int, required=True, help="Number of taxa in final tree")
    sim_parser.add_argument("--range_of_taxa_num", type=str, default='[5, 40]', help="Range of taxa numbers [lower, upper] (default: '[5, 40]')")
    sim_parser.add_argument("--len_of_msa_upper_bound", type=int, default=1200, help="Upper bound of MSA length (default: 1200)")
    sim_parser.add_argument("--len_of_msa_lower_bound", type=int, default=240, help="Lower bound of MSA length (default: 240)")
    sim_parser.add_argument("-p", "--num_of_process", type=int, default=cpu_count(), help=f"Number of processes for parallel execution (default: {cpu_count()}, number of CPU cores)")
    sim_parser.add_argument("--distribution_of_internal_branch_length", type=str, default='[1, 0.5, 0.3]', help="Internal branch length distribution [type, param1, param2] (default: '[1, 0.5, 0.3]')")
    sim_parser.add_argument("--distribution_of_external_branch_length", type=str, default='[1, 0.5, 0.3]', help="External branch length distribution [type, param1, param2] (default: '[1, 0.5, 0.3]')")
    sim_parser.add_argument("--range_of_mean_pairwise_divergence", type=str, default='[0.03, 0.3]', help="Range of mean pairwise divergence [min, max] (default: '[0.03, 0.3]')")
    sim_parser.add_argument("--range_of_indel_substitution_rate", type=str, default='[0.01, 0.25]', help="Range of indel substitution rate [min, max] (default: '[0.01, 0.25]')")
    sim_parser.add_argument("--max_indel_length", type=int, default=10, help="Maximum indel length (default: 10)")
    sim_parser.add_argument("--max_length", type=int, default=None, help="Maximum MSA length to keep (default: no limit)")
    sim_parser.add_argument("--indelible_path", type=str, default=None, help="Path to INDELible executable (default: auto-detect from simulation/indelible relative to fusang.py)")
    sim_parser.add_argument("-S", "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    sim_parser.add_argument("-b", "--batch_size", type=int, default=1000, help="Batch size for parallel processing (default: 1000)")
    sim_parser.add_argument("--no-cleanup", action="store_true", help="Do not remove simulate_data directory after completion")

    # Training mode
    train_parser = subparsers.add_parser(
        'train',
        help='Train a Fusang model',
        description='Train deep learning models for quartet topology prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    train_parser.add_argument("-d", "--data_dir", type=str, required=True, help="Data directory (numpy_data or simulation output dir). Automatically uses data_dir/seq and data_dir/label, or data_dir/numpy_data/seq and data_dir/numpy_data/label")
    train_parser.add_argument("-w", "--window_size", type=int, choices=[240, 1200], default=240, help="Window size for model (default: 240)")
    train_parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    train_parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    train_parser.add_argument("-r", "--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    train_parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training (default: 0.8)")
    train_parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data for validation (default: 0.1)")
    train_parser.add_argument("-o", "--output", dest="model_save_path", type=str, required=True, help="Path to save trained model weights (.h5 file)")
    train_parser.add_argument("--model_save_path", dest="model_save_path", type=str, help=argparse.SUPPRESS)  # Hidden alias for backward compatibility
    train_parser.add_argument("-M", "--monitor", type=str, default='val_loss',
                              choices=['val_loss', 'val_accuracy', 'loss', 'accuracy'],
                              help="Metric to monitor for checkpointing and early stopping (default: val_loss)")
    train_parser.add_argument("-P", "--patience", type=int, default=10,
                              help="Number of epochs with no improvement before early stopping. Set to 0 to disable (default: 10)")

    # Evaluation mode
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a model on numpy data',
        description='Evaluate a trained model on numpy sequence/label data.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    eval_parser.add_argument("-m", "--model", dest="model_path", type=str, required=True, help="Path to model weights (.h5)")
    eval_parser.add_argument("-d", "--data_dir", type=str, required=True, help="Data directory (numpy_data or simulation output dir). Automatically uses data_dir/seq and data_dir/label, or data_dir/numpy_data/seq and data_dir/numpy_data/label")
    eval_parser.add_argument("-o", "--output", dest="output_dir", type=str, default=None, help="Directory to save evaluation results (default: do not save)")
    eval_parser.add_argument("-w", "--window_size", type=int, choices=[240, 1200], default=None, help="Window size for evaluation (default: infer from model name)")
    eval_parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for prediction (default: 32)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    # Parse arguments
    args = parser.parse_args()

    # Validate that quiet and verbose are not both specified
    if hasattr(args, 'quiet') and hasattr(args, 'verbose') and args.quiet and args.verbose:
        parser.error("Cannot specify both -q/--quiet and -v/--verbose")

    # Derive numpy directories for train and evaluate modes from data_dir
    if hasattr(args, 'mode') and hasattr(args, 'data_dir'):
        if args.mode in ['train', 'evaluate']:
            data_path = Path(args.data_dir)
            # Check if data_dir points to numpy_data directory (has seq/ and label/ subdirectories)
            if (data_path / 'seq').exists() and (data_path / 'label').exists():
                # data_dir is numpy_data directory
                args.numpy_seq_dir = str(data_path / 'seq')
                args.numpy_label_dir = str(data_path / 'label')
            elif (data_path / 'numpy_data' / 'seq').exists() and (data_path / 'numpy_data' / 'label').exists():
                # data_dir is simulation output directory
                args.numpy_seq_dir = str(data_path / 'numpy_data' / 'seq')
                args.numpy_label_dir = str(data_path / 'numpy_data' / 'label')
            else:
                raise ValueError(f"Invalid data_dir: {args.data_dir}. Expected numpy_data directory (with seq/ and label/ subdirectories) or simulation output directory (with numpy_data/seq/ and numpy_data/label/ subdirectories)")

    # Determine verbosity
    verbose = args.verbose and not args.quiet

    # Use 'with' statement to ensure stderr is properly restored
    with _stderr_redirector:
        # Handle different modes
        if args.mode == 'simulate':
            # Simulation mode
            range_of_taxa_num = list(eval(args.range_of_taxa_num))
            distribution_of_internal_branch_length = list(eval(args.distribution_of_internal_branch_length))
            distribution_of_external_branch_length = list(eval(args.distribution_of_external_branch_length))
            range_of_mean_pairwise_divergence = list(eval(args.range_of_mean_pairwise_divergence))
            range_of_indel_substitution_rate = list(eval(args.range_of_indel_substitution_rate))

            result = run_full_simulation(
                args.simulation_dir, args.num_of_topology, args.taxa_num, range_of_taxa_num,
                args.len_of_msa_upper_bound, args.len_of_msa_lower_bound, args.num_of_process,
                distribution_of_internal_branch_length, distribution_of_external_branch_length,
                range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
                args.max_indel_length, max_length=args.max_length, cleanup=not args.no_cleanup,
                verbose=verbose, indelible_path=args.indelible_path,
                seed=args.seed, batch_size=args.batch_size
            )

            if verbose:
                _print_stderr(f"\nSimulation completed. Data saved to:")
                _print_stderr(f"  Sequences: {result['numpy_seq_dir']}")
                _print_stderr(f"  Labels: {result['numpy_label_dir']}")
                _print_stderr(f"  FASTA: {result['fasta_dir']}")
                if 'evaluation' in result:
                    _print_stderr("\nEvaluation results saved (see evaluation directory in simulation output).")

        elif args.mode == 'evaluate':
            # Evaluation mode (standalone) - numpy data evaluation only
            output_dir = args.output_dir

            evaluation_results = evaluate_numpy_data(
                args.model_path, args.numpy_seq_dir, args.numpy_label_dir,
                window_size=args.window_size, batch_size=args.batch_size,
                output_dir=output_dir, verbose=verbose
            )

            stats = evaluation_results['statistics']
            _print_stderr("\nEvaluation Results:")
            _print_stderr(f"  Total samples: {stats['total_samples']}")
            _print_stderr(f"  Accuracy: {stats['accuracy']:.4f}")
            _print_stderr(f"  Window size: {stats['window_size']}")
            _print_stderr(f"  Batch size: {stats['batch_size']}")
            if verbose and output_dir is not None:
                _print_stderr(f"\nResults saved to: {output_dir}")

        elif args.mode == 'train':
            # Training mode
            # Convert patience: 0 means disable early stopping (None)
            patience = None if args.patience == 0 else args.patience

            model, history = train_fusang_model(
                args.numpy_seq_dir, args.numpy_label_dir,
                window_size=args.window_size, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                model_save_path=args.model_save_path, verbose=2 if verbose else 1,
                monitor=args.monitor, patience=patience
            )

            if verbose:
                _print_stderr(f"\nTraining completed. Model saved to: {args.model_save_path}")

        elif args.mode == 'infer':
            msa_file, output_path, beam_size, sequence_type, branch_model, window_coverage = _extract_inference_args(args)

            # Use the unified inference function
            searched_tree = infer_tree_from_msa(
                msa_file, sequence_type=sequence_type, branch_model=branch_model,
                beam_size=beam_size, window_coverage=window_coverage, verbose=verbose
            )

            # Write output
            write_output(searched_tree, output_path)
        else:
            parser.error("Unknown mode. Use --help for available subcommands.")
