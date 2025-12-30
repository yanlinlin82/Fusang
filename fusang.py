import argparse
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import warnings
from io import StringIO
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import AlignIO
from ete3 import Tree

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
        print('Error of function get_current_topology_id, exit the program')
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
            print('Error of pruning 4 taxa from current tree, the current tree is:')
            print(crt_tree)
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
            processed_line = line[:].rstrip('\n').strip()
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
        import tensorflow as tf
        from keras import layers, models, optimizers
        globals()['tf'] = tf
        globals()['layers'] = layers
        globals()['models'] = models
        globals()['optimizers'] = optimizers
    return tf, layers, models, optimizers


def get_dl_model_1200():
    """
    Get the definition of DL model for sequences longer than 1200.
    This model aims to solve the default case for sequences with length larger than 1200.
    """
    _, layers, models, _ = _import_tensorflow()
    conv_x=[4,1,1,1,1,1,1,1]
    conv_y=[1,2,2,2,2,2,2,2]
    pool=[1,4,4,4,2,2,2,1]
    filter_s=[1024,1024,128,128,128,128,128,128]

    visible = layers.Input(shape=(4,1200,1))
    x = visible

    for l in list(range(0,8)):
        x = layers.ZeroPadding2D(padding=((0, 0), (0,conv_y[l]-1)))(x)
        x = layers.Conv2D(filters=filter_s[l], kernel_size=(conv_x[l], conv_y[l]), strides=1, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.AveragePooling2D(pool_size=(1,pool[l]))(x)

    flat = layers.Flatten()(x)

    y = layers.Reshape((4,1200))(visible)
    y = layers.Bidirectional(layers.LSTM(128,return_sequences=True))(y)
    y = layers.Bidirectional(layers.LSTM(128,return_sequences=True))(y)
    y = layers.Bidirectional(layers.LSTM(128))(y)
    flat = layers.concatenate([flat, y],axis=-1)

    hidden1 = layers.Dense(1024,activation='relu')(flat)
    drop1 = layers.Dropout(rate=0.2)(hidden1)
    output = layers.Dense(3, activation='softmax')(drop1)
    model = models.Model(inputs=visible, outputs=output)

    return model


def get_dl_model_240():
    """
    Get the definition of DL model for sequences up to 1210.
    This model aims to solve the short length case for sequences with length larger than 240.
    """
    _, layers, models, _ = _import_tensorflow()
    conv_x=[4,1,1,1,1,1,1,1]
    conv_y=[1,2,2,2,2,2,2,2]
    pool=[1,2,2,2,2,2,2,2]
    filter_s=[1024,1024,128,128,128,128,128,128]

    visible = layers.Input(shape=(4,240,1))
    x = visible

    for l in list(range(0,8)):
        x = layers.ZeroPadding2D(padding=((0, 0), (0,conv_y[l]-1)))(x)
        x = layers.Conv2D(filters=filter_s[l], kernel_size=(conv_x[l], conv_y[l]), strides=1, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.AveragePooling2D(pool_size=(1,pool[l]))(x)
        #print(x.shape)

    flat = layers.Flatten()(x)

    y = layers.Reshape((4,240))(visible)
    y = layers.Bidirectional(layers.LSTM(128,return_sequences=True))(y)
    y = layers.Bidirectional(layers.LSTM(128,return_sequences=True))(y)
    y = layers.Bidirectional(layers.LSTM(128))(y)
    flat = layers.concatenate([flat, y],axis=-1)

    hidden1 = layers.Dense(1024,activation='relu')(flat)
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


def parse_msa_file(msa_dir):
    """Parse MSA file and return alignment data and taxa names."""
    support_format = ['.fas', '.phy', '.fasta', '.phylip']
    bio_format = ['fasta', 'phylip', 'fasta', 'phylip']

    taxa_name = {}
    for i in range(len(support_format)):
        if msa_dir.endswith(support_format[i]):
            try:
                alignment = AlignIO.read(open(msa_dir), bio_format[i])
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
                print('Something wrong about your msa file, please check your msa file')
                sys.exit(1)

    print('we do not support this format of msa')
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


def write_output(searched_tree, output_path=None, save_prefix=None):
    """Write output tree to file or stdout."""
    if not searched_tree.endswith('\n'):
        searched_tree += '\n'

    if output_path is not None:
        output_file = Path(output_path)
    elif save_prefix is not None:
        output_file = Path('./dl_output') / f'{save_prefix}.txt'
    else:
        print(searched_tree, end='')
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'a') as f:
        f.write(searched_tree)


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
        dl_model = get_dl_model_240()
        model_num = '1'
        window_size = WINDOW_SIZE_SHORT
    else:
        dl_model = get_dl_model_1200()
        model_num = '2'
        window_size = WINDOW_SIZE_LONG

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

def run_simulation_topology(simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
                           len_of_msa_upper_bound, len_of_msa_lower_bound, num_of_process,
                           distribution_of_internal_branch_length, distribution_of_external_branch_length,
                           range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
                           max_indel_length, verbose=False):
    """
    Run topology simulation using the simulate_topology.py script.

    Parameters:
    simulation_dir: Path to simulation directory containing code/
    num_of_topology: Number of MSAs to simulate
    taxa_num: Number of taxa in final tree
    range_of_taxa_num: Range of taxa numbers for sampling [lower, upper]
    len_of_msa_upper_bound: Upper bound of MSA length
    len_of_msa_lower_bound: Lower bound of MSA length
    num_of_process: Number of processes for parallel execution
    distribution_of_internal_branch_length: Distribution parameters [type, param1, param2]
    distribution_of_external_branch_length: Distribution parameters [type, param1, param2]
    range_of_mean_pairwise_divergence: Range [min, max] for mean pairwise divergence
    range_of_indel_substitution_rate: Range [min, max] for indel substitution rate
    max_indel_length: Maximum indel length
    verbose: Whether to show verbose output
    """
    code_dir = Path(simulation_dir) / 'code'
    if not code_dir.exists():
        raise FileNotFoundError(f"Simulation code directory not found: {code_dir}")

    # Build command
    cmd = [
        'python', 'simulate_topology.py',
        '--num_of_topology', str(num_of_topology),
        '--taxa_num', str(taxa_num),
        '--range_of_taxa_num', str(range_of_taxa_num),
        '--len_of_msa_upper_bound', str(len_of_msa_upper_bound),
        '--len_of_msa_lower_bound', str(len_of_msa_lower_bound),
        '--num_of_process', str(num_of_process),
        '--distribution_of_internal_branch_length', str(distribution_of_internal_branch_length),
        '--distribution_of_external_branch_length', str(distribution_of_external_branch_length),
        '--range_of_mean_pairwise_divergence', str(range_of_mean_pairwise_divergence),
        '--range_of_indel_substitution_rate', str(range_of_indel_substitution_rate),
        '--max_indel_length', str(max_indel_length)
    ]

    if verbose:
        print(f"Running topology simulation: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(code_dir), capture_output=not verbose, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Topology simulation failed: {result.stderr}")

    return result


def run_indelible(simulation_dir, verbose=False):
    """
    Run INDELible to generate sequence alignments.

    Parameters:
    simulation_dir: Path to simulation directory
    verbose: Whether to show verbose output
    """
    simulate_data_dir = Path(simulation_dir) / 'simulate_data'
    indelible_path = Path(simulation_dir) / 'indelible'

    if not simulate_data_dir.exists():
        raise FileNotFoundError(f"Simulate data directory not found: {simulate_data_dir}")

    if not indelible_path.exists():
        raise FileNotFoundError(f"INDELible executable not found: {indelible_path}")

    # Make indelible executable
    indelible_path.chmod(0o777)

    # Copy indelible to simulate_data if not already there
    indelible_dest = simulate_data_dir / 'indelible'
    if not indelible_dest.exists():
        shutil.copy(indelible_path, indelible_dest)
        indelible_dest.chmod(0o777)

    if verbose:
        print("Running INDELible...")

    result = subprocess.run('./indelible', cwd=str(simulate_data_dir), capture_output=not verbose, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"INDELible execution failed: {result.stderr}")

    return result


def extract_fasta_data(simulation_dir, max_length=None, verbose=False):
    """
    Extract FASTA data from simulation output.

    Parameters:
    simulation_dir: Path to simulation directory
    max_length: Maximum MSA length to keep (None for no limit)
    verbose: Whether to show verbose output
    """
    code_dir = Path(simulation_dir) / 'code'
    if not code_dir.exists():
        raise FileNotFoundError(f"Simulation code directory not found: {code_dir}")

    cmd = ['python', 'extract_fasta_data.py']
    if max_length is not None:
        cmd.extend(['--length', str(max_length)])

    if verbose:
        print(f"Extracting FASTA data: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(code_dir), capture_output=not verbose, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FASTA extraction failed: {result.stderr}")

    return result


def generate_numpy_data(simulation_dir, verbose=False):
    """
    Generate numpy training data from FASTA files.

    Parameters:
    simulation_dir: Path to simulation directory
    verbose: Whether to show verbose output
    """
    code_dir = Path(simulation_dir) / 'code'
    if not code_dir.exists():
        raise FileNotFoundError(f"Simulation code directory not found: {code_dir}")

    # Copy trees.txt if it exists
    trees_src = Path(simulation_dir) / 'simulate_data' / 'trees.txt'
    trees_dest = Path(simulation_dir) / 'label_file' / 'trees.txt'
    if trees_src.exists():
        trees_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(trees_src, trees_dest)

    cmd = ['python', 'gen_numpy.py']

    if verbose:
        print(f"Generating numpy data: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(code_dir), capture_output=not verbose, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Numpy data generation failed: {result.stderr}")

    return result


def run_full_simulation(simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
                       len_of_msa_upper_bound, len_of_msa_lower_bound, num_of_process,
                       distribution_of_internal_branch_length, distribution_of_external_branch_length,
                       range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
                       max_indel_length, max_length=None, cleanup=True, verbose=False,
                       run_inference=False, evaluate_results=False, output_dir=None,
                       sequence_type='standard', branch_model='gamma', beam_size=1, window_coverage=1):
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
    if verbose:
        print("Starting full simulation pipeline...")

    # Step 1: Generate topologies
    if verbose:
        print("Step 1: Generating topologies...")
    run_simulation_topology(
        simulation_dir, num_of_topology, taxa_num, range_of_taxa_num,
        len_of_msa_upper_bound, len_of_msa_lower_bound, num_of_process,
        distribution_of_internal_branch_length, distribution_of_external_branch_length,
        range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
        max_indel_length, verbose=verbose
    )

    # Step 2: Run INDELible
    if verbose:
        print("Step 2: Running INDELible...")
    run_indelible(simulation_dir, verbose=verbose)

    # Step 3: Extract FASTA data
    if verbose:
        print("Step 3: Extracting FASTA data...")
    extract_fasta_data(simulation_dir, max_length=max_length, verbose=verbose)

    # Step 4: Copy trees.txt
    trees_src = Path(simulation_dir) / 'simulate_data' / 'trees.txt'
    trees_dest = Path(simulation_dir) / 'label_file' / 'trees.txt'
    if trees_src.exists():
        trees_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(trees_src, trees_dest)

    # Step 5: Generate numpy data
    if verbose:
        print("Step 4: Generating numpy training data...")
    generate_numpy_data(simulation_dir, verbose=verbose)

    # Step 6: Run inference and evaluation if requested
    evaluation_results = None
    if run_inference or (evaluate_results is True):
        if verbose:
            print("Step 5: Running inference and evaluation...")

        fasta_dir = Path(simulation_dir) / 'fasta_file'
        trees_file = Path(simulation_dir) / 'label_file' / 'trees.txt'

        if not trees_file.exists():
            # Try to get from simulate_data if cleanup hasn't happened yet
            trees_file_alt = Path(simulation_dir) / 'simulate_data' / 'trees.txt'
            if trees_file_alt.exists():
                trees_file = trees_file_alt

        if evaluate_results:
            evaluation_results = evaluate_simulation_results(
                str(fasta_dir), str(trees_file), output_dir=output_dir,
                sequence_type=sequence_type, branch_model=branch_model,
                beam_size=beam_size, window_coverage=window_coverage, verbose=verbose
            )

            if verbose:
                stats = evaluation_results['statistics']
                print(f"\nEvaluation Results:")
                print(f"  Total files: {stats['total_files']}")
                print(f"  Successful: {stats['successful_inferences']}")
                print(f"  Failed: {stats['failed_inferences']}")
                if stats['successful_inferences'] > 0:
                    print(f"  Mean RF distance: {stats['mean_rf_distance']:.4f} ± {stats['std_rf_distance']:.4f}")
                    print(f"  Mean normalized RF: {stats['mean_normalized_rf']:.4f} ± {stats['std_normalized_rf']:.4f}")

    # Step 7: Cleanup if requested
    if cleanup:
        simulate_data_dir = Path(simulation_dir) / 'simulate_data'
        if simulate_data_dir.exists():
            if verbose:
                print("Step 6: Cleaning up temporary files...")
            shutil.rmtree(simulate_data_dir)

    if verbose:
        print("Simulation pipeline completed successfully!")

    sim_path = Path(simulation_dir)
    result = {
        'numpy_seq_dir': str(sim_path / 'numpy_file' / 'seq'),
        'numpy_label_dir': str(sim_path / 'numpy_file' / 'label'),
        'fasta_dir': str(sim_path / 'fasta_file'),
        'trees_file': str(sim_path / 'label_file' / 'trees.txt')
    }

    if evaluation_results is not None:
        result['evaluation'] = evaluation_results

    return result


# ============================================================================
# Inference and Evaluation Functions
# ============================================================================

def read_true_trees(trees_file):
    """
    Read true trees from trees.txt file generated by INDELible.

    Parameters:
    trees_file: Path to trees.txt file

    Returns:
    Dictionary mapping file names to true tree Newick strings
    """
    trees_path = Path(trees_file)
    if not trees_path.exists():
        raise FileNotFoundError(f"Trees file not found: {trees_file}")

    # Read trees.txt (skip first 5 lines, tab-separated, column 8 contains tree)
    csv_data = pd.read_table(trees_path, skiprows=5, sep='\t', header=None)
    file_names = list(csv_data[0])
    true_trees = list(csv_data[8])

    tree_dict = {}
    for i in range(len(file_names)):
        tree_dict[file_names[i]] = true_trees[i].strip()

    return tree_dict


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


def evaluate_simulation_results(fasta_dir, trees_file, output_dir=None,
                                sequence_type='standard', branch_model='gamma',
                                beam_size=1, window_coverage=1, verbose=False):
    """
    Evaluate inference results on simulated data by comparing with true trees.

    Parameters:
    fasta_dir: Directory containing FASTA files
    trees_file: Path to trees.txt file with true trees
    output_dir: Directory to save inference results (None to skip saving)
    sequence_type: Sequence type for model selection
    branch_model: Branch length model for model selection
    beam_size: Beam search size
    window_coverage: Sliding window coverage factor
    verbose: Whether to show verbose output

    Returns:
    Dictionary containing evaluation results
    """
    # Read true trees
    if verbose:
        print("Reading true trees...")
    true_trees = read_true_trees(trees_file)

    # Get all FASTA files
    fasta_path = Path(fasta_dir)
    fasta_files = sorted(fasta_path.glob('*.fas'))
    fasta_files = [str(f) for f in fasta_files if 'TRUE' in f.name]

    if len(fasta_files) == 0:
        raise ValueError(f"No FASTA files found in {fasta_dir}")

    if verbose:
        print(f"Found {len(fasta_files)} FASTA files to process...")

    # Create output directory if needed
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        inferred_trees_file = output_path / 'inferred_trees.txt'
        results_file = output_path / 'evaluation_results.txt'

    results = []
    inferred_trees = {}

    # Process each FASTA file
    for i, fasta_file in enumerate(fasta_files):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(fasta_files)}...")

        # Get file name (without extension)
        file_name = Path(fasta_file).stem.replace('_TRUE', '')

        # Get true tree
        if file_name not in true_trees:
            if verbose:
                print(f"Warning: No true tree found for {file_name}, skipping...")
            continue

        true_tree_str = true_trees[file_name]

        try:
            # Infer tree
            inferred_tree_str = infer_tree_from_msa(
                fasta_file, sequence_type=sequence_type, branch_model=branch_model,
                beam_size=beam_size, window_coverage=window_coverage, verbose=False
            )

            # Calculate RF distance
            rf_distance, normalized_rf, max_rf = calculate_rf_distance(
                true_tree_str, inferred_tree_str, unrooted=True
            )

            if rf_distance is not None:
                result = {
                    'file_name': file_name,
                    'rf_distance': rf_distance,
                    'normalized_rf': normalized_rf,
                    'max_rf': max_rf,
                    'success': True
                }
                inferred_trees[file_name] = inferred_tree_str
            else:
                result = {
                    'file_name': file_name,
                    'rf_distance': None,
                    'normalized_rf': None,
                    'max_rf': None,
                    'success': False
                }

            results.append(result)

        except Exception as e:
            if verbose:
                print(f"Error processing {file_name}: {str(e)}")
            results.append({
                'file_name': file_name,
                'rf_distance': None,
                'normalized_rf': None,
                'max_rf': None,
                'success': False,
                'error': str(e)
            })

    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    if len(successful_results) > 0:
        rf_distances = [r['rf_distance'] for r in successful_results]
        normalized_rfs = [r['normalized_rf'] for r in successful_results]

        stats = {
            'total_files': len(results),
            'successful_inferences': len(successful_results),
            'failed_inferences': len(results) - len(successful_results),
            'mean_rf_distance': np.mean(rf_distances),
            'std_rf_distance': np.std(rf_distances),
            'min_rf_distance': np.min(rf_distances),
            'max_rf_distance': np.max(rf_distances),
            'mean_normalized_rf': np.mean(normalized_rfs),
            'std_normalized_rf': np.std(normalized_rfs),
            'min_normalized_rf': np.min(normalized_rfs),
            'max_normalized_rf': np.max(normalized_rfs)
        }
    else:
        stats = {
            'total_files': len(results),
            'successful_inferences': 0,
            'failed_inferences': len(results),
            'mean_rf_distance': None,
            'std_rf_distance': None,
            'min_rf_distance': None,
            'max_rf_distance': None,
            'mean_normalized_rf': None,
            'std_normalized_rf': None,
            'min_normalized_rf': None,
            'max_normalized_rf': None
        }

    # Save results if output directory provided
    if output_dir is not None:
        # Save inferred trees
        with open(inferred_trees_file, 'w') as f:
            for file_name, tree_str in inferred_trees.items():
                f.write(f"{file_name}\t{tree_str}\n")

        # Save evaluation results
        with open(results_file, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total files processed: {stats['total_files']}\n")
            f.write(f"Successful inferences: {stats['successful_inferences']}\n")
            f.write(f"Failed inferences: {stats['failed_inferences']}\n\n")

            if stats['successful_inferences'] > 0:
                f.write("RF Distance Statistics:\n")
                f.write(f"  Mean: {stats['mean_rf_distance']:.4f}\n")
                f.write(f"  Std:  {stats['std_rf_distance']:.4f}\n")
                f.write(f"  Min:  {stats['min_rf_distance']:.0f}\n")
                f.write(f"  Max:  {stats['max_rf_distance']:.0f}\n\n")

                f.write("Normalized RF Distance Statistics:\n")
                f.write(f"  Mean: {stats['mean_normalized_rf']:.4f}\n")
                f.write(f"  Std:  {stats['std_normalized_rf']:.4f}\n")
                f.write(f"  Min:  {stats['min_normalized_rf']:.4f}\n")
                f.write(f"  Max:  {stats['max_normalized_rf']:.4f}\n\n")

            f.write("\nDetailed Results:\n")
            f.write("-" * 50 + "\n")
            for result in results:
                f.write(f"{result['file_name']}\t")
                if result['success']:
                    f.write(f"RF={result['rf_distance']}\tNormRF={result['normalized_rf']:.4f}\n")
                else:
                    f.write(f"FAILED")
                    if 'error' in result:
                        f.write(f": {result['error']}")
                    f.write("\n")

    return {
        'results': results,
        'statistics': stats,
        'inferred_trees': inferred_trees
    }


# ============================================================================
# Model Training Functions
# ============================================================================

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
        print(f"Loading {len(seq_files)} data files...")

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
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32,
                learning_rate=0.001, model_save_path=None, verbose=1):
    """
    Train a deep learning model.

    Parameters:
    model: Keras model to train
    X_train: Training sequences
    y_train: Training labels
    X_val: Validation sequences
    y_val: Validation labels
    epochs: Number of training epochs
    batch_size: Batch size for training
    learning_rate: Learning rate
    model_save_path: Path to save trained model (None to not save)
    verbose: Verbosity level (0=silent, 1=progress, 2=one line per epoch)

    Returns:
    Training history
    """
    # Compile model
    _, _, _, optimizers = _import_tensorflow()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # Save model if path provided
    if model_save_path is not None:
        model_path = Path(model_save_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(str(model_path))
        if verbose >= 1:
            print(f"Model saved to {model_save_path}")

    return history


def train_fusang_model(numpy_seq_dir, numpy_label_dir, window_size=240, epochs=50,
                       batch_size=32, learning_rate=0.001, train_ratio=0.8, val_ratio=0.1,
                       model_save_path=None, verbose=1):
    """
    Train a Fusang model from numpy data.

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

    Returns:
    Trained model and training history
    """
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_training_data(
        numpy_seq_dir, numpy_label_dir, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio, verbose=verbose >= 1
    )

    # Create model based on window size
    if window_size == 240:
        model = get_dl_model_240()
    elif window_size == 1200:
        model = get_dl_model_1200()
    else:
        raise ValueError(f"Unsupported window size: {window_size}. Must be 240 or 1200.")

    # Train model
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        model_save_path=model_save_path, verbose=verbose
    )

    # Evaluate on test set
    if verbose >= 1:
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")

    return model, history


# ============================================================================
# Command Line Interface
# ============================================================================

def _add_inference_args(parser):
    """Add inference arguments to a parser (shared between infer subcommand and legacy mode).
    
    Avoids using argument groups to prevent deprecation warnings in newer argparse versions.
    """
    # Add directly to parser (avoids deprecation warning with nested argument groups)
    parser.add_argument("msa_file", nargs='?', type=str, metavar='MSA_FILE', help="Input MSA file (positional)")
    parser.add_argument("output_file", nargs='?', type=str, metavar='OUTPUT_FILE', help="Output tree file in Newick format (positional, optional; if not specified, output to stdout)")
    parser.add_argument("-i", "--input", dest="msa_dir", type=str, help="Input MSA file path")
    parser.add_argument("-m", "--msa_dir", type=str, help=argparse.SUPPRESS)  # Hidden alias for backward compatibility
    parser.add_argument("-o", "--output", type=str, help="Output tree file path (Newick format)")
    parser.add_argument("-s", "--save_prefix", type=str, help="Output file prefix (saved to ./dl_output/<prefix>.txt)")
    parser.add_argument("-b", "--beam_size", type=str, default='1', help="Beam search size (default: 1)")
    parser.add_argument("-t", "--sequence_type", type=str, default='standard', choices=['standard', 'coding', 'noncoding'], help="Sequence type for model selection (default: standard)")
    parser.add_argument("-r", "--branch_model", type=str, default='gamma', choices=['gamma', 'uniform'], help="Branch length model (default: gamma)")
    parser.add_argument("-w", "--window_coverage", type=str, default='1', help="Sliding window coverage factor (default: 1)")


def _extract_inference_args(args):
    """Extract inference parameters from args namespace."""
    msa_file = getattr(args, 'msa_dir', None) or getattr(args, 'input', None) or getattr(args, 'msa_file', None)
    output_path = getattr(args, 'output', None) or getattr(args, 'output_file', None)
    save_prefix = getattr(args, 'save_prefix', None)
    beam_size = int(getattr(args, 'beam_size', '1'))
    sequence_type = getattr(args, 'sequence_type', 'standard')
    branch_model = getattr(args, 'branch_model', 'gamma')
    window_coverage = float(getattr(args, 'window_coverage', '1'))
    return msa_file, output_path, save_prefix, beam_size, sequence_type, branch_model, window_coverage


def _is_file_path(arg):
    """Check if argument looks like a file path."""
    if not arg or arg.startswith('-'):
        return False
    path = Path(arg)
    return (path.exists() or '/' in arg or '\\' in arg or
            arg.endswith(('.fas', '.fasta', '.fa', '.phy', '.phylip')))


if __name__ == '__main__':
    # Determine if first argument is a subcommand or a file path
    first_arg = sys.argv[1] if len(sys.argv) > 1 else None
    has_subcommand = first_arg in ['simulate', 'train', 'infer', 'evaluate']
    is_file_path = _is_file_path(first_arg) if first_arg else False
    show_full_help = '-h' in sys.argv or '--help' in sys.argv

    def print_simplified_help():
        print("Fusang: Deep learning-based phylogenetic tree inference")
        print("\nusage: fusang [-h] <subcommand> ...\n")
        print("Subcommands:")
        print("  infer      Infer phylogenetic tree from MSA")
        print("  simulate   Generate simulation data for training")
        print("  train      Train a Fusang model")
        print("  evaluate   Evaluate inference results on simulated data")
        print("\nLegacy mode (backward compatible):")
        print("  fusang input.fasta [output.tree]")
        print("  (if output.tree is not specified, output goes to stdout)")
        print("\nUse 'fusang --help' for detailed help, or 'fusang <subcommand> --help' for subcommand help.")

    parser = argparse.ArgumentParser(
        'fusang',
        description='Fusang: Deep learning-based phylogenetic tree inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Infer phylogenetic tree (using subcommand):
  fusang infer -i input.fasta -o output.tree

  # Infer phylogenetic tree (legacy mode, backward compatible):
  fusang input.fasta output.tree

  # Generate simulation data:
  fusang simulate --simulation_dir ./simulation --num_of_topology 20 --taxa_num 5 ...

  # Train a model:
  fusang train --numpy_seq_dir ./data/seq --numpy_label_dir ./data/label --model_save_path ./model.h5 ...

  # Evaluate inference results:
  fusang evaluate --fasta_dir ./fasta --trees_file ./trees.txt --output_dir ./results ...

For detailed help on a subcommand, use: fusang <subcommand> --help
        '''
    )

    # Global arguments
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress warning messages")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output including stderr messages")

    # Always create subparsers to show them in help (but not required for backward compatibility)
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', metavar='{infer,simulate,train,evaluate}')
    # If first argument looks like a file path, make subcommand optional
    if is_file_path:
        subparsers.required = False

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
    sim_parser.add_argument("--simulation_dir", type=str, required=True, help="Path to simulation directory containing code/ and indelible")
    sim_parser.add_argument("--num_of_topology", type=int, required=True, help="Number of MSAs to simulate")
    sim_parser.add_argument("--taxa_num", type=int, required=True, help="Number of taxa in final tree")
    sim_parser.add_argument("--range_of_taxa_num", type=str, required=True, help="Range of taxa numbers [lower, upper] (e.g., '[5, 40]')")
    sim_parser.add_argument("--len_of_msa_upper_bound", type=int, required=True, help="Upper bound of MSA length")
    sim_parser.add_argument("--len_of_msa_lower_bound", type=int, required=True, help="Lower bound of MSA length")
    sim_parser.add_argument("--num_of_process", type=int, default=24, help="Number of processes for parallel execution (default: 24)")
    sim_parser.add_argument("--distribution_of_internal_branch_length", type=str, default='[1, 0.5, 0.3]', help="Internal branch length distribution [type, param1, param2] (default: '[1, 0.5, 0.3]')")
    sim_parser.add_argument("--distribution_of_external_branch_length", type=str, default='[1, 0.5, 0.3]', help="External branch length distribution [type, param1, param2] (default: '[1, 0.5, 0.3]')")
    sim_parser.add_argument("--range_of_mean_pairwise_divergence", type=str, default='[0.03, 0.3]', help="Range of mean pairwise divergence [min, max] (default: '[0.03, 0.3]')")
    sim_parser.add_argument("--range_of_indel_substitution_rate", type=str, default='[0.01, 0.25]', help="Range of indel substitution rate [min, max] (default: '[0.01, 0.25]')")
    sim_parser.add_argument("--max_indel_length", type=int, required=True, help="Maximum indel length")
    sim_parser.add_argument("--max_length", type=int, default=None, help="Maximum MSA length to keep (default: no limit)")
    sim_parser.add_argument("--no-cleanup", action="store_true", help="Do not remove simulate_data directory after completion")
    sim_parser.add_argument("--evaluate", action="store_true", help="Run inference and evaluation on generated data")
    sim_parser.add_argument("--evaluation_output", type=str, default=None, help="Directory to save evaluation results (default: simulation_dir/evaluation)")
    sim_parser.add_argument("--sequence_type", type=str, default='standard', help="Sequence type for inference: standard (default), coding, or noncoding")
    sim_parser.add_argument("--branch_model", type=str, default='gamma', help="Branch length model for inference: gamma (default) or uniform")
    sim_parser.add_argument("--beam_size", type=int, default=1, help="Beam search size for inference (default: 1)")
    sim_parser.add_argument("--window_coverage", type=float, default=1.0, help="Sliding window coverage factor for inference (default: 1.0)")

    # Evaluation mode (standalone)
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate inference results on simulated data',
        description='Evaluate inference accuracy by comparing predicted trees with true trees using RF distance.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    eval_parser.add_argument("--fasta_dir", type=str, required=True, help="Directory containing FASTA files")
    eval_parser.add_argument("--trees_file", type=str, required=True, help="Path to trees.txt file with true trees")
    eval_parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results (default: current directory)")
    eval_parser.add_argument("--sequence_type", type=str, default='standard', help="Sequence type for inference: standard (default), coding, or noncoding")
    eval_parser.add_argument("--branch_model", type=str, default='gamma', help="Branch length model for inference: gamma (default) or uniform")
    eval_parser.add_argument("--beam_size", type=int, default=1, help="Beam search size for inference (default: 1)")
    eval_parser.add_argument("--window_coverage", type=float, default=1.0, help="Sliding window coverage factor for inference (default: 1.0)")

    # Training mode
    train_parser = subparsers.add_parser(
        'train',
        help='Train a Fusang model',
        description='Train deep learning models for quartet topology prediction.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    train_parser.add_argument("--numpy_seq_dir", type=str, required=True, help="Directory containing sequence numpy files")
    train_parser.add_argument("--numpy_label_dir", type=str, required=True, help="Directory containing label numpy files")
    train_parser.add_argument("--window_size", type=int, choices=[240, 1200], default=240, help="Window size for model (default: 240)")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    train_parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    train_parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of data for training (default: 0.8)")
    train_parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data for validation (default: 0.1)")
    train_parser.add_argument("--model_save_path", type=str, required=True, help="Path to save trained model weights (.h5 file)")

    # Backward compatibility: add inference arguments to main parser for old-style usage
    if not has_subcommand:
        # Add directly to parser to avoid deprecation warnings
        _add_inference_args(parser)

    # Parse arguments
    # If first argument is a file path, handle as legacy mode directly (before argparse tries to parse it)
    if is_file_path:
        # Create args namespace for legacy mode without parsing
        args = argparse.Namespace()
        args.mode = None
        args.msa_dir = None
        args.input = None
        args.msa_file = sys.argv[1]
        args.output = None
        args.output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else None
        args.save_prefix = None
        args.beam_size = '1'
        args.sequence_type = 'standard'
        args.branch_model = 'gamma'
        args.window_coverage = '1'
        args.quiet = '-q' in sys.argv or '--quiet' in sys.argv
        args.verbose = '-v' in sys.argv or '--verbose' in sys.argv
    else:
        try:
            args = parser.parse_args()
        except SystemExit as e:
            if show_full_help:
                sys.exit(0)
            if e.code != 0 and not has_subcommand:
                print_simplified_help()
                sys.exit(2)
            sys.exit(e.code if e.code else 2)

    # Validate that quiet and verbose are not both specified
    if hasattr(args, 'quiet') and hasattr(args, 'verbose') and args.quiet and args.verbose:
        parser.error("Cannot specify both -q/--quiet and -v/--verbose")

    # Determine verbosity
    verbose = args.verbose and not args.quiet

    # Use 'with' statement to ensure stderr is properly restored
    with _stderr_redirector:
        # Handle backward compatibility: if no mode but positional args exist
        if args.mode is None:
            if is_file_path or (len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in ['simulate', 'train', 'infer', 'evaluate']):
                # Legacy mode: ensure mode is set
                args.mode = 'infer'
            else:
                # No args: show help
                if show_full_help:
                    parser.print_help()
                else:
                    print_simplified_help()
                sys.exit(0)

        # Handle different modes
        if args.mode == 'simulate':
            # Simulation mode
            range_of_taxa_num = list(eval(args.range_of_taxa_num))
            distribution_of_internal_branch_length = list(eval(args.distribution_of_internal_branch_length))
            distribution_of_external_branch_length = list(eval(args.distribution_of_external_branch_length))
            range_of_mean_pairwise_divergence = list(eval(args.range_of_mean_pairwise_divergence))
            range_of_indel_substitution_rate = list(eval(args.range_of_indel_substitution_rate))

            # Determine evaluation output directory
            eval_output_dir = args.evaluation_output
            if eval_output_dir is None and args.evaluate:
                eval_output_dir = str(Path(args.simulation_dir) / 'evaluation')

            result = run_full_simulation(
                args.simulation_dir, args.num_of_topology, args.taxa_num, range_of_taxa_num,
                args.len_of_msa_upper_bound, args.len_of_msa_lower_bound, args.num_of_process,
                distribution_of_internal_branch_length, distribution_of_external_branch_length,
                range_of_mean_pairwise_divergence, range_of_indel_substitution_rate,
                args.max_indel_length, max_length=args.max_length, cleanup=not args.no_cleanup,
                verbose=verbose, run_inference=args.evaluate, evaluate_results=args.evaluate,
                output_dir=eval_output_dir, sequence_type=args.sequence_type,
                branch_model=args.branch_model, beam_size=args.beam_size,
                window_coverage=args.window_coverage
            )

            if verbose:
                print(f"\nSimulation completed. Data saved to:")
                print(f"  Sequences: {result['numpy_seq_dir']}")
                print(f"  Labels: {result['numpy_label_dir']}")
                print(f"  FASTA: {result['fasta_dir']}")
                if 'evaluation' in result:
                    stats = result['evaluation']['statistics']
                    print(f"\nEvaluation results saved to: {eval_output_dir}")

        elif args.mode == 'evaluate':
            # Evaluation mode (standalone)
            output_dir = args.output_dir
            if output_dir is None:
                output_dir = os.getcwd()

            evaluation_results = evaluate_simulation_results(
                args.fasta_dir, args.trees_file, output_dir=output_dir,
                sequence_type=args.sequence_type, branch_model=args.branch_model,
                beam_size=args.beam_size, window_coverage=args.window_coverage, verbose=verbose
            )

            if verbose:
                stats = evaluation_results['statistics']
                print(f"\nEvaluation Results:")
                print(f"  Total files: {stats['total_files']}")
                print(f"  Successful: {stats['successful_inferences']}")
                print(f"  Failed: {stats['failed_inferences']}")
                if stats['successful_inferences'] > 0:
                    print(f"  Mean RF distance: {stats['mean_rf_distance']:.4f} ± {stats['std_rf_distance']:.4f}")
                    print(f"  Mean normalized RF: {stats['mean_normalized_rf']:.4f} ± {stats['std_normalized_rf']:.4f}")
                print(f"\nResults saved to: {output_dir}")

        elif hasattr(args, 'mode') and args.mode == 'train':
            # Training mode
            model, history = train_fusang_model(
                args.numpy_seq_dir, args.numpy_label_dir,
                window_size=args.window_size, epochs=args.epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                model_save_path=args.model_save_path, verbose=2 if verbose else 1
            )

            if verbose:
                print(f"\nTraining completed. Model saved to: {args.model_save_path}")

        else:
            # Inference mode (default, backward compatible)
            msa_file, output_path, save_prefix, beam_size, sequence_type, branch_model, window_coverage = _extract_inference_args(args)

            if not msa_file:
                parser.error("MSA file must be provided either as positional argument or with -m/--msa_dir or -i/--input")

            # Use the unified inference function
            searched_tree = infer_tree_from_msa(
                msa_file, sequence_type=sequence_type, branch_model=branch_model,
                beam_size=beam_size, window_coverage=window_coverage, verbose=verbose
            )

            # Write output
            write_output(searched_tree, output_path, save_prefix)
