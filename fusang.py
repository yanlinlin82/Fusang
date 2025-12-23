import os
import re
import sys
import warnings
from io import StringIO
import argparse
import math
import functools
from itertools import combinations
from multiprocessing import Pool
import multiprocessing

import numpy as np
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


# Set TensorFlow and NumPy warning levels based on verbosity
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

import tensorflow as tf
from keras import layers, models


def comb_math(n, m):
    """Calculate combination C(n, m) = n! / (m! * (n-m)!)."""
    return math.factorial(n) // (math.factorial(n - m) * math.factorial(m))


def nlargest_indices(arr, n):
    uniques = np.unique(arr)
    threshold = uniques[-n]
    return np.where(arr >= threshold)


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
    str_b = ''
    id_set = [chr(ord(u'\u4e00') + i) for i in range(taxa_num)]

    for char in str_a:
        if char in id_set:
            str_b += taxa_name[ord(char) - ord(u'\u4e00')]
        else:
            str_b += char

    return str_b


def cmp(a, b):
    """Compare function for sorting taxa names numerically."""
    a_int = int(a)
    b_int = int(b)
    if a_int < b_int:
        return -1
    if a_int > b_int:
        return 1
    return 0


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
    for taxa in sorted(list(fasta_dic.keys()), key=functools.cmp_to_key(cmp)):
        taxa_block.append(fasta_dic[taxa.strip()])

    return np.array([taxa_block])


def get_dl_model_1200():
    """
    Get the definition of DL model for sequences longer than 1200.
    This model aims to solve the default case for sequences with length larger than 1200.
    """
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


def write_output(searched_tree, output_path, save_prefix):
    """Write output tree to file or stdout."""
    # Ensure output ends with newline
    if not searched_tree.endswith('\n'):
        searched_tree += '\n'

    # Determine output destination
    if output_path is not None:
        output_file = output_path
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'a') as f:
            f.write(searched_tree)
    elif save_prefix is not None:
        output_file = f'./dl_output/{save_prefix}.txt'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'a') as f:
            f.write(searched_tree)
    else:
        # Default to stdout
        print(searched_tree, end='')


def load_dl_model(len_of_msa, sequence_type, branch_model):
    """
    Load deep learning model based on MSA length and parameters.
    Returns the loaded model and window size (240 or 1200).
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

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
    model_path = os.path.join(script_dir, 'model', model_filename)
    dl_model.load_weights(filepath=model_path)

    return dl_model, window_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser('get_msa_dir')
    p_input = parser.add_argument_group("INPUT")
    p_input.add_argument("msa_file", nargs='?', type=str, help="Input MSA file (positional argument)")
    p_input.add_argument("output_file", nargs='?', type=str, help="Output file in Newick format (positional argument)")
    p_input.add_argument("-m", "--msa_dir", action="store", type=str, required=False, help="Input MSA file (alternative to positional argument)")
    p_input.add_argument("-i", "--input", action="store", type=str, required=False, help="Input MSA file (alias for -m/--msa_dir)")
    p_input.add_argument("-s", "--save_prefix", action="store", type=str, required=False, default=None, help="Prefix of output file (output will be in Newick format)")
    p_input.add_argument("-o", "--output", action="store", type=str, required=False, default=None, help="Output file path (output will be in Newick format)")
    p_input.add_argument("-b", "--beam_size", action="store", type=str, default='1', required=False, help="Beam search size (default: 1)")
    p_input.add_argument("-t", "--sequence_type", action="store", type=str, default='standard', required=False, help="Sequence type for model selection: standard (default), coding, or noncoding")
    p_input.add_argument("-r", "--branch_model", action="store", type=str, default='gamma', required=False, help="Branch length model for model selection: gamma (default) or uniform")
    p_input.add_argument("-w", "--window_coverage", action="store", type=str, default='1', required=False, help="Sliding window coverage factor (default: 1)")
    p_input.add_argument("-q", "--quiet", action="store_true", help="Suppress warning messages")
    p_input.add_argument("-v", "--verbose", action="store_true", help="Show verbose output including stderr messages")

    args = parser.parse_args()

    # Validate that quiet and verbose are not both specified
    if args.quiet and args.verbose:
        parser.error("Cannot specify both -q/--quiet and -v/--verbose")

    # Determine verbosity level for model.predict() calls
    # 0 = silent, 1 = progress bar, 2 = one line per epoch (not used for predict)
    if args.quiet:
        predict_verbose = 0  # Suppress progress bars in quiet mode
    elif args.verbose:
        predict_verbose = 1  # Show progress bars in verbose mode
    else:
        predict_verbose = 1  # Default: show progress bars (warnings still suppressed via TF_CPP_MIN_LOG_LEVEL)

    # Determine MSA file: use -m/--msa_dir or -i/--input if provided, otherwise use positional argument
    if args.msa_dir is not None:
        msa_dir = args.msa_dir
    elif args.input is not None:
        msa_dir = args.input
    elif args.msa_file is not None:
        msa_dir = args.msa_file
    else:
        parser.error("MSA file must be provided either as positional argument or with -m/--msa_dir or -i/--input")

    # Determine output path: use -o/--output if provided, otherwise use positional output argument
    if args.output is not None:
        output_path = args.output
    elif args.output_file is not None:
        output_path = args.output_file
    else:
        output_path = None

    save_prefix = args.save_prefix
    beam_size = args.beam_size
    sequence_type = args.sequence_type
    branch_model = args.branch_model
    window_coverage = args.window_coverage

    # Use 'with' statement to ensure stderr is properly restored
    # The redirector was already entered at module level, so this will properly exit it
    with _stderr_redirector:
        # Parse MSA file
        save_alignment, taxa_name, len_of_msa, taxa_num = parse_msa_file(msa_dir)

        # Initialize quartet data structures
        (start_end_list, comb_of_id, leave_node_name, leave_node_comb_name,
         dic_for_leave_node_comb_name, internal_node_name_pool) = initialize_quartet_data(taxa_num)

        # Convert alignment to numpy array
        org_seq = get_numpy(save_alignment)

        # Load deep learning model based on MSA length and parameters
        dl_model, window_size = load_dl_model(len_of_msa, sequence_type, branch_model)
        window_number = int(len_of_msa * float(window_coverage) // window_size + 1)

        # Generate predictions
        dl_predict = np.zeros((len(comb_of_id), 3))
        fill_dl_predict(window_number, window_size, len_of_msa, comb_of_id, org_seq,
                       dl_model, dl_predict, verbose=predict_verbose)
        dl_predict /= window_number

        # Generate phylogenetic tree in Newick format
        use_masking = taxa_num > MIN_TAXA_FOR_MASKING
        searched_tree_str = gen_phylogenetic_tree(
            dl_predict, int(beam_size), taxa_num, leave_node_comb_name,
            dic_for_leave_node_comb_name, internal_node_name_pool,
            use_masking=use_masking, start_end_list=start_end_list, comb_of_id=comb_of_id
        )
        searched_tree = transform_str(searched_tree_str, taxa_name, taxa_num)

        # Write output
        write_output(searched_tree, output_path, save_prefix)
