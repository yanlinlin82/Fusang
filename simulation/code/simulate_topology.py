import random
import ete3
import scipy.stats
import re
import argparse
from pandas.core.frame import DataFrame
import numpy as np
from multiprocessing import Pool, cpu_count
import multiprocessing

def _get_extremes(tree):
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
        branch_ratio = ((internal_distance + max(sdis1, sdis2))
                        / min(ldis1, ldis2))
        if branch_ratio < min_branch_ratio:
            leaves = [short1, long1, short2, long2]
            min_branch_ratio = branch_ratio
    leaves = [tree & leaf.name for leaf in leaves]
    return min_branch_ratio, leaves


def gen_newick(q, seed, taxa_num, range_of_taxa_num, distribution_of_internal_branch_length,
    distribution_of_external_branch_length, range_of_mean_pairwise_divergence):

    random.seed(seed)
    np.random.seed(seed)

    taxon_count_model = scipy.stats.uniform(range_of_taxa_num[0], range_of_taxa_num[1])
    tree = ete3.PhyloTree()
    tree.populate(int(taxon_count_model.rvs()), random_branches=True)
    current_internal_index = 0
    current_leaf_index = 0

    if distribution_of_internal_branch_length[0] == 1:
        internal_branch_model = scipy.stats.gamma(a=distribution_of_internal_branch_length[1], scale=distribution_of_internal_branch_length[2])
    elif distribution_of_internal_branch_length[0] == 0:
        internal_branch_model = scipy.stats.uniform(distribution_of_internal_branch_length[1], distribution_of_internal_branch_length[2])

    if distribution_of_external_branch_length[0] == 1:
        external_branch_model = scipy.stats.gamma(a=distribution_of_external_branch_length[1], scale=distribution_of_external_branch_length[2])
    elif distribution_of_external_branch_length[0] == 0:
        external_branch_model = scipy.stats.uniform(distribution_of_external_branch_length[1], distribution_of_external_branch_length[2])

    expected_mean_pairwise_divergence = scipy.stats.uniform(range_of_mean_pairwise_divergence[0], range_of_mean_pairwise_divergence[1]).rvs()

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

    for i in range(0, len(total_tree_leaves)-1):
        for j in range(i+1, len(total_tree_leaves)):
            tmp_dist = tree.get_distance(total_tree_leaves[i], total_tree_leaves[j])
            if tmp_dist > 1e-4:
                pairwise_divergence_list.append(tmp_dist)

    mean_pairwise_divergence = np.mean(np.array(pairwise_divergence_list))

    scale_ratio = expected_mean_pairwise_divergence / mean_pairwise_divergence

    for node in tree.traverse("preorder"):
        node.dist = max(0.0001, node.dist*scale_ratio)
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

    for i in range(0,taxa_num-1):
        sample_number = random.choice(number_set)
        number_set.remove(sample_number)
        idx.append(str(sample_number))

    for i in range(0, len(match)):
        ans = ans.replace(match[i], idx[i]+':')

    q.put(ans)


parser = argparse.ArgumentParser('get_parameters_of_simulation')
p_input = parser.add_argument_group("INPUT")
p_input.add_argument("--seed", action="store", type=int, required=True)
p_input.add_argument("--num_of_topology", action="store", type=int, required=True)
p_input.add_argument("--taxa_num", action="store", type=int, required=True)
p_input.add_argument("--range_of_taxa_num", action="store", type=str, required=True)
p_input.add_argument("--num_of_process", action="store", type=int, required=True)
p_input.add_argument("--distribution_of_internal_branch_length", action="store", type=str, required=True)
p_input.add_argument("--distribution_of_external_branch_length", action="store", type=str, required=True)
p_input.add_argument("--range_of_mean_pairwise_divergence", action="store", type=str, required=True)
p_input.add_argument("--max_indel_length", action="store", type=int, required=True)
p_input.add_argument("--output_newick", action="store", type=str, required=True)


args = parser.parse_args()
seed = args.seed
num_of_topology = args.num_of_topology
taxa_num = args.taxa_num
range_of_taxa_num = list(eval(args.range_of_taxa_num))
num_of_process = args.num_of_process
distribution_of_internal_branch_length = list(eval(args.distribution_of_internal_branch_length))
distribution_of_external_branch_length = list(eval(args.distribution_of_external_branch_length))
range_of_mean_pairwise_divergence = list(eval(args.range_of_mean_pairwise_divergence))
max_indel_length = args.max_indel_length
output_newick = args.output_newick


q = multiprocessing.Manager().Queue()

if num_of_process <= 0:
    num_of_process = max(1, cpu_count() - 1)

para_list = [(q, seed + i, taxa_num, range_of_taxa_num, distribution_of_internal_branch_length,
    distribution_of_external_branch_length, range_of_mean_pairwise_divergence) for i in range(0, num_of_topology)]
pool = Pool(num_of_process)
pool.starmap(gen_newick, para_list)
pool.close()
pool.join()

csv_list = []
while not q.empty():
    tmp_topo = q.get()
    csv_list.append(tmp_topo)

csv_list.sort() # keep the same order for reproducibility

dictionary = {"newick" : csv_list}
data=DataFrame(dictionary)
data.to_csv(output_newick)

print(f"Generated {len(csv_list)} topologies and saved to '{output_newick}'.")
