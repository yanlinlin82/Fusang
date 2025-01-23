import sys
import numpy as np
from multiprocessing import Pool, cpu_count

def format_num(x, digits=5):
    """Format number with fixed decimal places"""
    if hasattr(x, 'ndim') and x.ndim > 0:
        x = x.item()
    return f"{float(x):.{digits}f}"

def process_single_model(args):
    """Process a single model configuration"""
    i, model, max_indel_length, indel_rate_bounds = args
    model_orig = model
    model_name = f"{model}Model{i}"

    # Generate parameters
    I = np.random.uniform(0, 1, size=1)
    A = np.random.uniform(0, 5, size=1)
    Pi = np.random.dirichlet([5,5,5,5], size=1)[0]  # Nucleotide proportions
    Pi = [format_num(p, 5) for p in Pi]

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
        params = [format_num(np.random.uniform(0, 3), 2) for _ in range(n_params)]
        submodel = f"{model} {' '.join(params)}"
    else:
        submodel = model

    output_lines.extend([
        f' [submodel] {submodel}',
        f' [rates] {format_num(I, 2)} {format_num(A, 2)} 0',
        f' [indelmodel] POW 1.5 {max_indel_length}',
        f' [indelrate] {format_num(np.random.uniform(*indel_rate_bounds), 5)}'
    ])

    if model_orig in ['F81', 'HKY', 'TrN', 'TIM', 'TVM', 'GTR']:
        output_lines.append(f' [statefreq] {" ".join(Pi)}')

    return {'lines': output_lines, 'name': model_name}

def model_gen(modelset, file_path, max_indel_length,
              indel_substitution_rate_lower_bound,
              indel_substitution_rate_upper_bound, n_process):
    """Generate model configurations in parallel"""
    # Set random seed
    np.random.seed(None)  # Ensure different random seeds for each process
    
    # Prepare parameters for parallel processing
    indel_rate_bounds = (indel_substitution_rate_lower_bound,
                        indel_substitution_rate_upper_bound)
    args = [(i, model, max_indel_length, indel_rate_bounds)
            for i, model in enumerate(modelset, 1)]

    # Use process pool for parallel processing
    if n_process <= 0:
        n_cores = max(1, cpu_count() - 1)  # Reserve one core for system
    else:
        n_cores = n_process
    with Pool(n_cores) as pool:
        results = pool.map(process_single_model, args)

    # Write to file
    with open(file_path, 'a') as f:
        f.write("\n")
        for result in results:
            f.write('\n'.join(result['lines']) + '\n')

    # Return list of model names
    return [result['name'] for result in results]

def indelib_gen(n_taxa, n_sim, n_process, len_of_msa_bounds, indel_rate_bounds,
                max_indel_length, in_newick, out_control):
    """Main function for generating control file"""
    # Write basic settings
    with open(out_control, 'w') as f:
        f.write("[TYPE] NUCLEOTIDE 2\n")
        f.write("\n")
        f.write("[SETTINGS]\n")
        f.write(" [output] FASTA\n")
        f.write(f" [randomseed] {np.random.randint(1, 100000)}\n")

    # Generate models
    possible_models = ['JC', 'TIM', 'TIMef', 'GTR', 'UNREST']
    modelset = np.random.choice(possible_models, size=n_sim, replace=True)

    model_names = model_gen(modelset, out_control, max_indel_length,
                            indel_rate_bounds[0], indel_rate_bounds[1], n_process)

    # Read Newick file
    import pandas as pd
    newick_data = pd.read_csv(in_newick)
    newick = newick_data.iloc[:, 1]

    # Generate tree IDs
    tree_ids = [f"t_sim{i}" for i in range(1, n_sim + 1)]

    # Write TREE block
    with open(out_control, 'a') as f:
        f.write("\n")
        for tid, nw in zip(tree_ids, newick):
            f.write(f"[TREE] {tid} {nw}\n")

    # Write PARTITIONS block
    partition_names = [f"p{i}" for i in range(1, n_sim + 1)]
    print(f"len_of_msa_bounds = {len_of_msa_bounds}")
    if len_of_msa_bounds[0] == len_of_msa_bounds[1]:
        msa_lengths = np.repeat(len_of_msa_bounds[0], n_sim)
    else:
        msa_lengths = np.random.randint(len_of_msa_bounds[0], len_of_msa_bounds[1], size=n_sim)

    with open(out_control, 'a') as f:
        f.write("\n")
        for pname, tid, mname, length in zip(partition_names, tree_ids,
                                           model_names, msa_lengths):
            f.write(f"[PARTITIONS] {pname} [ {tid} {mname} {length} ]\n")

    # Write EVOLVE block
    with open(out_control, 'a') as f:
        f.write("\n[EVOLVE]\n")
        for pname, tid in zip(partition_names, tree_ids):
            f.write(f"{pname} 1 {tid}\n")

if __name__ == "__main__":
    if len(sys.argv) != 12:
        print("Usage: python gen_control_file.py seed n_taxa n_sim n_process "
              "len_of_msa_lower len_of_msa_upper "
              "indel_rate_lower indel_rate_upper "
              "max_indel_length in.newick.csv out.control.txt")
        sys.exit(1)

    # Parse arguments
    seed = int(sys.argv[1])
    n_taxa = int(sys.argv[2])
    n_sim = int(sys.argv[3])
    n_process = int(sys.argv[4])
    len_of_msa_bounds = (int(sys.argv[5]), int(sys.argv[6]))
    indel_rate_bounds = (float(sys.argv[7]), float(sys.argv[8]))
    max_indel_length = int(sys.argv[9])
    in_newick = sys.argv[10]
    out_control = sys.argv[11]

    # Set random seed
    np.random.seed(seed)

    # Run main function
    indelib_gen(n_taxa, n_sim, n_process, len_of_msa_bounds, indel_rate_bounds,
                max_indel_length, in_newick, out_control)
