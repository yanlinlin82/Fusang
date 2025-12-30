#!/bin/bash

# Script to generate simulation dataset using fusang.py
# Usage: ./gen-dataset.sh <out-dir> <seed> <num_of_topology> <taxa_num> <range_of_taxa_num> \
#        <len_of_msa_lower_bound> <len_of_msa_upper_bound> <num_of_process> \
#        <distribution_of_internal_branch_length> <distribution_of_external_branch_length> \
#        <range_of_mean_pairwise_divergence> \
#        <indel_substitution_rate_lower_bound> <indel_substitution_rate_upper_bound> \
#        <max_indel_length>

if [ -z "${14}" ]; then
    echo
    echo "Usage: $0 <out-dir> <seed> \\"
    echo "          <num_of_topology> <taxa_num> <range_of_taxa_num> \\"
    echo "          <len_of_msa_lower_bound> <len_of_msa_upper_bound> <num_of_process> \\"
    echo "          <distribution_of_internal_branch_length> <distribution_of_external_branch_length> \\"
    echo "          <range_of_mean_pairwise_divergence> \\"
    echo "          <indel_substitution_rate_lower_bound> <indel_substitution_rate_upper_bound> \\"
    echo "          <max_indel_length>"
    echo
    echo "Example: $0 out/ 42 600 5 '[5, 40]' \\"
    echo "            200 200 24 \\"
    echo "            '[1, 0.5, 0.3]' '[1, 0.5, 0.3]' \\"
    echo "            '[0.03, 0.3]' \\"
    echo "            0.01 0.25 \\"
    echo "            10"
    echo
    exit 1
fi

out_dir="${1}"
seed="${2}"
num_of_topology="${3}"
taxa_num="${4}"
range_of_taxa_num="${5}"
len_of_msa_lower_bound="${6}"
len_of_msa_upper_bound="${7}"
num_of_process="${8}"
distribution_of_internal_branch_length="${9}"
distribution_of_external_branch_length="${10}"
range_of_mean_pairwise_divergence="${11}"
indel_substitution_rate_lower_bound="${12}"
indel_substitution_rate_upper_bound="${13}"
max_indel_length="${14}"

# Get the script directory to find indelible
script_dir=$(dirname $(readlink -f $0))
project_root=$(dirname "$script_dir")
indelible_path="${script_dir}/indelible"

# Check if indelible exists
if [ ! -f "${indelible_path}" ]; then
    echo "Error: INDELible executable not found at ${indelible_path}"
    echo "Please ensure indelible is in the simulation/ directory"
    exit 1
fi

# Create output directory structure
mkdir -pv "${out_dir}"

echo "=========================================="
echo "Generating dataset: ${out_dir}"
echo "=========================================="
echo "Parameters:"
echo "  Seed: ${seed}"
echo "  Number of topologies: ${num_of_topology}"
echo "  Taxa number: ${taxa_num}"
echo "  Range of taxa: ${range_of_taxa_num}"
echo "  MSA length: ${len_of_msa_lower_bound}-${len_of_msa_upper_bound}"
echo "  Processes: ${num_of_process}"
echo "  Max indel length: ${max_indel_length}"
echo "=========================================="

# Run fusang.py simulate command
# Note: --verbose is a global argument, must come before the subcommand
time uv run "${project_root}/fusang.py" --verbose simulate \
    --simulation_dir "${out_dir}" \
    --num_of_topology ${num_of_topology} \
    --taxa_num ${taxa_num} \
    --range_of_taxa_num "${range_of_taxa_num}" \
    --len_of_msa_lower_bound ${len_of_msa_lower_bound} \
    --len_of_msa_upper_bound ${len_of_msa_upper_bound} \
    --num_of_process ${num_of_process} \
    --distribution_of_internal_branch_length "${distribution_of_internal_branch_length}" \
    --distribution_of_external_branch_length "${distribution_of_external_branch_length}" \
    --range_of_mean_pairwise_divergence "${range_of_mean_pairwise_divergence}" \
    --range_of_indel_substitution_rate "[${indel_substitution_rate_lower_bound}, ${indel_substitution_rate_upper_bound}]" \
    --max_indel_length ${max_indel_length} \
    --indelible_path "${indelible_path}" \
    --seed ${seed} \
    --batch_size 1000

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Dataset generation completed: ${out_dir}"
    echo "=========================================="
else
    echo "=========================================="
    echo "Error: Dataset generation failed for ${out_dir}"
    echo "=========================================="
    exit 1
fi

