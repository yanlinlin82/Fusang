#!/bin/bash

if [ -z "${12}" ]; then
    echo
    echo "Usage: $0 <out-dir> <num_of_topology> <taxa_num> <range_of_taxa_num> \\"
    echo "          <len_of_msa_upper_bound> <len_of_msa_lower_bound> <num_of_process> \\"
    echo "          <distribution_of_internal_branch_length> <distribution_of_external_branch_length> \\"
    echo "          <range_of_mean_pairwise_divergence> <range_of_indel_substitution_rate> <max_indel_length>"
    echo
    echo "Example: $0 out/ 20 5 '[5, 40]' \\"
    echo "            1200 240 24 \\"
    echo "            '[1, 0.5, 0.3]' '[1, 0.5, 0.3]' \\"
    echo "            '[0.03, 0.3]' '[0.01, 0.25]' 10"
    echo
    exit 1
fi

out_dir="${1}"
num_of_topology="${2}"
taxa_num="${3}"
range_of_taxa_num="${4}"
len_of_msa_upper_bound="${5}"
len_of_msa_lower_bound="${6}"
num_of_process="${7}"
distribution_of_internal_branch_length="${8}"
distribution_of_external_branch_length="${9}"
range_of_mean_pairwise_divergence="${10}"
range_of_indel_substitution_rate="${11}"
max_indel_length="${12}"

if [ -e "${out_dir}" ]; then
    echo "Error: Output directory already exists."
    exit 1
fi

mkdir -p "${out_dir}/code"

cd "${out_dir}/code"
ln -sv ../../code/* ./

time python simulate_topology.py \
    --num_of_topology ${num_of_topology} \
    --taxa_num ${taxa_num} \
    --range_of_taxa_num "${range_of_taxa_num}" \
    --len_of_msa_upper_bound ${len_of_msa_upper_bound} \
    --len_of_msa_lower_bound ${len_of_msa_lower_bound} \
    --num_of_process ${num_of_process} \
    --distribution_of_internal_branch_length "${distribution_of_internal_branch_length}" \
    --distribution_of_external_branch_length "${distribution_of_external_branch_length}" \
    --range_of_mean_pairwise_divergence "${range_of_mean_pairwise_divergence}" \
    --range_of_indel_substitution_rate "${range_of_indel_substitution_rate}" \
    --max_indel_length ${max_indel_length}

cd ../simulate_data/
../../indelible

cd ../code/
time python extract_fasta_data.py --length 1200

cp ../simulate_data/trees.txt ../label_file/trees.txt

time python gen_numpy.py
