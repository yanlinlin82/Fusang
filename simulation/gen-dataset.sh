#!/bin/bash

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
    echo "Example: $0 out/ 20 5 '[5, 40]' \\"
    echo "            240 1200 24 \\"
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

if [ -e "${out_dir}" ]; then
    echo "Error: Output directory already exists."
    exit 1
fi

mkdir -pv "${out_dir}"/{simulate_data,label_file,fasta_data,numpy_data}/

time python code/simulate_topology.py \
    --seed ${seed} \
    --num_of_topology ${num_of_topology} \
    --taxa_num ${taxa_num} \
    --range_of_taxa_num "${range_of_taxa_num}" \
    --num_of_process ${num_of_process} \
    --distribution_of_internal_branch_length "${distribution_of_internal_branch_length}" \
    --distribution_of_external_branch_length "${distribution_of_external_branch_length}" \
    --range_of_mean_pairwise_divergence "${range_of_mean_pairwise_divergence}" \
    --max_indel_length ${max_indel_length} \
    --output_newick ${out_dir}/label_file/newick.csv

time Rscript code/gen_control_file.R \
    ${seed} ${taxa_num} ${num_of_topology} ${len_of_msa_lower_bound} ${len_of_msa_upper_bound} \
    ${indel_substitution_rate_lower_bound} ${indel_substitution_rate_upper_bound} \
    ${max_indel_length} ${out_dir}/label_file/newick.csv ${out_dir}/simulate_data/control.txt

(
    cp -v ./indelible ${out_dir}/
    chmod +x ${out_dir}/indelible
    cd ${out_dir}/simulate_data/
    ../indelible
    rm -fv ../indelible
)

time python code/extract_fasta_data.py \
    --in_dir ${out_dir}/simulate_data/ \
    --out_dir ${out_dir}/fasta_data/
#    --length 1200 \

cp -av ${out_dir}/simulate_data/trees.txt ${out_dir}/label_file/

time python code/gen_numpy.py \
    --in_trees_txt ${out_dir}/label_file/trees.txt \
    --in_fasta_dir ${out_dir}/fasta_data/ \
    --out_dir ${out_dir}/numpy_data/

#rm -rf ${out_dir}/simulate_data/
