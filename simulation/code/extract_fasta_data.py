import os
import shutil
import argparse
from multiprocessing import Process, Pool
import multiprocessing
import subprocess
from Bio import AlignIO

parser = argparse.ArgumentParser('filter output msa length')
p_input = parser.add_argument_group("INPUT")
p_input.add_argument("--length", action="store", type=int, default=1e10, required=False)
p_input.add_argument("--in_dir", action="store", type=str, required=True)
p_input.add_argument("--out_dir", action="store", type=str, required=True)

args = parser.parse_args()
length = args.length
in_dir = args.in_dir
out_dir = args.out_dir

def get_msa_length(msa_dir):
    alignment = AlignIO.read(open(msa_dir), 'fasta')
    len_of_msa = len(alignment[0].seq)
    return len_of_msa

def extract(ele):
    if('.fas' in ele and 'TRUE' in ele):
        file = os.path.join(in_dir, ele)
        msa_length = get_msa_length(file)
        if msa_length > length:
            out_fail_dir = os.path.join(out_dir, 'fail')
            if not os.path.exists(out_fail_dir):
                os.mkdir(out_fail_dir)
            file_fasta = os.path.join(out_fail_dir, ele)
        else:
            file_fasta = os.path.join(out_dir, ele)
        shutil.copy(file, file_fasta)

para_list = os.listdir(in_dir)
pool = Pool(8)
pool.map(extract, para_list)
pool.close()
pool.join()
