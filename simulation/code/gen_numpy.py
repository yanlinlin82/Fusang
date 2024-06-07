import os
import argparse
import numpy as np
import random
import pandas as pd
from ete3 import Tree

parser = argparse.ArgumentParser('filter output msa length')
p_input = parser.add_argument_group("INPUT")
p_input.add_argument("--in_trees_txt", action="store", type=str, required=True)
p_input.add_argument("--in_fasta_dir", action="store", type=str, required=True)
p_input.add_argument("--out_dir", action="store", type=str, required=True)

args = parser.parse_args()
in_trees_txt = args.in_trees_txt
in_fasta_dir = args.in_fasta_dir
out_dir = args.out_dir


folder_numpy_seq = out_dir + 'seq/'
folder_numpy_label = out_dir + 'label/'
if not os.path.exists(folder_numpy_seq):
    os.mkdir(folder_numpy_seq)
if not os.path.exists(folder_numpy_label):
    os.mkdir(folder_numpy_label)

csv_data = pd.read_table(in_trees_txt, skiprows=5, sep='\t', header=None)
file = list(csv_data[0])
topo = list(csv_data[8])
dic = {}

for i in range(0, len(file)):
    dic[file[i]] = topo[i]

def assign_label(str):
    t1 = Tree(str,format=5)
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

def get_numpy(folder_dir, fasta_dir):
    aln_file = folder_dir + fasta_dir
    aln = open(aln_file)
    dic = {'A':'0','T':'1','C':'2','G':'3','-':'4'}

    matrix_out=[]
    fasta_dic={}
    for line in aln:
        if line[0]==">":
            header=line[1:].rstrip('\n').strip()
            fasta_dic[header]=[]
        elif line[0].isalpha() or line[0]=='-':
            for base, num in dic.items():
                line=line[:].rstrip('\n').strip().replace(base,num)
            line=list(line)
            line=[int(n) for n in line]

            tmp_line = line+[4]*(2000-len(line))
            fasta_dic[header] += tmp_line[0:2000]

    taxa_block=[]
    for taxa in sorted(list(fasta_dic.keys())):
        taxa_block.append(fasta_dic[taxa.strip()])
    fasta_dic={}
    matrix_out.append(taxa_block)

    return np.array(matrix_out)

file_list = os.listdir(in_fasta_dir)
random.shuffle(file_list)

for ele in file_list:
    if ele == 'fail':
        continue
    tmp_file = ele.split('.')[0][:-5]
    current_label = dic[tmp_file] 
    current_label = np.array(assign_label(current_label))
    current_seq = get_numpy(in_fasta_dir, ele)
    seq_dir = folder_numpy_seq + ele.split('_TRUE')[0].split('sim')[1] + '.npy'
    label_dir = folder_numpy_label + ele.split('_TRUE')[0].split('sim')[1] + '.npy'
    np.save(seq_dir, current_seq)
    np.save(label_dir, current_label)
    print(f'[{current_label}] {current_seq.shape}')
