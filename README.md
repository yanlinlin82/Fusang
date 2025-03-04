# Fusang <img align="right" src="logo.jpg" width="170" height="170"/>

- forked from: [Jerry-0591/Fusang](https://github.com/Jerry-0591/Fusang)
- website at: <https://fusang.cibr.ac.cn/>

Fusang is a framework used for the reconstruction of phylogenetic tree via deep learning methods. For current version, it supports the reconstruction of MSA with 4-40 taxas and the length of it should be less than 10,000.

## Hardware requirements 

This repository can be run on both CPU and GPU environment, but we recommend users use GPU for accelerating. More details can be seen from Environment_setting.md

The limit usage of memory is ~24GB for current repository, for most cases, the memory usage is less than 20GB.

## Software requirements

The configuration of the environment (as in the paper) see [Environment_setting.md](Environment_setting.md) of this repository.

Or simply run following commands to create a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -U -r requirements.txt
```

## Example of usage

### 1. Quick start

You can run Fusang using default parameter setting through the command as follows:

```
python fusang.py --msa_dir /path/to/your_msa.fas --save_prefix your_prefix
```

An example of command as follows:

```
python fusang.py --msa_dir ./example_msa/msa1.fas --save_prefix dl_output_1
```

This command will do phylogenetic reconstruction of your MSA file, the result will be saved in file with the prefix that you set in `--save_prefix`

The meaning of these two mandatory parameter:

`--msa_dir` The path to MSA file,  for current version of Fusang, we support both fasta and phylip format of MSA. The example of current MSA format can be seen in the directory of `example_msa`

`--save_prefix`  The prefix of output file, the predicted tree will be saved on the directory of `dl_output` , with the prefix that set in this parameter. You can see `example_dl_output` to find the example of predicted tree.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 2. Parameter setting

`--msa_dir` The path of your msa file

`--save_prefix` The prefix of the result file

You can set the parameters as follows for specific scenario

`--beam_size` The size of beam in the beam search procedure, default beam size is 1

`--sequence_type` The type of the sequences in msa, which has three choices, coding, noncoding and standard (default, means both coding and noncoding)

`--branch_model` The distribution type of the branches, which has gamma (default) and uniform as choices

`--window coverage` The coverage of slide window, which decides the step of this algorithm. The default setting is 1


## Meaning of each file in this repository

`dl_model` The directory that saves the model of deep learning

`example_dl_output` The directory that saves the example of predicted tree

`example_msa` The directory that saves the example of input msa file

`public data` The data of Fusang manuscript

`simulation` The scripts used for simulation

`Environment_setting.md` The description of setting Fusang environment

`Experimental phylogeny.zip` This file contains the data of experimental phylogeny experiments

`example_command.sh` This command will generate the prdicted tree (in `example_dl_output` ) of input msa file (in `example_msa` ) 

`fusang.py` The code for tree reconstruction

`logo.jpg` The logo of Fusang project

`requirements.txt` This file will be used for setting environment

## References

- Zou Z, Zhang H, Guan Y, Zhang JJMB, Evolution. 2020. Deep residual neural networks resolve quartet molecular phylogenies. 37:1495-1507. doi: [10.1093/nar/gkad805](https://doi.org/10.1093/nar/gkad805)
- Suvorov A, Hochuli J, Schrider DRJSB. 2019. Accurate Inference of Tree Topologies from Multiple Sequence Alignments Using Deep Learning. 69:221-233. doi: [10.1093/sysbio/syz060](https://doi.org/10.1093/sysbio/syz060)
- <https://github.com/martin-sicho/PTreeGenerator/blob/c6eddaf613a0058959b2f077458fad6fe689241e/src/ptreegen/parsimony.py>


