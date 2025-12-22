# Fusang (扶桑)

<img align="right" src="https://github.com/yanlinlin82/Fusang/blob/main/logo.jpg" width="170" height="170"/>

- Forked from: [Jerry-0591/Fusang](https://github.com/Jerry-0591/Fusang)
- Official website: <https://fusang.cibr.ac.cn/>

Fusang is a framework used for the reconstruction of phylogenetic tree via deep learning methods. For current version, it supports the reconstruction of MSA with 4-40 taxas and the length of it should be less than 10,000.

## Hardware requirements

This repository can be run on both CPU and GPU environment, but we recommend users use GPU for accelerating. More details can be seen from Environment_setting.md

The limit usage of memory is ~24GB for current repository, for most cases, the memory usage is less than 20GB.

## Software requirements

Run the following commands:

```sh
# install uv if required
curl -LsSf https://astral.sh/uv/install.sh | sh

# use uv to install all dependencies
uv sync
```

or see [Environment_setting.md](Environment_setting.md) for details

## Example of usage

### 1. Quick start

You can run Fusang using default parameter setting through the command as follows:

```sh
uv run fusang.py -m /path/to/your_msa.fas -s your_prefix
```

An example of command as follows:

```sh
python fusang.py -m ./example_msa/msa1.fas -s dl_output_1
```

This command will do phylogenetic reconstruction of your MSA file, the result will be saved in file with the prefix that you set in `--save_prefix` (or `-s`). The output is a phylogenetic tree in Newick format.

The meaning of these two mandatory parameter:

`-m, --msa_dir` The path to MSA file, for current version of Fusang, we support both fasta and phylip format of MSA. The example of current MSA format can be seen in the directory of `example_msa`

`-s, --save_prefix`  The prefix of output file, the predicted tree will be saved on the directory of `dl_output` , with the prefix that set in this parameter. The output file contains a phylogenetic tree in Newick format. You can see `example_dl_output` to find the example of predicted tree.

---

### 2. Parameter setting

**Required parameters:**

`-m, --msa_dir` The path of your msa file

`-s, --save_prefix` The prefix of the result file

**Optional parameters:**

You can set the parameters as follows for specific scenario

`-b, --beam_size` The size of beam in the beam search procedure, default beam size is 1

`-t, --sequence_type` The type of the sequences in msa, which has three choices, coding, noncoding and standard (default, means both coding and noncoding)

`-r, --branch_model` The distribution type of the branches, which has gamma (default) and uniform as choices

`-w, --window_coverage` The coverage of slide window, which decides the step of this algorithm. The default setting is 1

---

### 3. Model Selection

Fusang automatically selects the appropriate deep learning model based on your MSA characteristics. The model selection process involves two steps:

#### Step 1: Model Architecture Selection (Based on MSA Length)

The program automatically chooses the model architecture based on the length of your MSA:

- **Short sequences (≤ 1210 bp)**: Uses the 240-bp model architecture
  - Input shape: 4 taxa × 240 positions × 1 channel
  - Optimized for shorter alignments
  - Uses smaller sliding windows (240 bp)

- **Long sequences (> 1210 bp)**: Uses the 1200-bp model architecture
  - Input shape: 4 taxa × 1200 positions × 1 channel
  - Optimized for longer alignments
  - Uses larger sliding windows (1200 bp)

#### Step 2: Model Weight Selection (Based on Sequence Type and Branch Model)

After selecting the architecture, Fusang loads the appropriate pre-trained weights based on two parameters:

**Sequence Type (`-t, --sequence_type`):**

- `standard` (default): General-purpose model for mixed or unknown sequence types
- `coding`: Optimized for protein-coding sequences
- `noncoding`: Optimized for non-coding sequences (e.g., intergenic regions, introns)

**Branch Model (`-r, --branch_model`):**

- `gamma` (default): Assumes branch lengths follow a gamma distribution (more realistic for most biological data)
- `uniform`: Assumes uniform branch length distribution (simpler model)

#### Model Path Structure

The selected model weights are loaded from the `model/` directory with a flat structure:

```txt
model/
  ├── S1G.h5          (standard, model 1, gamma) - for MSA ≤ 1210 bp
  ├── S1U.h5          (standard, model 1, uniform) - for MSA ≤ 1210 bp
  ├── C1G.h5          (coding, model 1, gamma) - for MSA ≤ 1210 bp
  ├── C1U.h5          (coding, model 1, uniform) - for MSA ≤ 1210 bp
  ├── N1G.h5          (noncoding, model 1, gamma) - for MSA ≤ 1210 bp
  ├── N1U.h5          (noncoding, model 1, uniform) - for MSA ≤ 1210 bp
  ├── S2G.h5          (standard, model 2, gamma) - for MSA > 1210 bp
  ├── S2U.h5          (standard, model 2, uniform) - for MSA > 1210 bp
  ├── C2G.h5          (coding, model 2, gamma) - for MSA > 1210 bp
  ├── C2U.h5          (coding, model 2, uniform) - for MSA > 1210 bp
  ├── N2G.h5          (noncoding, model 2, gamma) - for MSA > 1210 bp
  └── N2U.h5          (noncoding, model 2, uniform) - for MSA > 1210 bp
```

Each `.h5` file contains the pre-trained weights for the corresponding model configuration.

#### Examples

```sh
# Default: standard sequence type, gamma branch model
# For MSA ≤ 1210 bp: loads model/S1G.h5
# For MSA > 1210 bp: loads model/S2G.h5
uv run fusang.py -m input.fas -s output

# Coding sequences with gamma branch model
# For MSA ≤ 1210 bp: loads model/C1G.h5
uv run fusang.py -m input.fas -s output -t coding

# Non-coding sequences with uniform branch model
# For MSA > 1210 bp: loads model/N2U.h5
uv run fusang.py -m input.fas -s output -t noncoding -r uniform
```

#### Recommendation

- Use `-t coding` if your MSA contains protein-coding sequences (CDS regions)
- Use `-t noncoding` if your MSA contains non-coding sequences (introns, intergenic regions, UTRs)
- Use `-t standard` (default) for mixed sequences or when uncertain
- Use `-r gamma` (default) for most biological datasets
- Use `-r uniform` only if you have specific reasons to assume uniform branch lengths

## Meaning of each file in this repository

`model` The directory that saves the pre-trained model weights (`.h5` files) for deep learning

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

- Zou Z, Zhang H, Guan Y, Zhang JJMB, Evolution. 2020. Deep residual neural networks resolve quartet molecular phylogenies.  37:1495-1507.

- Suvorov A, Hochuli J, Schrider DRJSB. 2019. Accurate Inference of Tree Topologies from Multiple Sequence Alignments Using Deep Learning.  69:221-233.

- https://github.com/martin-sicho/PTreeGenerator/blob/c6eddaf613a0058959b2f077458fad6fe689241e/src/ptreegen/parsimony.py
