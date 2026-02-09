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

## Development

### Installing Development Dependencies

To install development dependencies (including pytest for running unit tests):

```sh
uv sync --with dev
```

This will install all production dependencies plus development dependencies defined in `[dependency-groups]` section of `pyproject.toml`.

### Running Unit Tests

The project includes unit tests for core algorithm functions in `fusang.py`. To run the tests:

```sh
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_math_utils.py

# Run tests with coverage report (requires pytest-cov)
uv run pytest tests/ --cov=fusang --cov-report=html
```

For more details about the test suite, see [tests/README.md](tests/README.md).

## Example of usage

### 1. Quick start

Fusang now uses subcommands. Use `infer` for tree inference:

```sh
uv run fusang.py infer -i /path/to/your_msa.fas -o dl_output/result.txt
```

An example of command as follows:

```sh
uv run fusang.py infer -i ./example_msa/a.fas -o dl_output/your_prefix.txt
```

This command runs phylogenetic reconstruction of your MSA file. The output is a phylogenetic tree in Newick format. If `-o/--output` is omitted, the tree is written to stdout.

The required parameter:

- `-i, --input` The path to the MSA file. For current version of Fusang, we support both FASTA and PHYLIP format of MSA. The example of current MSA format can be seen in the directory of `example_msa`.

Optional output:

- `-o, --output` The output tree file path (Newick). If not set, Fusang writes the tree to stdout.

**Viewing the tree structure:**

You can pipe the output directly to `treeview.py` to visualize the tree structure in ASCII format:

```sh
uv run fusang.py infer -i example_msa/a.fas -q | uv run treeview.py
```

This command will:

1. Run Fusang on the input MSA file (`example_msa/a.fas`) with quiet mode (`-q`)
2. Pipe the Newick format output to `treeview.py`
3. Display the tree in a human-readable ASCII format

You can also use `treeview.py` to view a saved tree file:

```sh
uv run treeview.py dl_output/your_prefix.txt
```

Or pipe from stdin:

```sh
cat output.tree | uv run treeview.py
```

---

### 2. Parameter setting

**Required parameters:**

`-i, --input` The path of your MSA file

**Optional parameters:**

You can set the parameters as follows for specific scenario

`-o, --output` Output tree file path (Newick). If not set, prints to stdout

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
uv run fusang.py infer -i input.fas -o output.tree

# Coding sequences with gamma branch model
# For MSA ≤ 1210 bp: loads model/C1G.h5
uv run fusang.py infer -i input.fas -o output.tree -t coding

# Non-coding sequences with uniform branch model
# For MSA > 1210 bp: loads model/N2U.h5
uv run fusang.py infer -i input.fas -o output.tree -t noncoding -r uniform
```

#### Recommendation

- Use `-t coding` if your MSA contains protein-coding sequences (CDS regions)
- Use `-t noncoding` if your MSA contains non-coding sequences (introns, intergenic regions, UTRs)
- Use `-t standard` (default) for mixed sequences or when uncertain
- Use `-r gamma` (default) for most biological datasets
- Use `-r uniform` only if you have specific reasons to assume uniform branch lengths

---

## Simulation Data Generation

Use `simulate` to generate FASTA and numpy training data with INDELible:

```sh
uv run fusang.py simulate -o ./simulation/out/S1U -n 20 -t 5
```

Key options:

- `-o, --output`: Simulation output directory (creates `fasta_data/` and `numpy_data/`)
- `-n, --num_of_topology`: Number of MSAs to simulate
- `-t, --taxa_num`: Number of taxa in the final tree
- `-p, --num_of_process`: Number of processes (defaults to CPU core count)
- `--evaluate`: Run inference and evaluation after simulation
- `--evaluation_output`: Directory to save evaluation results

## Model Training

Fusang supports training custom models from simulated or real phylogenetic data. The training process includes automatic best model selection, early stopping, and comprehensive training history logging.

### Quick Start

To train a model, you need:

1. Sequence data in numpy format (`.npy` files)
2. Label data in numpy format (`.npy` files)

```sh
uv run fusang.py train \
  -d /path/to/simulation/out/S1U \
  -o model/my_model.weights.h5
```

### Training Parameters

**Required parameters:**

- `-d, --data_dir`: Data directory. Accepts a `numpy_data` directory (with `seq/` and `label/`) or a simulation output directory (with `numpy_data/seq/` and `numpy_data/label/`).
- `-o, --output`: Path to save trained model weights (`.weights.h5` file). If another extension is provided, it will be adjusted to `.weights.h5`.

**Optional parameters:**

- `-w, --window_size`: Window size for model (240 or 1200, default: 240)
- `-e, --epochs`: Number of training epochs (default: 100)
- `-b, --batch_size`: Batch size for training (default: 32)
- `-r, --learning_rate`: Learning rate (default: 0.001)
- `--train_ratio`: Ratio of data for training (default: 0.8)
- `--val_ratio`: Ratio of data for validation (default: 0.1)
- `-M, --monitor`: Metric to monitor for checkpointing and early stopping
  - Options: `val_loss`, `val_accuracy`, `loss`, `accuracy` (default: `val_loss`)
- `-P, --patience`: Number of epochs with no improvement before early stopping (default: 10)
  - Set to `0` to disable early stopping

### Training Features

#### 1. Automatic Best Model Selection

The training process automatically saves the best model based on validation performance:

- **Best model checkpoint**: Saved to `best_model.weights.h5` in the same directory as `model_save_path`
- **Final model**: The best model weights are automatically loaded and saved to `model_save_path` after training completes
- **Monitoring metric**: By default, monitors `val_loss` (lower is better)
  - For `val_loss` or `loss`: Lower values are better
  - For `val_accuracy` or `accuracy`: Higher values are better

#### 2. Early Stopping

Early stopping prevents overfitting by automatically stopping training when the monitored metric stops improving:

- **Default patience**: 10 epochs
- **Trigger condition**: The monitored metric (e.g., `val_loss`) doesn't improve for `patience` consecutive epochs
- **Automatic weight restoration**: When early stopping triggers, the model automatically restores the best weights
- **Disable early stopping**: Set `--patience 0` to train for the full number of epochs

**Example scenarios:**

- Epochs 1-5: `val_loss` decreases from 0.8 to 0.5 (best at epoch 5)
- Epochs 6-15: `val_loss` fluctuates between 0.5-0.6 (no improvement)
- Epoch 16: Early stopping triggers (10 epochs without improvement)
- Result: Training stops, model restored to epoch 5 weights (best model)

#### 3. Training History Logging

Each epoch's metrics are automatically saved to `training_history.tsv` in the same directory as the model:

- **File location**: `{model_save_path directory}/training_history.tsv`
- **Format**: TSV (tab-separated values)
- **Columns**: `epoch`, `loss`, `accuracy`, `val_loss`, `val_accuracy`
- **Usage**: Can be used to plot training curves and analyze training progress

**Example training history file:**

```tsv
epoch	loss	accuracy	val_loss	val_accuracy
1	0.8234	0.6543	0.7123	0.7234
2	0.7123	0.7234	0.6543	0.7654
3	0.6543	0.7654	0.6123	0.7890
...
```

### Training Output

After training completes, the following files are created in the model directory:

```
model_directory/
├── my_model.weights.h5      # Final model (contains best weights)
├── best_model.weights.h5    # Best model checkpoint (backup)
└── training_history.tsv     # Training metrics for each epoch
```

### Training Workflow

1. **Data Splitting**: Data is automatically split into training (80%), validation (10%), and test (10%) sets
2. **Model Initialization**: Model architecture is selected based on `window_size` (240 or 1200)
3. **Training Loop**:
   - Each epoch trains on the training set
   - Validates on the validation set
   - Saves metrics to `training_history.tsv`
   - Updates best model checkpoint if validation performance improves
4. **Early Stopping Check**: If no improvement for `patience` epochs, training stops early
5. **Final Model Save**: Best model weights are loaded and saved to `model_save_path`

### Best Practices

1. **Use validation set**: Always use a validation set (`val_ratio > 0`) to enable best model selection and early stopping
2. **Monitor validation metrics**: Use `val_loss` or `val_accuracy` (not training metrics) to avoid overfitting
3. **Set appropriate patience**:
   - Small datasets: Lower patience (5-10)
   - Large datasets: Higher patience (10-20)
4. **Analyze training history**: Check `training_history.tsv` to identify overfitting or underfitting
5. **Continue training**: If training was interrupted, you can continue by loading the existing model weights

### Example: Training with Custom Parameters

```sh
# Train with custom parameters
uv run fusang.py train \
  -d simulation/out/S1U \
  -o model_new/S1U.weights.h5 \
  -w 240 \
  -e 100 \
  -b 64 \
  -r 0.0005 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  -M val_loss \
  -P 15
```

### Example: Disable Early Stopping

```sh
# Train for full number of epochs (no early stopping)
uv run fusang.py train \
  -d /path/to/numpy_data \
  -o model.weights.h5 \
  -e 200 \
  -P 0
```

### Example: Monitor Validation Accuracy

```sh
# Monitor validation accuracy instead of loss
uv run fusang.py train \
  -d /path/to/numpy_data \
  -o model.weights.h5 \
  -M val_accuracy \
  -P 10
```

For more information about generating training data, see [simulation/README.md](simulation/README.md).

---

## Model Evaluation (numpy data)

Evaluate a model on numpy data generated from simulations or existing datasets:

```sh
uv run fusang.py evaluate \
  -m model/S1U.weights.h5 \
  -d ./simulation/out/S1U \
  -o ./results
```

Optional parameters:

- `--window_size`: Window size for evaluation (240 or 1200). If not set, it is inferred from the model filename.
- `--batch_size`: Batch size for prediction (default: 32).

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
