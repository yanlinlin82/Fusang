# Old Model Directory

This directory contains the legacy deep learning model files in TensorFlow checkpoint format. These models need to be upgraded to the newer Keras H5 format for use with the current version of Fusang.

## Directory Structure

The old models are organized by sequence length and model parameters:

```text
old_model/
├── len_240/          # Models for MSA length ≤ 1210
│   ├── S1G/         # Standard sequence, Model 1, Gamma branch model
│   ├── S1U/         # Standard sequence, Model 1, Uniform branch model
│   ├── C1G/         # Coding sequence, Model 1, Gamma branch model
│   ├── C1U/         # Coding sequence, Model 1, Uniform branch model
│   ├── N1G/         # Noncoding sequence, Model 1, Gamma branch model
│   └── N1U/         # Noncoding sequence, Model 1, Uniform branch model
├── len_1200/        # Models for MSA length > 1210
│   ├── S2G/         # Standard sequence, Model 2, Gamma branch model
│   ├── S2U/         # Standard sequence, Model 2, Uniform branch model
│   ├── C2G/         # Coding sequence, Model 2, Gamma branch model
│   ├── C2U/         # Coding sequence, Model 2, Uniform branch model
│   ├── N2G/         # Noncoding sequence, Model 2, Gamma branch model
│   └── N2U/         # Noncoding sequence, Model 2, Uniform branch model
└── upgrade-models.py
```

## Model Naming Convention

Each model directory contains checkpoint files with the following naming pattern:

- `best_weights_clas.data-00000-of-00002`
- `best_weights_clas.data-00001-of-00002`
- `best_weights_clas.index`
- `checkpoint`

The directory names follow the pattern: `{SequenceType}{ModelNumber}{BranchModel}`

- **Sequence Type**: `S` (standard), `C` (coding), `N` (noncoding)
- **Model Number**: `1` (for len_240, window size 240) or `2` (for len_1200, window size 1200)
- **Branch Model**: `G` (gamma) or `U` (uniform)

## Model Architecture

### Model 240 (len_240)

- **Input shape**: `(4, 240, 1)`
- **Architecture**:
  - 8-layer CNN with bidirectional LSTM
  - Convolutional layers with filters: [1024, 1024, 128, 128, 128, 128, 128, 128]
  - Pooling sizes: [1, 2, 2, 2, 2, 2, 2, 2]
  - 3-layer bidirectional LSTM (128 units each)
  - Dense layer (1024 units) with dropout (0.2)
  - Output layer (3 units, softmax activation)

### Model 1200 (len_1200)

- **Input shape**: `(4, 1200, 1)`
- **Architecture**:
  - 8-layer CNN with bidirectional LSTM
  - Convolutional layers with filters: [1024, 1024, 128, 128, 128, 128, 128, 128]
  - Pooling sizes: [1, 4, 4, 4, 2, 2, 2, 1]
  - 3-layer bidirectional LSTM (128 units each)
  - Dense layer (1024 units) with dropout (0.2)
  - Output layer (3 units, softmax activation)

## Upgrading Models

The old checkpoint format models need to be converted to Keras H5 format for compatibility with newer TensorFlow/Keras versions.

### Prerequisites

1. Install `uv` and Python 3.8:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install python@3.8
```

2. Create a virtual environment with old TensorFlow/Keras versions:

```sh
uv venv .venv-old-keras
source .venv-old-keras/bin/activate
uv pip install tensorflow==2.8
uv pip install protobuf==3.20
```

### Upgrade Process

1. Run the upgrade script:

```sh
cd old_model
uv run python upgrade-models.py
```

The script will:

- Load each old checkpoint model
- Convert it to H5 format
- Save the upgraded models to the `model/` directory with names like `S1G.h5`, `C2U.h5`, etc.

2. Verify the upgrade:

   - Check that all 12 model files exist in the `model/` directory:
     - `S1G.h5`, `S1U.h5`, `C1G.h5`, `C1U.h5`, `N1G.h5`, `N1U.h5`
     - `S2G.h5`, `S2U.h5`, `C2G.h5`, `C2U.h5`, `N2G.h5`, `N2U.h5`

3. Test the upgraded models:

```sh
cd ..
uv run fusang.py -m example_msa/a.fas -s test_output
```

## Cleanup

After successfully upgrading all models and verifying they work correctly, you can clean up:

1. **Deactivate the old virtual environment**:

```sh
deactivate
```

2. **Remove the old virtual environment** (optional):

```sh
rm -rf .venv-old-keras
```

3. **Uninstall Python 3.8** (optional, if not needed elsewhere):

```sh
uv python uninstall python@3.8
```

4. **Remove old checkpoint files** (optional, after verification):

   - The `old_model/` directory can be kept as a backup
   - Or remove individual checkpoint files if disk space is a concern
   - **Warning**: Only remove after confirming all upgraded models work correctly

### Cleanup Checklist

- [ ] All 12 models successfully upgraded to H5 format
- [ ] Upgraded models tested and working correctly
- [ ] Old virtual environment deactivated
- [ ] Old virtual environment removed (optional)
- [ ] Python 3.8 uninstalled (optional)
- [ ] Old checkpoint files removed or archived (optional)

## Notes

- The upgrade process requires TensorFlow 2.8 and protobuf 3.20 due to compatibility with the old checkpoint format
- The upgraded H5 models are compatible with newer TensorFlow/Keras versions
- The model architecture definitions are identical between old and new formats; only the file format changes
- Keep the `old_model/` directory as a backup until you're confident the upgraded models work correctly
