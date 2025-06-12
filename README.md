# Workspace for Levanter and Anticipation Repos

**Reference:** [Anticipatory Music Transformer](https://arxiv.org/abs/2306.08620) &nbsp; ([Blog](https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html))

## Lakh MIDI Data Training with Levanter

This README outlines how to preprocess the lakh midi data and train a transformer model on it using levanter.

## Setup Steps

### 1. Setup Anticipation Environment
Set up anticipation in a new conda environment under the anticipation repo. The anticipation repo contains code to preprocess data and do inference on a saved model checkpoint. It does not support direct training of the AMT model.

### 2. Data Preprocessing
Follow the anticipation/train repo instructions to download and preprocess the lakh midi dataset. Specifics are in the README file in `anticipation/train`. This will generate pre-tokenized `.txt` files with space-separated integers representing musical tokens.

### 3. Levanter Setup and Training
Switch to the levanter repo and set up the training environment:

#### Environment Setup
```bash
# Create and activate the levanter conda environment
conda create -n levanter-main python=3.10
conda activate levanter-main

# Install levanter dependencies (follow levanter README for specific instructions)
```

#### Configuration
Create a config file `config/lakh_small.yaml` with the following structure:
```yaml
data:
  train_urls:
    - "/path/to/your/train.txt"
  validation_urls:
    - "/path/to/your/valid.txt"
  
  cache_dir: "/path/to/your/cache/"
  
  # Use passthrough tokenizer for pre-tokenized data
  tokenizer: "passthrough"
  vocab_size: 55028  # Required for passthrough tokenizer
  enforce_eos: false

model:
  type: gpt2
  hidden_dim: 768
  num_heads: 12
  num_layers: 12
  seq_len: 1024
  scale_attn_by_inverse_layer_idx: true

trainer:
  mp: p=f32,c=bfloat16
  model_axis_size: 1
  per_device_parallelism: 4
  train_batch_size: 512

  checkpointer:
    base_path: "/path/to/your/checkpoints/"
    save_interval: 30m

  axis_resources:
    batch: "data"
    vocab: "model"
    mlp: "model"
    heads: "model"
  parameter_axis_resources:
    embed: "data"

optimizer:
  learning_rate: 6E-4
  weight_decay: 0.1
```

#### Training Command
```bash
# Run training with the custom config
python -m levanter.main.train_lm --config_path config/lakh_small.yaml

# For testing with minimal steps:
python -m levanter.main.train_lm --config_path config/lakh_small.yaml --trainer.num_train_steps 1
```

#### Cache Verification (Optional)
If you want to verify that your cache was built correctly and contains actual token data:

```bash
# Create a simple cache verification script
cat > check_cache.py << 'EOF'
import numpy as np
from levanter.store.cache import TreeCache

# Load and inspect the training cache
cache = TreeCache.load("/path/to/your/cache/train", item_type=np.ndarray)
print(f"Cache length: {len(cache)}")

# Check a few samples
for i in range(min(5, len(cache))):
    sample = cache[i]
    print(f"Sample {i}: shape={sample.shape}, first 10 tokens={sample[:10]}")
    non_zero_count = np.count_nonzero(sample)
    print(f"  Non-zero tokens: {non_zero_count}/{len(sample)}")
EOF

# Run the verification script
python check_cache.py
```

**Note**: Replace `/path/to/your/cache/` with your actual cache directory path from the config file.

## Debugging Notes - June 11, 2025

### Issues Encountered and Resolved

#### 1. PassthroughTokenizer Initialization Issues
**Problem**: `IndexError: list index out of range` in `tree_store.py` during cache building.

**Root Cause**: The `BatchTokenizer` class was testing the `PassthroughTokenizer` with regular text `"hi there"` during initialization, but `PassthroughTokenizer` expects space-separated integers. This caused empty arrays to be returned, leading to index errors.

**Files Modified**:
- `src/levanter/data/text.py` (lines 233-244): Added special handling for `PassthroughTokenizer` in `BatchTokenizer.__init__()` to skip EOS/BOS tests
- `src/levanter/data/text.py` (lines 329-337): Modified `output_exemplar` property to use sample integers `"1 2 3 4 5"` for `PassthroughTokenizer`
- `src/levanter/data/text.py` (lines 866-870): Added `PassthroughTokenizer` check in `mk_single_turn_cached_sft_dataset` function

#### 2. .txt File Format Support
**Problem**: `ValueError: Unknown format .txt` during data loading.

**Root Cause**: The `UrlDataSource` class only supported `.jsonl`, `.json`, and `.parquet` formats, but not `.txt` files. However, `TextUrlDataSource` had built-in support for `.txt` files.

**Solution**: Modified `UrlDatasetSourceConfig.get_shard_source()` to automatically detect `.txt` files and use `TextUrlDataSource` instead of `UrlDataSource`.

**Files Modified**:
- `src/levanter/data/text.py` (lines 508-520): Added automatic detection of `.txt` files in `UrlDatasetSourceConfig.get_shard_source()` method, with fallback to `TextUrlDataSource` for `.txt` files

#### 3. Token Byte Length Calculation
**Problem**: `IndexError: list index out of range` in evaluation setup phase in `hf_utils.py`.

**Root Cause**: The `byte_length_of_token` function was trying to tokenize `"."` with `PassthroughTokenizer` for byte length calculations, causing the same empty array issue.

**Files Modified**:
- `src/levanter/utils/hf_utils.py` (lines 36-41): Added special case for `PassthroughTokenizer` to return a fixed 4-byte estimate per token

### Verification
- **Cache Building**: Successfully built cache with 120,985 rows and 123,888,640 tokens
- **Data Integrity**: Verified cached data contains actual token IDs (e.g., `55026, 55025, 10012, 27426`) rather than zeros
- **Pipeline**: Complete data pipeline from `.txt` files through tokenization to cache storage working correctly

### Key Insights
1. `PassthroughTokenizer` requires special handling throughout the codebase since it expects pre-tokenized integer data rather than raw text
2. Levanter's built-in `TextUrlDataSource` already supports `.txt` files, but the configuration system needed modification to use it automatically
3. The error progression showed the fixes were working: first fixed tree_store errors, then format errors, then evaluation setup errors

### Files Created/Modified Summary
- `config/lakh_small.yaml`: Training configuration for pre-tokenized music data
- `src/levanter/data/text.py`: Multiple fixes for PassthroughTokenizer compatibility
- `src/levanter/utils/hf_utils.py`: Token byte length calculation fix
- `check_cache.py`: Temporary verification script (can be removed) 
