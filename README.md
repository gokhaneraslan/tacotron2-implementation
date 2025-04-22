# Tacotron 2 TTS Training Implementation

A Python-based implementation for training Tacotron 2 Text-to-Speech (TTS) models. This repository provides tools for preparing your dataset, configuring training parameters, and running the training process with multi-language support.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Dataset Preparation](#dataset-preparation)
- [Configuration (`config.yaml`)](#configuration-configyaml)
  - [Paths](#paths)
  - [Training Control](#training-control)
  - [Data Loading & Preprocessing](#data-loading--preprocessing)
  - [Language Selection](#language-selection)
  - [Hardware & Performance](#hardware--performance)
  - [Optimizer & Learning Rate](#optimizer--learning-rate)
  - [Logging & Visualization](#logging--visualization)
- [Usage Examples](#usage-examples)
- [Key Configuration Parameters (Needs Attention!)](#key-configuration-parameters-needs-attention)
- [Text Cleaners and Language Support](#text-cleaners-and-language-support)
- [Output Files](#output-files)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Training Workflow](#training-workflow)

## Features

- **Complete Tacotron 2 Training Pipeline:** Implements the core Tacotron 2 architecture for TTS.
- **Multi-Language Support:** Works with English, Turkish, Spanish, Arabic, and many other languages through customizable text cleaners and symbol sets.
- **Dataset Preprocessing:** Includes utilities to:
  - Generate Mel spectrograms from `.wav` files.
  - Create file lists (`list.txt`) from metadata (`metadata.csv`).
  - Check dataset integrity and file existence.
- **Flexible Configuration:** Uses a `config.yaml` file for easy management of paths, hyperparameters, and training settings.
- **Advanced Training Features:**
  - Checkpointing: Saves model checkpoints periodically and allows resuming training.
  - Warm Start: Ability to initialize training from a pre-trained model's weights.
  - Validation: Performs validation runs at specified intervals.
  - Distributed Training: Supports multi-GPU training using `torch.distributed`.
  - Mixed-Precision Training (FP16): Option for faster training and reduced memory usage.
- **Comprehensive Logging:** Monitor progress through console, log files (`training.log`), and TensorBoard.

## Requirements

- Python 3.7+
- PyTorch (>= 1.7 recommended, check CUDA compatibility if using GPU)
- CUDA Toolkit & cuDNN (required for GPU acceleration)
- Additional Python packages:

```bash
pip install torch torchvision torchaudio numpy pyyaml tqdm num2words librosa tensorboard
```
*(Adjust the PyTorch installation command based on your system and CUDA version - see [PyTorch Official Website](https://pytorch.org/))*

Full list of dependencies:
- NumPy
- PyYAML (for config loading)
- Tqdm (for progress bars)
- Num2Words (for number expansion in text cleaners)
- Librosa (for audio processing)
- TensorBoard (for visualization)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gokhaneraslan/tacotron2-tts-training.git
   cd tacotron2-tts-training
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:** (See [Dataset Preparation](#dataset-preparation))

4. **Configure `config.yaml`:** Create and edit `config/config.yaml` (See [Configuration (`config.yaml`)](#configuration-configyaml))

## Dataset Preparation

The training script expects your dataset in a specific format:

```
/base_project_path/
└── dataset_name/
    ├── metadata.csv
    ├── list.txt (generated automatically)
    └── wavs/
        ├── audio1.wav
        ├── audio2.wav
        ├── audio1.npy (generated if configure)
        ├── audio2.npy (generated if configure)
        └── ...
```

1. **Audio Files:** Place all your `.wav` audio files inside a directory named `wavs` within your main dataset directory.
   - **Critical:** Ensure all `.wav` files have the **same sampling rate** as specified in `config.yaml` (`sampling_rate`).

2. **Metadata File:** Create a metadata file (e.g., `metadata.csv`) in your main dataset directory with the format:
   ```
   <wav_filename_without_extension>|<text_1>|<text_2_transcription>
   ```
   - Example: `audio1|This is the original text.|This is the normalized transcription.`
   - The script primarily uses the **third column** (`text_2_transcription`) as the text input.
   - The delimiter is the pipe symbol (`|`).

3. **File List (`list.txt`):** Generated automatically by the training script.
   - Format: `<path/to/audio_or_mel.ext>|<transcription>`

4. **Mel Spectrograms (`.npy`):** (Optional but Recommended for Speed)
   - If `generate_mels: True`, the script will process `.wav` files and save corresponding Mel spectrograms as `.npy` files.
   - If `load_mel_from_disk: True`, the script will use these `.npy` files during training, significantly speeding up data loading.

## Configuration (`config.yaml`)

Create a configuration file at `config/config.yaml`. Here are the key sections and parameters:

### Paths

```yaml
# --- Paths (❗ CRITICAL - SET THESE CAREFULLY ❗) ---
base_project_path: /path/to/your/project/tacotron2-implementation # Root directory
dataset_name: MyTTSDataset             # Name of the dataset directory
output_base_path: output                # Base directory for outputs
output_dir_name: checkpoints           # Subdirectory for model checkpoints
log_dir_name: logs                     # Subdirectory for logs
model_filename: tacotron_model          # Base name for saved checkpoint files
metadata_filename: metadata.csv         # Name of your metadata file
filelist_name: list.txt                # Name for the generated filelist
default_pretrained_path: /path/to/pretrained/tacotron2_statedict.pt # Optional path for warm start
```

### Training Control

```yaml
# --- Training Control ---
epochs: 250                     # Total number of training epochs
warm_start: False               # True: Load only model weights, False: Load full checkpoint
save_interval: 10               # Save main checkpoint every N epochs (0=disable)
backup_interval: 25             # Save backup checkpoint every N epochs (0=disable)
validation_interval: 5          # Run validation every N epochs
log_interval: 100               # Log training progress every N iterations
```

### Data Loading & Preprocessing

```yaml
# --- Data Loading & Preprocessing (❗ IMPORTANT ❗) ---
generate_mels: True             # Generate Mel spectrograms (.npy) before training
load_mel_from_disk: True        # Load generated .npy Mel files during training (faster)
```

### Language Selection

```yaml
# --- Language Selection (❗ CHOOSE ONE ❗) ---
text_cleaners: turkish_cleaners # Select cleaner for your dataset language
#text_cleaners: english_cleaners
#text_cleaners: spanish_cleaners
# ... other languages ...
```

### Hardware & Performance

```yaml
# --- Hardware & Performance ---
n_gpus: 1                       # Number of GPUs for distributed training
rank: 0                         # Process rank for distributed training
fp16_run: True                  # Enable mixed-precision (FP16) training
cudnn_enabled: True             # Enable cuDNN backend
cudnn_benchmark: True           # Enable cuDNN benchmark mode
num_workers: 4                  # Number of CPU workers for data loading
```

### Audio Parameters

```yaml
# --- Audio Parameters (❗ CRITICAL - MATCH YOUR DATASET ❗) ---
sampling_rate: 22050            # The exact sampling rate of ALL your .wav files
mel_fmax: 11025.0               # Max frequency for Mel spectrograms (usually sampling_rate / 2)
```

### Optimizer & Learning Rate

```yaml
# --- Optimizer & Learning Rate ---
lr_schedule_A: 1e-4             # Initial learning rate
lr_schedule_B: 8000             # Decay rate factor
lr_schedule_C: 0                # LR schedule offset
lr_schedule_decay_start: 10000  # Iteration at which LR decay begins
min_learning_rate: 1e-5         # Minimum learning rate clamp
```

### Logging & Visualization

```yaml
# --- Logging & Visualization ---
show_alignments: True           # Log alignment plots to TensorBoard
alignment_graph_height: 600     # Height of the alignment plot
alignment_graph_width: 1000     # Width of the alignment plot
```

## Usage Examples

### Basic Training Workflow

| Step | Command | Description |
|------|---------|-------------|
| **1. Configure** | `nano config/config.yaml` | Edit configuration file with your paths and settings |
| **2. First Run** | `python train.py` | Initial run with `generate_mels: True` to create mel spectrograms |
| **3. Regular Training** | `python train.py` | Subsequent runs with `generate_mels: False` and `load_mel_from_disk: True` |
| **4. Monitor Progress** | `tensorboard --logdir output/logs` | View training metrics and alignment plots |

### Configuration Examples for Different Scenarios

| Scenario | Configuration Settings | Description |
|----------|------------------------|-------------|
| **Initial Setup** | `generate_mels: True`<br>`load_mel_from_disk: True` | First run to generate mel spectrograms |
| **Regular Training** | `generate_mels: False`<br>`load_mel_from_disk: True` | Standard training using cached mels |
| **Limited GPU Memory** | `batch_size: 4`<br>`fp16_run: True` | Settings for smaller GPU memory |
| **High-end GPU** | `batch_size: 16`<br>`fp16_run: True` | Settings for larger GPU memory |
| **English Dataset** | `text_cleaners: english_cleaners`<br>`sampling_rate: 22050` | Configuration for English TTS |
| **Turkish Dataset** | `text_cleaners: turkish_cleaners`<br>`sampling_rate: 22050` | Configuration for Turkish TTS |
| **Resume Training** | `warm_start: False`<br>`generate_mels: False` | Continue from last checkpoint |
| **Transfer Learning** | `warm_start: True`<br>`default_pretrained_path: path/to/model.pt` | Fine-tune from pretrained model |
| **Multi-GPU Training** | `n_gpus: 4`<br>`batch_size: 32` | Distributed training across 4 GPUs |

## Key Configuration Parameters (Needs Attention!)

Getting these parameters wrong is the most common source of errors:

| Parameter | Importance | Description |
|-----------|------------|-------------|
| **`base_project_path`, `dataset_name`** | ⚠️ Critical | Must point to correct directories |
| **`sampling_rate`** | ⚠️ Critical | MUST match ALL your `.wav` files exactly |
| **`text_cleaners`** | ⚠️ Critical | Must match your dataset language |
| **`generate_mels` & `load_mel_from_disk`** | ⚠️ Important | Follow the workflow (first run vs. subsequent runs) |
| **`batch_size`** | ⚠️ Important | Adjust based on GPU memory (start small, increase if possible) |
| **`warm_start`** | ⚠️ Important | Controls whether to load only weights or full checkpoint |
| **`n_gpus`** | ⚠️ Important | Set correctly for distributed training |
| **`lr_schedule_A`** | ⚠️ Important | Initial learning rate (reduce if training unstable) |

## Text Cleaners and Language Support

The `text_cleaners` parameter selects text normalization rules and character sets for your language:

| Language | Cleaner | Features |
|----------|---------|----------|
| **English** | `english_cleaners` | Lowercase conversion, number expansion, punctuation handling |
| **Turkish** | `turkish_cleaners` | Turkish character normalization (ğ, ş, ı, etc.), number expansion |
| **Spanish** | `spanish_cleaners` | Spanish-specific characters (ñ, accented vowels), number expansion |
| **French** | `french_cleaners` | French-specific normalization and number expansion |
| **German** | `german_cleaners` | German character handling (ä, ö, ü, ß) and number expansion |
| **Italian** | `italian_cleaners` | Italian-specific normalization |
| **Portuguese** | `portuguese_cleaners` | Portuguese character handling and normalization |
| **Russian** | `russian_cleaners` | Cyrillic characters and Russian-specific normalization |
| **Arabic** | `arabic_cleaners` | Arabic script handling and normalization |

Each cleaner performs:
- Lowercase conversion
- Whitespace normalization
- Language-specific character normalization
- Number-to-words expansion
- Punctuation handling

## Output Files

| File Path | Description |
|-----------|-------------|
| **`output/checkpoints/tacotron_model.pt`** | Latest main checkpoint (saved every `save_interval` epochs) |
| **`output/checkpoints/tacotron_model_epoch_<N>.pt`** | Backup checkpoints (saved every `backup_interval` epochs) |
| **`output/checkpoints/tacotron_model_final.pt`** | Final checkpoint at end of training |
| **`output/logs/training.log`** | Detailed text logs of training progress |
| **`output/logs/events.out.tfevents.*`** | TensorBoard event files (loss, learning rate, alignments) |

## Common Issues and Solutions

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| **`FileNotFoundError`** | Incorrect paths in config | Double-check all paths in `config.yaml` |
| | Missing dataset files | Verify `metadata.csv` and `wavs` directory exist |
| **`CUDA Out of Memory`** | Batch size too large | Reduce `batch_size` in `config.yaml` |
| | Other GPU applications running | Close other GPU-intensive applications |
| **Tensor Shape Mismatch** | Sampling rate mismatch | Ensure `sampling_rate` matches ALL `.wav` files |
| | Incorrect text cleaner | Verify `text_cleaners` matches dataset language |
| | Mel generation issues | Check for errors during mel generation |
| **`NaN` Loss** | Learning rate too high | Reduce `lr_schedule_A` |
| | Data issues | Check for silent or corrupt audio files |
| | FP16 instability | Try with `fp16_run: False` |
| **Slow Training** | Not using pre-generated mels | Set `generate_mels: True` once, then `load_mel_from_disk: True` |
| | Insufficient workers | Adjust `num_workers` based on CPU cores |
| | GPU underutilization | Check with `nvidia-smi` that GPU is being used |
| **Poor Audio Quality** | Insufficient training | Train for more epochs |
| | Incorrect language cleaner | Verify `text_cleaners` is appropriate |
| | Data quality issues | Ensure high-quality, clean audio data |
| | Hyperparameter issues | Tune learning rate, dropout, etc. |
| **Distributed Training Errors** | Environment setup | Check NCCL installation and firewall settings |
| | Variable issues | Set proper environment variables (MASTER_ADDR, MASTER_PORT) |

## Training Workflow

1. **Dataset Preparation:**
   - Organize `.wav` files in the `wavs` directory
   - Create `metadata.csv` with proper format
   - Ensure consistent sampling rate across all files

2. **Initial Configuration:**
   - Set `base_project_path`, `dataset_name` correctly
   - Choose appropriate `text_cleaners` for your language
   - Set `generate_mels: True` and `load_mel_from_disk: True`
   - Adjust `batch_size` based on your GPU memory

3. **First Training Run:**
   ```bash
   python train.py
   ```
   - This will:
     - Generate mel spectrograms (`.npy` files)
     - Create/update `list.txt`
     - Begin initial training

4. **Subsequent Training Runs:**
   - Set `generate_mels: False` to skip mel generation
   - Keep `load_mel_from_disk: True` for faster training
   - Run `python train.py` to continue training

5. **Monitoring Progress:**
   ```bash
   tensorboard --logdir output/logs
   ```
   - View loss curves
   - Check alignment plots
   - Monitor learning rate

6. **Fine-tuning (if needed):**
   - Adjust learning rate if training unstable
   - Increase/decrease dropout if overfitting/underfitting
   - Try different batch sizes for performance

7. **Checkpointing:**
   - Main checkpoint: `tacotron_model.pt`
   - Backup checkpoints: `tacotron_model_epoch_<N>.pt`
   - Final model: `tacotron_model_final.pt`

For more details about the Tacotron 2 architecture, refer to the [original paper](https://arxiv.org/abs/1712.05884).
