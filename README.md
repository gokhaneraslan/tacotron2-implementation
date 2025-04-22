# Tacotron 2 TTS Training Implementation

A Python-based implementation for training Tacotron 2 Text-to-Speech (TTS) models. This repository provides a comprehensive framework for preparing your dataset, configuring training parameters, and running the training process with multi-language support.

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
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Example Training Commands](#example-training-commands)
- [Key Configuration Parameters (⚠️ Needs Attention!)](#key-configuration-parameters-️-needs-attention)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Text Cleaners and Language Support](#text-cleaners-and-language-support)
- [Output Files](#output-files)
- [Training Workflow](#training-workflow)
- [Advanced Features](#advanced-features)
  - [Mixed Precision Training](#mixed-precision-training)
  - [Distributed Training](#distributed-training)
  - [Warm Starting](#warm-starting)
- [License](#license)
- [Citations](#citations)

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
  - Mixed-Precision Training (FP16): Option for faster training and reduced memory usage.
  - Distributed Training: Supports multi-GPU training using `torch.distributed`.
- **Comprehensive Logging:** Monitor training through console, log files (`training.log`), and TensorBoard.

## Requirements

- Python 3.7+
- PyTorch (>= 1.7 recommended, check CUDA compatibility if using GPU)
- CUDA Toolkit & cuDNN (for GPU acceleration)
- Additional Python packages:

```bash
pip install torch torchvision torchaudio tensorboard matplotlib tensorflow numpy PyYAML tqdm num2words librosa inflect scipy
```
*(Adjust the PyTorch installation command based on your system and CUDA version - see [PyTorch Official Website](https://pytorch.org/))*

## Setup

1. **Install Git LFS (for downloading pretrained models):**
   ```bash
   git lfs install
   ```

2. **Set up virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv/Scripts/activate # Windows
   source venv/bin/activate # Linux
   ```

3. **Clone the repository:**
   ```bash
   git clone https://github.com/gokhaneraslan/tacotron2-tts-training.git
   cd tacotron2-tts-training
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## Dataset Preparation

If you're unsure how to prepare your dataset,

please check the [tts-dataset-generator](https://github.com/gokhaneraslan/tts-dataset-generator.git) repository,

which provides tools and instructions for creating properly formatted TTS datasets.

The training script expects your dataset in a specific format:

### Required Structure

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

### Metadata Format

Create a metadata file (e.g., `metadata.csv`) with three pipe-separated columns:
```
<wav_filename_without_extension>|<original_text>|<normalized_transcription>
```

**Example lines:**
```
audio1|This is the original text.|this is the normalized text
audio2|Another example.|another example
```

> **IMPORTANT:** The script primarily uses the **third column** (`normalized_transcription`) as the text input for the model. Ensure this text is clean and suitable for your target language.

### Audio Files

- Place all `.wav` audio files inside a directory named `wavs` within your main dataset directory.
- **Critical:** Ensure all `.wav` files have the **same sampling rate** as specified in `config.yaml` (`sampling_rate`).

If you don't know your audio's sample rate:
```python
import torchaudio
audio_path = "your_audio_path"  # Check one of your audio files
print(torchaudio.info(audio_path))
```

Then set:
- `sampling_rate` = sample rate from above
- `mel_fmax` = sample_rate / 2

### File List Generation

The training script will automatically generate or update a file named `list.txt` inside your dataset directory. This file links audio/mel files to their transcriptions for the data loader.

### Mel Spectrogram Generation

- If you set `generate_mels: True` in `config.yaml`, the script will process all `.wav` files and save corresponding Mel spectrograms as `.npy` files.
- If `load_mel_from_disk: True`, the script will use these pre-generated `.npy` files, which significantly speeds up data loading.

## Configuration (`config.yaml`)

Create a configuration file at `config/config.yaml`. Below is a comprehensive example with key sections:

### Paths

```yaml
# --- Paths (⚠️ CRITICAL - SET THESE CAREFULLY ⚠️) ---
base_project_path: /path/to/your/project/tacotron2-tts-training # Root directory
dataset_name: MyTTSDataset             # Name of the dataset directory
output_base_path: output                # Base directory for outputs
output_dir_name: checkpoints           # Subdirectory for model checkpoints
log_dir_name: logs                     # Subdirectory for logs
model_filename: tacotron_model          # Base name for checkpoint files
metadata_filename: metadata.csv         # Name of your metadata file
filelist_name: list.txt                # Name for the generated filelist
default_pretrained_path: /path/to/your/project/tacotron2-tts-training/pretrained_model/tacotron2_statedict.pt # Optional path for warm start
```

### Training Control

```yaml
# --- Training Control ---
epochs: 250                     # Total training epochs
warm_start: False               # True: Load only model weights, False: Load full checkpoint
save_interval: 10               # Save main checkpoint every N epochs
backup_interval: 25             # Save backup checkpoint every N epochs
validation_interval: 5          # Run validation every N epochs
log_interval: 100               # Log training progress every N iterations
```

### Data Loading & Preprocessing

```yaml
# --- Data Loading & Preprocessing (⚠️ IMPORTANT ⚠️) ---
generate_mels: True             # Generate Mel spectrograms before training
load_mel_from_disk: True        # Load pre-generated mels during training
```

### Language Selection

```yaml

# --- Language Selection (⚠️ CHOOSE ONE ⚠️) ---

language: turkish
#language: english
#language: spanish
#language: french
#language: german
#language: italian
#language: portuguese

```
### Text Cleaners Selection

```yaml
# --- Text Cleaners Selection (⚠️ CHOOSE ONE ⚠️) ---

text_cleaners: turkish_cleaners # Text cleaner(s)
#text_cleaners: english_cleaners # Text cleaner(s)
#text_cleaners: spanish_cleaners # Text cleaner(s)
#text_cleaners: french_cleaners # Text cleaner(s)
#text_cleaners: german_cleaners # Text cleaner(s)
#text_cleaners: italian_cleaners # Text cleaner(s)
#text_cleaners: portuguese_cleaners # Text cleaner(s)

```

### Hardware & Performance

```yaml
# --- Hardware & Performance ---
n_gpus: 1                       # Number of GPUs for training
rank: 0                         # Process rank for distributed training
fp16_run: True                  # Enable mixed-precision training
cudnn_enabled: True             # Enable cuDNN backend
cudnn_benchmark: False           # Enable cuDNN benchmark mode
num_workers: 4                  # CPU workers for data loading
batch_size: 4                   # Adjust based on GPU memory
```

### Audio Parameters

```yaml
# --- Audio Parameters (⚠️ CRITICAL - MATCH YOUR DATASET ⚠️) ---
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
p_attention_dropout: 0.1        # Dropout rate for attention
p_decoder_dropout: 0.1          # Dropout rate for decoder
```

### Logging & Visualization

```yaml
# --- Logging & Visualization ---
show_alignments: True           # Log alignment plots to TensorBoard
alignment_graph_height: 600     # Height of alignment plot
alignment_graph_width: 1000     # Width of alignment plot
```

## Usage

### Quick Start

1. **Configure `config/config.yaml`** with your paths and settings
2. **Prepare your dataset** as described above
3. **Start training:**
   ```bash
   python train.py
   ```
4. **Monitor progress:**
   ```bash
   tensorboard --logdir /path/to/your/project/output/logs
   ```

### Example Training Commands

| Purpose | Command | Notes |
|---------|---------|-------|
| **Standard Training** | `python train.py` | Uses settings from `config.yaml` |
| **Monitor Training** | `tensorboard --logdir output/logs` | Open URL in browser (e.g., `http://localhost:6006/`) |
| **Check Audio Sample Rate** | `python -c "import torchaudio; print(torchaudio.info('path/to/audio.wav'))"` | Verify sampling rate matches `config.yaml` |

## Key Configuration Parameters (⚠️ Needs Attention!)

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `base_project_path`, `dataset_name` | Root paths for your project | Verify these point to correct directories |
| `sampling_rate` | Audio sample rate | **MUST match** all your `.wav` files exactly |
| `text_cleaners` | Language cleaner | Must match your dataset's language |
| `batch_size` | Batch size for training | Start small (4-8) and increase if memory allows |
| `generate_mels` | Generate spectrograms | `True` for first run, `False` for subsequent runs |
| `load_mel_from_disk` | Use pre-generated mels | Keep `True` after first run for faster training |
| `warm_start` | Loading behavior | `True` to load model weights only, `False` to resume from checkpoint |

## Common Issues and Solutions

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| **`FileNotFoundError`** | Incorrect paths in config | Double-check all paths in `config.yaml` and dataset structure |
| **`CUDA Out of Memory`** | Batch size too large | Reduce `batch_size` in `config.yaml` |
| **Mismatched tensor shapes** | Sample rate mismatch or wrong cleaner | Verify `sampling_rate` matches WAVs and `text_cleaners` matches language |
| **`NaN` Loss values** | Learning rate too high or data issues | Reduce `lr_schedule_A` or check for silent/corrupt audio files |
| **Slow training** | Not using pre-generated mels | Set `generate_mels: True` once, then `load_mel_from_disk: True` |
| **Poor audio quality** | Wrong cleaner or insufficient training | Verify correct `text_cleaners` for your language and train longer |
| **Distributed training issues** | Environment setup | Configure environment variables properly and check NCCL installation |

## Text Cleaners and Language Support

The `text_cleaners` parameter selects which set of text normalization rules and character sets to use:

| Language | Cleaner Name | Features |
|----------|--------------|----------|
| Turkish | `turkish_cleaners` | Handles Turkish characters (İ, ı, etc.) |
| English | `english_cleaners` | Standard English processing |
| Spanish | `spanish_cleaners` | Spanish-specific character handling |
| French | `french_cleaners` | French accent and character support |
| German | `german_cleaners` | German-specific characters (ä, ö, ü, ß) |
| Italian | `italian_cleaners` | Italian accent handling |
| Portuguese | `portuguese_cleaners` | Portuguese-specific processing |
| Russian | `russian_cleaners` | Cyrillic character support |
| Arabic | `arabic_cleaners` | Arabic script handling |

Each cleaner typically performs:
- Lowercase conversion
- Whitespace normalization
- Language-specific character normalization
- Number-to-words expansion (using `num2words`)
- Punctuation handling

## Output Files

Training artifacts are saved in the directory specified by `output_base_path`:

| File | Path | Description |
|------|------|-------------|
| **Main Checkpoints** | `output/checkpoints/tacotron_model.pt` | Latest checkpoint, saved every `save_interval` epochs |
| **Backup Checkpoints** | `output/checkpoints/tacotron_model_epoch_<N>.pt` | Backups saved every `backup_interval` epochs |
| **Final Model** | `output/checkpoints/tacotron_model_final.pt` | Model saved at end of training |
| **Log File** | `output/logs/training.log` | Detailed training logs |
| **TensorBoard** | `output/logs/events.out.tfevents.*` | Files for TensorBoard visualization |

## Training Workflow

The script performs these steps automatically based on your `config.yaml`:

1. **Load Configuration:** Reads settings from `config/config.yaml`
2. **Create/Update Filelist:** Generates/updates `list.txt` using your `metadata.csv`
3. **Calculate Audio Duration:** Scans `.wav` files to report total duration
4. **Generate Mel Spectrograms:** Creates `.npy` files for each `.wav` file (if `generate_mels: True`)
5. **Update Filelist for Mels:** Points to `.npy` files (if `load_mel_from_disk: True`)
6. **Check Dataset:** Verifies file existence listed in the final `list.txt`
7. **Initialize Training:** Sets up model, optimizer, data loaders, logger
8. **Load Checkpoint/Warm Start:** Loads existing checkpoint or pre-trained model if configured
9. **Start Training Loop:** Begins training with regular validation and checkpoint saving

## Advanced Features

### Mixed Precision Training

Set `fp16_run: True` to enable mixed precision training, which can significantly speed up training on modern GPUs with minimal accuracy loss.

### Distributed Training

To use multiple GPUs:
1. Set `n_gpus` to the number of available GPUs
2. Ensure NCCL is installed if using Nvidia GPUs
3. Set appropriate environment variables or use tools like `torchrun`

### Warm Starting

Use the `warm_start` parameter to control how previous checkpoints are loaded:

- `warm_start: True`: Loads only model weights from `checkpoint_path` or `default_pretrained_path`. Useful for fine-tuning on a new dataset or changing optimizers.
- `warm_start: False`: Loads the entire state (model, optimizer, iteration count, learning rate). Resumes training exactly where it left off.

## License

This implementation builds upon [NVIDIA's Tacotron 2 implementation](https://github.com/NVIDIA/tacotron2), which is licensed under the [BSD 3-Clause License](https://github.com/NVIDIA/tacotron2/blob/master/LICENSE).

## Citations

```
@article{shen2018natural,
  title={Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions},
  author={Shen, Jonathan and Pang, Ruoming and Weiss, Ron J and Schuster, Mike and Jaitly, Navdeep and Yang, Zongheng and Chen, Zhifeng and Zhang, Yu and Wang, Yuxuan and Skerrv-Ryan, Rj and others},
  journal={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2018}
}

@inproceedings{
  wang2017tacotron,
  title={Tacotron: Towards End-to-End Speech Synthesis},
  author={Yuxuan Wang and RJ Skerry-Ryan and Daisy Stanton and Yonghui Wu and Ron J. Weiss and Navdeep Jaitly and Zongheng Yang and Ying Xiao and Zhifeng Chen and Samy Bengio and Quoc Le and Christopher Dean and Mengdao Yang and George F. Raffel},
  booktitle={Proceedings of Interspeech},
  year={2017}
}
```

For more details about the Tacotron 2 architecture, refer to the [original paper](https://arxiv.org/abs/1712.05884).

If you don't know how to prepare the dataset, check the [tts-dataset-generator](https://github.com/gokhaneraslan/tts-dataset-generator.git) repository.
