```markdown
# Tacotron 2 TTS Training Implementation

This repository contains a Python script for training a Tacotron 2 Text-to-Speech (TTS) model. It provides a framework for preparing your dataset, configuring training parameters, and running the training process, including support for multiple languages, distributed training, and mixed-precision.

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
  - [1. Configure `config.yaml`](#1-configure-configyaml)
  - [2. Prepare Dataset](#2-prepare-dataset)
  - [3. Run Training](#3-run-training)
- [Key Configuration Parameters (Needs Attention!)](#key-configuration-parameters-needs-attention)
- [Text Cleaners and Language Support](#text-cleaners-and-language-support)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## Features

*   **Tacotron 2 Model Training:** Implements the core Tacotron 2 architecture for TTS.
*   **Configurable:** Uses a `config.yaml` file for easy management of paths, hyperparameters, and training settings.
*   **Dataset Preprocessing:** Includes utilities to:
    *   Generate Mel spectrograms from `.wav` files.
    *   Create file lists (`list.txt`) from metadata (`metadata.csv`).
    *   Check dataset integrity and file existence.
*   **Multi-Language Support:** Supports various languages through customizable text cleaners and symbol sets (e.g., Turkish, English, Spanish, Arabic, etc.).
*   **Checkpointing:** Saves model checkpoints periodically and allows resuming training or warm-starting.
*   **Warm Start:** Ability to initialize training from a pre-trained model's weights.
*   **Validation:** Performs validation runs at specified intervals.
*   **Logging:** Comprehensive logging to console, file (`training.log`), and TensorBoard.
*   **Distributed Training:** Supports multi-GPU training using `torch.distributed`.
*   **Mixed-Precision Training (FP16):** Option to use FP16 for faster training and reduced memory usage on compatible GPUs.

## Requirements

*   Python 3.7+
*   PyTorch (>= 1.7 recommended, check CUDA compatibility if using GPU)
*   NumPy
*   PyYAML (for config loading)
*   Tqdm (for progress bars)
*   Num2Words (for number expansion in text cleaners)
*   Librosa (often needed for audio processing, might be an indirect dependency)
*   TensorBoard (for visualization: `pip install tensorboard`)
*   **CUDA Toolkit & cuDNN:** Required for GPU acceleration (FP16, Distributed Training). Ensure PyTorch is installed with CUDA support.

You can typically install the Python dependencies using pip:

```bash
pip install torch torchvision torchaudio numpy pyyaml tqdm num2words librosa tensorboard
```
*(Adjust the PyTorch installation command based on your system and CUDA version - see [PyTorch Official Website](https://pytorch.org/))*

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
2.  **Install requirements:** (See [Requirements](#requirements))
3.  **Prepare your dataset:** (See [Dataset Preparation](#dataset-preparation))
4.  **Configure `config.yaml`:** Create and edit `config/config.yaml` (See [Configuration (`config.yaml`)](#configuration-configyaml))

## Dataset Preparation

The training script expects your dataset in a specific format:

1.  **Audio Files:** Place all your `.wav` audio files inside a directory named `wavs` within your main dataset directory.
    *   Example structure: `/path/to/project/MyTTSDataset/wavs/audio1.wav`, `/path/to/project/MyTTSDataset/wavs/audio2.wav`, ...
    *   **Critical:** Ensure all `.wav` files have the **same sampling rate** as specified in `config.yaml` (`sampling_rate`).

2.  **Metadata File:** Create a metadata file (e.g., `metadata.csv`) in your main dataset directory. This file links audio files to their transcriptions. Each line should follow the format:
    ```
    <wav_filename_without_extension>|<text_1>|<text_2_transcription>
    ```
    *   Example line: `audio1|This is the original text.|This is the normalized transcription.`
    *   The script primarily uses the **third column** (`text_2_transcription`) as the text input for the model. Ensure this text is clean and suitable for your target language.
    *   The delimiter is the pipe symbol (`|`).

3.  **File List (`list.txt`):** The training script (`train.py`) will automatically generate or update a file named `list.txt` (or the name specified by `filelist_name` in `config.yaml`) inside your dataset directory. This file lists the paths to the audio/mel files and their corresponding transcriptions, ready for the data loader.
    *   Format: `<path/to/audio_or_mel.ext>|<transcription>`

4.  **Mel Spectrograms (`.npy`):** (Optional but Recommended for Speed)
    *   If you set `generate_mels: True` in `config.yaml`, the script will process all `.wav` files in the `wavs` directory and save corresponding Mel spectrograms as `.npy` files in the *same* `wavs` directory (e.g., `audio1.npy`).
    *   If `load_mel_from_disk: True`, the script will expect these `.npy` files to exist and will update `list.txt` to point to them instead of `.wav` files. This significantly speeds up data loading during training after the initial generation.

## Configuration (`config.yaml`)

Create a configuration file at `config/config.yaml`. This file controls all aspects of the training process. Below are the key sections and parameters based on your provided code:

```yaml
# config/config.yaml

# --- Paths (❗ CRITICAL - SET THESE CAREFULLY ❗) ---
base_project_path: /path/to/your/project/tacotron2-implementation # Root directory where dataset and outputs reside
dataset_name: MyTTSDataset             # Name of the dataset directory (relative to base_project_path)
output_base_path: output                # Base directory for outputs (relative to base_project_path)
output_dir_name: checkpoints           # Subdirectory for model checkpoints (within output_base_path)
log_dir_name: logs                     # Subdirectory for logs (TensorBoard & file) (within output_base_path)
model_filename: tacotron_model          # Base name for saved checkpoint files (e.g., tacotron_model.pt)
metadata_filename: metadata.csv         # Name of your metadata file (inside dataset_name directory)
filelist_name: list.txt                # Name for the generated filelist (inside dataset_name directory)
default_pretrained_path: /path/to/pretrained/tacotron2_statedict.pt # Optional: Path to a pretrained model for warm start if no checkpoint exists

# --- Training Control ---
epochs: 250                     # Total number of training epochs
warm_start: False               # True: Load only model weights (from checkpoint_path or default_pretrained_path). False: Load full checkpoint (model, optimizer, iteration).
save_interval: 10               # Save main checkpoint every N epochs (0=disable)
backup_interval: 25             # Save backup checkpoint (with epoch number) every N epochs (0=disable)
validation_interval: 5          # Run validation every N epochs
log_interval: 100               # Log training progress to console every N iterations

# --- Data Loading & Preprocessing (❗ IMPORTANT ❗) ---
generate_mels: True             # True: Generate Mel spectrograms (.npy) before training starts. Set to False after first run if mels are generated.
load_mel_from_disk: True        # True: Load generated .npy Mel files during training (faster). False: Generate mels from .wav on-the-fly (slower). REQUIRES .npy files to exist if True.

# --- Language Selection (❗ CHOOSE ONE ❗) ---
text_cleaners: turkish_cleaners # Select the cleaner function corresponding to your dataset language.
#text_cleaners: english_cleaners
#text_cleaners: spanish_cleaners
# ... other languages ...

# --- Hardware & Performance ---
n_gpus: 1                       # Number of GPUs. Set > 1 for distributed training.
rank: 0                         # Process rank for distributed training (usually 0 unless launching manually).
fp16_run: True                  # Enable mixed-precision (FP16) training (requires compatible GPU & CUDA). Checked automatically.
cudnn_enabled: True             # Enable cuDNN backend.
cudnn_benchmark: True           # Enable cuDNN benchmark mode (can speed up training if input sizes are constant).
num_workers: 4                  # Number of CPU workers for data loading (adjust based on CPU cores).

# --- Audio Parameters (❗ CRITICAL - MATCH YOUR DATASET ❗) ---
sampling_rate: 22050            # The exact sampling rate of ALL your .wav files.
mel_fmax: 11025.0               # Max frequency for Mel spectrograms (usually sampling_rate / 2).

# --- Model Parameters (hparams) ---
# (These are often set within hparams.py or create_hparams function, but can be overridden here if needed)
# Example (Add specific Tacotron 2 model hparams here if not defaults):
# n_mel_channels: 80
# n_symbols: ... # Often determined by the chosen language/cleaner
# ... other model hparams ...

# --- Optimizer & Learning Rate (hparams) ---
lr_schedule_A: 1e-4             # Initial learning rate for the exponential decay schedule
lr_schedule_B: 8000             # Decay rate factor (iterations) for the LR schedule
lr_schedule_C: 0                # LR schedule offset (often 0)
lr_schedule_decay_start: 10000  # Iteration at which LR decay begins
min_learning_rate: 1e-5         # Minimum learning rate clamp

# --- Dropout (hparams) ---
p_attention_dropout: 0.1        # Dropout rate for the attention mechanism
p_decoder_dropout: 0.1          # Dropout rate for the decoder LSTM/RNNs

# --- Distributed Training (Advanced) ---
# dist_backend: nccl            # Backend for distributed comms ('nccl' recommended for Nvidia GPUs)
# dist_url: tcp://localhost:54321 # URL for initializing distributed process group

# --- Logging & Visualization ---
show_alignments: True           # Log alignment plots to TensorBoard during validation
alignment_graph_height: 600     # Height of the alignment plot in TensorBoard
alignment_graph_width: 1000     # Width of the alignment plot in TensorBoard

# --- Other HParams (often less modified) ---
# batch_size: Set this based on GPU Memory! Start lower (e.g., 4, 8) and increase if possible.
batch_size: 4
# weight_decay: 1e-6
# grad_clip_thresh: 1.0
# n_frames_per_step: 1          # Decoder steps per output frame (usually 1)
# max_wav_value: 32768.0        # Normalization factor for audio waveforms
# dynamic_loss_scaling: True    # Used with FP16 scaler (usually managed automatically)
# seed: 1234                    # Random seed for reproducibility

```

## Usage

### 1. Configure `config.yaml`

Carefully edit the `config/config.yaml` file, paying close attention to the paths, dataset details, language selection, and audio parameters. See [Configuration (`config.yaml`)](#configuration-configyaml) and [Key Configuration Parameters](#key-configuration-parameters-needs-attention).

### 2. Prepare Dataset

Ensure your dataset (`.wav` files and `metadata.csv`) is structured correctly as described in [Dataset Preparation](#dataset-preparation).

### 3. Run Training

Navigate to the repository's root directory in your terminal and run the main training script:

```bash
python train.py
```

The script will perform the following steps automatically based on your `config.yaml`:

1.  **Load Configuration:** Reads settings from `config/config.yaml`.
2.  **Create/Update Filelist:** Generates/updates `list.txt` using your `metadata.csv`. Initially points to `.wav` files.
3.  **Calculate Audio Duration:** Scans `.wav` files to report total duration and warn about very long files.
4.  **Generate Mel Spectrograms (if `generate_mels: True`):** Creates `.npy` files for each `.wav` file.
5.  **Update Filelist for Mels (if `load_mel_from_disk: True`):** Modifies `list.txt` to point to the `.npy` files.
6.  **Check Dataset:** Verifies file existence listed in the final `list.txt`.
7.  **Initialize Training:** Sets up the model, optimizer, data loaders, logger, and distributed training (if `n_gpus > 1`).
8.  **Load Checkpoint/Warm Start:** Loads an existing checkpoint or uses a pre-trained model if configured.
9.  **Start Training Loop:** Begins iterating through epochs and batches, performing forward/backward passes, logging, validation, and saving checkpoints.

**To monitor training progress with TensorBoard:**

```bash
tensorboard --logdir /path/to/your/project/tacotron2-implementation/output/logs
```
Then open the provided URL (usually `http://localhost:6006/`) in your web browser.

## Key Configuration Parameters (Needs Attention!)

Getting these wrong is the most common source of errors or poor results:

*   **`base_project_path`, `dataset_name`, `output_base_path`:** Ensure these paths correctly point to your project structure. Errors here lead to "File Not Found".
*   **`metadata_filename`, `filelist_name`:** Make sure these match the actual filenames used.
*   **`sampling_rate`:** MUST match the sampling rate of ALL your `.wav` files exactly. Mismatched rates lead to errors or nonsensical audio output.
*   **`text_cleaners`:** MUST match the language and character set of your dataset's transcriptions (`metadata.csv`, 3rd column). Using the wrong cleaner will cause errors or poor pronunciation. See [Text Cleaners and Language Support](#text-cleaners-and-language-support).
*   **`generate_mels` & `load_mel_from_disk`:** Understand the workflow.
    *   First run: `generate_mels: True`, `load_mel_from_disk: True` (or `False` if you want to test on-the-fly generation). This creates `.npy` files.
    *   Subsequent runs: `generate_mels: False`, `load_mel_from_disk: True`. This skips generation and loads the existing `.npy` files, which is much faster.
*   **`batch_size`:** Needs to be adjusted based on your GPU memory. Start small (e.g., 4, 8, 16) and increase carefully. If you get "CUDA Out of Memory" errors, reduce this value.
*   **`n_gpus`:** Set to the number of GPUs you want to use. If > 1, ensure distributed training prerequisites (NCCL, etc.) are met.
*   **`warm_start` & `checkpoint_path` / `default_pretrained_path`:** Understand the difference:
    *   `warm_start: True`: Loads *only* model weights from the specified checkpoint or default path. Resets optimizer and starts training from iteration 0. Good for fine-tuning on a new dataset or changing optimizers.
    *   `warm_start: False`: Loads the *entire* state (model, optimizer, iteration count, learning rate) from `checkpoint_path`. Resumes training exactly where it left off. If `checkpoint_path` doesn't exist, it might fall back to `default_pretrained_path` (as a warm start) or start from scratch.

## Text Cleaners and Language Support

The `text_cleaners` parameter in `config.yaml` selects which set of text normalization rules and character sets (symbols) to use. This is crucial for handling language-specific characters, punctuation, and number expansion correctly.

*   **Selection:** Uncomment *only one* `text_cleaners` line in `config.yaml` corresponding to your dataset's language.
*   **Available Cleaners (based on your code):** `turkish_cleaners`, `english_cleaners`, `spanish_cleaners`, `french_cleaners`, `german_cleaners`, `italian_cleaners`, `portuguese_cleaners`, `russian_cleaners`, `arabic_cleaners`.
*   **Functionality:** Cleaners typically perform:
    *   Lowercase conversion.
    *   Whitespace normalization.
    *   Language-specific character normalization (e.g., Turkish `İ`->`i`, German `ß`).
    *   Number-to-words expansion (using `num2words`).
    *   Punctuation removal (or handling, depending on the cleaner).
*   **Symbols:** Each language typically has a corresponding list of symbols defined (e.g., `turkish_symbols`, `english_symbols`). The cleaner ensures the input text is converted into a sequence of IDs based on these valid symbols.

**Example:** For a Turkish dataset, ensure you have:
```yaml
text_cleaners: turkish_cleaners
```

## Output Files

Training artifacts are saved in the directory specified by `output_base_path` relative to `base_project_path`:

*   **Checkpoints (`output/checkpoints/`):**
    *   `tacotron_model.pt` (or `<model_filename>.pt`): The latest main checkpoint, saved every `save_interval` epochs. Contains model state, optimizer state, iteration count, and learning rate. Used for resuming.
    *   `tacotron_model_epoch_<N>.pt`: Backup checkpoints saved every `backup_interval` epochs.
    *   `tacotron_model_final.pt`: Checkpoint saved at the very end of training.
*   **Logs (`output/logs/`):**
    *   `training.log`: A text file containing detailed logs of the training process (configuration, progress, warnings, errors).
    *   **TensorBoard Files:** Event files (`events.out.tfevents.*`) containing scalar values (loss, learning rate, gradient norm), alignment plots, and potentially audio samples for monitoring in TensorBoard.

## Troubleshooting

*   **`FileNotFoundError`:** Double-check all paths in `config.yaml` (`base_project_path`, `dataset_name`, `metadata_filename`, etc.) and your dataset structure. Ensure the `metadata.csv` and `wavs` directory exist where expected. Check if `list.txt` or `.npy` files were generated correctly.
*   **`CUDA Out of Memory`:** Reduce `batch_size` in `config.yaml`. If using FP16 (`fp16_run: True`), ensure your GPU supports it well. Close other GPU-intensive applications.
*   **`RuntimeError: Mismatch in shape...` or similar Tensor errors:** Often related to data loading or model definition. Check:
    *   `sampling_rate` in `config.yaml` matches ALL `.wav` files.
    *   `text_cleaners` matches your dataset language and characters. Check `metadata.csv` for unexpected characters.
    *   Mel spectrogram generation (`create_mels`) completed without errors if `load_mel_from_disk: True`.
*   **`NaN` Loss or Gradient Norm:** Can indicate:
    *   Learning rate is too high (try reducing `lr_schedule_A`).
    *   Data issues (e.g., silent audio files, very short files, mismatched text/audio). Use `check_dataset` logs and `calculate_audio_duration` warnings.
    *   Numerical instability (FP16 can sometimes be less stable, though the script uses `GradScaler`).
*   **Slow Training:**
    *   Ensure GPU is being utilized (check `nvidia-smi`).
    *   If loading `.wav` on-the-fly (`load_mel_from_disk: False`), pre-generating mels (`generate_mels: True` once, then `load_mel_from_disk: True`) is much faster.
    *   Adjust `num_workers` based on your CPU cores (e.g., 4, 8). Too high can sometimes be slower.
    *   `cudnn_benchmark: True` might help if input sizes are consistent.
*   **Distributed Training Issues:** Requires careful setup of environment variables (MASTER_ADDR, MASTER_PORT) or using tools like `torchrun`. Ensure NCCL is installed and working if using Nvidia GPUs. Check firewall settings.
*   **Incorrect Pronunciation/Audio Quality:** This is a complex issue related to model convergence, data quality/quantity, hyperparameters, and text normalization.
    *   Train longer (more `epochs`).
    *   Ensure high-quality, clean audio data.
    *   Verify the `text_cleaners` are appropriate and normalizing text correctly (especially numbers, abbreviations, symbols).
    *   Tune hyperparameters (learning rate schedule, dropout, model size if applicable).
    *   Ensure sufficient data variety.
```
