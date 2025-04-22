# Tacotron 2 TTS Training Implementation

A Python-based implementation for training Tacotron 2 Text-to-Speech (TTS) models. This repository provides tools for preparing your dataset, configuring training parameters, and running the training process with multi-language support.

## Features

- **Complete Tacotron 2 Training Pipeline:** Train state-of-the-art TTS models from your own audio data
- **Multi-Language Support:** Works with English, Turkish, Spanish, Arabic, and many other languages
- **Preprocessing Tools:** Includes utilities for generating Mel spectrograms and preparing training data
- **Flexible Configuration:** Easily customize all aspects of training through a single YAML file
- **Advanced Training Features:** Supports distributed training, mixed-precision (FP16), warm-starting from pretrained models
- **Comprehensive Logging:** Monitor training progress through console, log files, and TensorBoard

## Requirements

- Python 3.7+
- PyTorch (1.7+ recommended)
- CUDA Toolkit & cuDNN (for GPU acceleration)
- Additional Python packages:
  ```
  pip install -r requirements.txt
  ```

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gokhaneraslan/tacotron2-tts-training.git
   cd tacotron2-tts-training
   ```

2. **Prepare your dataset:**
   - Place WAV files in `/path/to/project/MyTTSDataset/wavs/`
   - Create `metadata.csv` with format: `filename|original_text|normalized_text`
   - Example: `audio1|Hello world!|hello world`

3. **Configure training:**
   - Edit `config/config.yaml` with your paths and settings

4. **Start training:**
   ```bash
   python train.py
   ```

5. **Monitor progress:**
   ```bash
   tensorboard --logdir /path/to/your/project/output/logs
   ```

## Dataset Preparation

The script expects your dataset in this structure:

```
/base_project_path/
└── dataset_name/
    ├── metadata.csv
    └── wavs/
        ├── audio1.wav
        ├── audio2.wav
        └── ...
```

Where `metadata.csv` contains three pipe-separated columns:
```
audio1|This is the original text.|this is the normalized text
audio2|Another example.|another example
```

> **Important:** All WAV files must have the same sampling rate as specified in your `config.yaml`

## Configuration Guide

Create a `config/config.yaml` file with these key sections:

```yaml
# Paths - Set these carefully!
base_project_path: /path/to/your/project  # Root directory
dataset_name: MyTTSDataset                # Dataset folder name
output_base_path: output                  # Output directory
metadata_filename: metadata.csv           # Metadata file name
default_pretrained_path: /path/to/pretrained/model.pt  # Optional pretrained model

# Training Control
epochs: 250                    # Total training epochs
warm_start: False              # Load only weights (True) or full checkpoint (False)
save_interval: 10              # Save checkpoint every N epochs
validation_interval: 5         # Run validation every N epochs

# Data Processing  
generate_mels: True            # Generate mel spectrograms before training
load_mel_from_disk: True       # Load pre-generated mels (faster training)

# Language Selection - Choose one!
text_cleaners: turkish_cleaners # Text cleaner(s)
#text_cleaners: engilish_cleaners # Text cleaner(s)
#text_cleaners: spanish_cleaners # Text cleaner(s)
#text_cleaners: french_cleaners # Text cleaner(s)
#text_cleaners: german_cleaners # Text cleaner(s)
#text_cleaners: italian_cleaners # Text cleaner(s)
#text_cleaners: portuguese_cleaners # Text cleaner(s)
#text_cleaners: russian_cleaners # Text cleaner(s)
#text_cleaners: arabic_cleaners # Text cleaner(s)

# Hardware Settings
n_gpus: 1                      # Number of GPUs for training
batch_size: 4                  # Adjust based on GPU memory
fp16_run: True                 # Use mixed-precision training
num_workers: 4                 # Data loading workers

# Audio Parameters - Must match your dataset!
sampling_rate: 22050           # Sampling rate of all WAV files
```

## ⚠️ Important Settings (Pay Special Attention!)

- **`sampling_rate`**: MUST match your WAV files exactly
- **`text_cleaners`**: Select the correct language for your dataset
- **`batch_size`**: Start small (4-8) and increase if your GPU has enough memory
- **`generate_mels`**: Set to `True` for first run, then `False` to use cached mels
- **`load_mel_from_disk`**: Keep as `True` for faster training after first run
- **`base_project_path`**: Absolute path to where your project and dataset are located

## Language Support

Select the appropriate text cleaner for your dataset's language:

```yaml
# For Turkish dataset:
text_cleaners: turkish_cleaners

# For English dataset:
text_cleaners: english_cleaners

# For Spanish dataset:
text_cleaners: spanish_cleaners
```

Each cleaner handles language-specific text normalization, including:
- Converting numbers to words
- Handling special characters
- Normalizing punctuation

## Training Workflow

1. **Initial Setup**: First run with `generate_mels: True` to create mel spectrograms
2. **Regular Training**: Subsequent runs with `generate_mels: False` and `load_mel_from_disk: True`
3. **Checkpoints**: Saved to `output/checkpoints/` directory
4. **Logs**: View detailed logs in `output/logs/` and with TensorBoard

## Troubleshooting

- **Out of Memory Errors**: Reduce `batch_size` in config
- **File Not Found Errors**: Double-check all paths in `config.yaml`
- **NaN Loss**: Try reducing learning rate or check for corrupt audio files
- **Slow Training**: Ensure `load_mel_from_disk: True` and mels are pre-generated
- **Poor Audio Quality**: Verify correct `text_cleaners` for your language and increase training epochs

## Output Files

- **Model Checkpoints**: `output/checkpoints/tacotron_model.pt`
- **Backup Checkpoints**: `output/checkpoints/tacotron_model_epoch_N.pt`
- **Log Files**: `output/logs/training.log`
- **TensorBoard**: Event files in `output/logs/`

## Example Training Command

```bash
# Standard training
python train.py

# Monitor with TensorBoard
tensorboard --logdir output/logs
```

---

This implementation provides a flexible framework for training your own Tacotron 2 TTS models with multi-language support. For more details about the Tacotron 2 architecture, refer to the [original paper](https://arxiv.org/abs/1712.05884).
