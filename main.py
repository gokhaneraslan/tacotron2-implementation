import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import math
import logging
import traceback
from numpy import finfo
import torch
import torch.amp

try:
    from tacotron2.distributed import apply_gradient_allreduce
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    IS_DISTRIBUTED_AVAILABLE = True
except ImportError:
    IS_DISTRIBUTED_AVAILABLE = False
    apply_gradient_allreduce = None
    dist = None
    DistributedSampler = None
    logging.warning("Distributed training packages (distributed, torch.distributed) not found. Distributed training will be disabled.")

from torch.utils.data import DataLoader
from tacotron2.model import Tacotron2
from tacotron2.data_utils import TextMelLoader, TextMelCollate
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.logger import Tacotron2Logger
from tacotron2.hparams import create_hparams
import tacotron2.layers as layers 
from tacotron2.utils import load_wav_to_torch, load_filepaths_and_text
import numpy as np
from math import e

from tqdm import tqdm 
import wave 
import datetime 
import yaml
from pathlib import Path


# ================== Logging Setup ==================
# Define the format for log messages
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
# Get the root logger
root_logger = logging.getLogger()
# Set the minimum logging level to INFO
root_logger.setLevel(logging.INFO)

# Console Handler: Log messages to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

file_handler = None
# =======================================================

def setup_file_logger(log_path: Path):
    """Sets up the file handler to write logs to a specified file."""
    global file_handler
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True) 
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_path}")
    except Exception as e:
        logging.error(f"Failed to set up file logger at {log_path}: {e}")

def create_mels(hparams, dataset_path: Path):
    """
    Generates Mel spectrograms from WAV files in the specified path and saves them as .npy files.

    Args:
        hparams: Hyperparameter object.
        dataset_path: The main dataset path containing the 'wavs' subdirectory.
    """
    logging.info("Generating Mel spectrograms...")
    # Construct the path to the 'wavs' directory
    wavs_path = dataset_path / "wavs"

    if not wavs_path.is_dir():
        logging.error(f"Wavs directory not found at: {wavs_path}")
        return

    try:
        stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
    except Exception as e:
        logging.error(f"Error initializing TacotronSTFT: {e}")
        return

    def save_mel(wav_filepath: Path):
        try:

            audio, sampling_rate = load_wav_to_torch(str(wav_filepath))
            # Check if the sampling rate matches the expected rate
            if sampling_rate != stft.sampling_rate:
                # Warn and skip if sampling rates don't match. Alternatively, resampling could be added.
                logging.warning(f"{wav_filepath.name}: SR {sampling_rate} doesn't match target {stft.sampling_rate}. Skipping.")
                return

            # Normalize audio and prepare tensor for STFT
            audio_norm = audio / hparams.max_wav_value
            audio_norm = audio_norm.unsqueeze(0) # Add batch dimension
            # Calculate Mel spectrogram without computing gradients
            with torch.no_grad():
                melspec = stft.mel_spectrogram(audio_norm)

            # Remove batch dimension and move to CPU as numpy array
            melspec = torch.squeeze(melspec, 0).cpu().numpy()

            # Save the Mel spectrogram as a .npy file, replacing the .wav extension
            npy_filepath = wav_filepath.with_suffix('.npy') # Save the .npy file
            np.save(npy_filepath, melspec)

        except FileNotFoundError:
            logging.error(f"WAV file not found: {wav_filepath}")
        except ValueError as ve:
            logging.error(f"Value error processing {wav_filepath.name}: {ve}")
        except Exception as ex:
            logging.error(f"Unexpected error processing {wav_filepath.name}: {ex}")
            logging.debug(traceback.format_exc())

    # Find all .wav files in the directory
    wav_files = list(wavs_path.glob('*.wav'))
    logging.info(f"Found {len(wav_files)} .wav files in {wavs_path}")

    for wav_file in tqdm(wav_files, desc="Generating Mels"):
        save_mel(wav_file)

    logging.info("Finished generating Mel spectrograms.")


def reduce_tensor(tensor, n_gpus):
    """Averages the tensor across all GPUs in distributed training."""
    if not IS_DISTRIBUTED_AVAILABLE or dist is None:
        return tensor
    rt = tensor.clone()
    try:
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)

        rt /= n_gpus
    except AttributeError:

         logging.warning("torch.distributed.ReduceOp not found, trying legacy dist.reduce_op.SUM.")
         try:
             dist.all_reduce(rt, op=dist.reduce_op.SUM)
             rt /= n_gpus
         except Exception as e:
            logging.error(f"Error during distributed tensor reduction (legacy fallback): {e}. Returning original tensor.")
            return tensor 
    except Exception as e:
        logging.error(f"Error during distributed tensor reduction: {e}. Returning original tensor.")
        return tensor
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    """Initializes the distributed training environment."""

    if not IS_DISTRIBUTED_AVAILABLE:
        logging.error("Cannot initialize distributed training: Packages not available.")
        return False 

    # Check if CUDA is available (required for distributed GPU training)
    if not torch.cuda.is_available():
        logging.error("Distributed mode requires CUDA, but CUDA is not available.")
        return False # Initialization failed

    if dist is None:
        logging.error("Cannot initialize distributed training: torch.distributed is not available.")
        return False
    logging.info("Initializing Distributed Training...")
    try:
        # Set the correct GPU device for the current process
        # This ensures operations run on the assigned GPU.
        torch.cuda.set_device(rank % torch.cuda.device_count()) # Set cuda device for this process

        # Initialize the process group for distributed communication
        dist.init_process_group(
            backend=hparams.dist_backend, # Communication backend (e.g., 'nccl', 'gloo')
            init_method=hparams.dist_url, # URL for initializing the process group
            world_size=n_gpus,            # Total number of processes (GPUs)
            rank=rank,                    # Rank of the current process (0 to n_gpus-1)
            group_name=group_name)        # Optional name for the process group

        logging.info(f"Distributed training initialized successfully. Rank: {rank}, World Size: {n_gpus}")
        return True
    except Exception as e:
        logging.error(f"Error initializing distributed process group: {e}")
        logging.debug(traceback.format_exc())
        return False


def prepare_dataloaders(hparams):
    """Prepares the training and validation dataloaders."""
    logging.info("Preparing dataloaders...")
    try:

        trainset = TextMelLoader(hparams.training_files, hparams)
        valset = TextMelLoader(hparams.validation_files, hparams)

        collate_fn = TextMelCollate(hparams.n_frames_per_step)
    except FileNotFoundError as e:
        logging.error(f"Error creating dataset: {e}. Check training/validation file paths in hparams.")
        raise
    except Exception as e:
        logging.error(f"Error creating dataset or collate function: {e}")
        raise

    train_sampler = None
    shuffle = True

    if hparams.distributed_run:
        if IS_DISTRIBUTED_AVAILABLE and DistributedSampler is not None:
            train_sampler = DistributedSampler(trainset)
            shuffle = False
            logging.info("Using DistributedSampler for training data.")
        else:
            logging.warning("Distributed run is enabled but DistributedSampler is not available. Disabling distributed sampling.")

    try:
        # Create the training DataLoader
        train_loader = DataLoader(trainset,
                                  num_workers=hparams.num_workers,
                                  shuffle=shuffle,
                                  sampler=train_sampler,
                                  batch_size=hparams.batch_size,
                                  pin_memory=True, # often speeds up CPU-to-GPU transfer
                                  drop_last=True,
                                  collate_fn=collate_fn)
        logging.info("Dataloaders prepared successfully.")
        return train_loader, valset, collate_fn
    except Exception as e:
        logging.error(f"Error creating DataLoader: {e}")
        raise


def prepare_directories_and_logger(output_directory: Path, log_directory: Path, rank):
    """Creates output and log directories and sets up the logger."""
    tensorboard_logger = None # Initialize TensorBoard logger variable
    # Only the main process (rank 0) creates directories and the main logger
    if rank == 0:
        try:

            output_directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Output directory ensured: {output_directory}")

            # Create the specific log directory within the output directory
            log_dir_path = output_directory / log_directory
            log_dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Log directory ensured: {log_dir_path}")

            # Set up the file logger for general application logs
            setup_file_logger(log_dir_path / "training.log") # Set up file logger for application logs

            # Initialize Tacotron2Logger (presumably for TensorBoard)
            tensorboard_logger = Tacotron2Logger(log_dir_path) # For TensorBoard
            logging.info(f"TensorBoard logger initialized. Logs will be saved in: {log_dir_path}")

        except OSError as e:
            logging.error(f"Error creating directories: {e}. Check permissions for {output_directory}")
            raise
        except Exception as e:
            logging.error(f"Error initializing logger: {e}")
            raise
    else:
        # Other ranks do not write TensorBoard logs but continue console logging.
        logging.info(f"Rank {rank}: Skipping directory creation and TensorBoard logger initialization.")

    # Return the TensorBoard logger instance (or None for non-zero ranks)
    return tensorboard_logger # Return the TensorBoard logger


def load_model(hparams):
    """Loads the Tacotron2 model and moves it to the GPU if available."""
    logging.info("Loading Tacotron2 model...")
    try:
        # Initialize the model
        model = Tacotron2(hparams)
        # Move the model to GPU if CUDA is available
        if torch.cuda.is_available():
            model = model.cuda()
            logging.info("Model moved to CUDA.") # Model moved to CUDA.
        else:
            logging.warning("CUDA not available. Model will run on CPU.")

        # Set mask value for FP16 attention mechanism if FP16 is enabled
        if hparams.fp16_run:
            model.decoder.attention_layer.score_mask_value = finfo('float16').min
            logging.info("Set score_mask_value for FP16 run.") # Set score_mask_value for FP16 run.

        if hparams.distributed_run and IS_DISTRIBUTED_AVAILABLE and apply_gradient_allreduce:
            model = apply_gradient_allreduce(model)
            logging.info("Applied gradient allreduce wrapper for distributed training.")
        elif hparams.distributed_run:
             logging.warning("Distributed run is enabled but gradient allreduce could not be applied.")


        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.debug(traceback.format_exc())
        raise


def warm_start_model(checkpoint_path: Path, model, ignore_layers):
    """Initializes model weights from a checkpoint, ignoring the optimizer state and specified layers."""
    # Check if the checkpoint file exists
    if not checkpoint_path.is_file():
        logging.error(f"Warm start checkpoint not found: {checkpoint_path}")
        raise FileNotFoundError(f"Warm start checkpoint not found: {checkpoint_path}")

    logging.info(f"Warm starting model from checkpoint: {checkpoint_path}")
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_dict = checkpoint_dict['state_dict']

        if ignore_layers:
            model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
            logging.info(f"Ignoring layers for warm start: {ignore_layers}")

            dummy_dict = model.state_dict()
            dummy_dict.update(model_dict)
            model_dict = dummy_dict

        model.load_state_dict(model_dict, strict=False)
        logging.info("Model warm started successfully.")
        return model
    except KeyError as e:
        logging.error(f"Error loading state_dict from checkpoint {checkpoint_path}. Missing key: {e}")
        return None
    except FileNotFoundError:
        logging.error(f"Warm start checkpoint file not found at: {checkpoint_path}")
        return None
    except Exception as e:
        logging.error(f"Error during warm start from {checkpoint_path}: {e}")
        return None


def load_checkpoint(checkpoint_path: Path, model, optimizer):
    """Loads the model, optimizer state, iteration, and learning rate from a full checkpoint."""
    if not checkpoint_path.is_file():
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint: {checkpoint_path}")
    try:

        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint_dict['state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

        learning_rate = checkpoint_dict['learning_rate']
        iteration = checkpoint_dict['iteration']

        logging.info(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
        return model, optimizer, learning_rate, iteration
    except KeyError as e:
        logging.error(f"Error loading data from checkpoint {checkpoint_path}. Missing key: {e}")
        raise
    except FileNotFoundError:
         logging.error(f"Checkpoint file not found at: {checkpoint_path}")
         raise
    except Exception as e:
        logging.error(f"Error loading checkpoint from {checkpoint_path}: {e}")
        logging.debug(traceback.format_exc())
        raise


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath: Path):
    """Saves the model and optimizer state to a checkpoint file."""
    logging.info(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True) 
        save_data = {
            'iteration': iteration,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'learning_rate': learning_rate
        }

        torch.save(save_data, filepath)
        logging.info(f"Checkpoint saved successfully to {filepath}")
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received during save. Attempting to save again...")
        try:
            torch.save(save_data, filepath)
            logging.info(f"Checkpoint saved successfully after interrupt: {filepath}")
        except Exception as e_int:
            logging.error(f"Could not save checkpoint after interrupt: {e_int}")
    except OSError as e:
        logging.error(f"Error saving checkpoint to {filepath}: {e}. Check permissions or disk space.")
    except Exception as e:
        logging.error(f"Unexpected error saving checkpoint: {e}")
        logging.debug(traceback.format_exc())


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, tensorboard_logger, distributed_run, rank, epoch, epoch_start_time, learning_rate):
    """Evaluates the model on the validation set and logs the results."""
    logging.info(f"Starting validation for iteration {iteration} (Epoch {epoch})...")
    model.eval()
    val_start_time = time.perf_counter()
    val_loss = 0.0
    val_steps = 0

    val_sampler = None
    if distributed_run and IS_DISTRIBUTED_AVAILABLE and DistributedSampler is not None:
        val_sampler = DistributedSampler(valset, shuffle=False)

    try:
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=True, collate_fn=collate_fn)

        with torch.no_grad():

            for i, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False)):
                try:
                    x, y = model.parse_batch(batch)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)

                    if distributed_run:
                        reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
                    else:
                        reduced_val_loss = loss.item()

                    if math.isnan(reduced_val_loss):
                        logging.warning(f"Validation loss is NaN at step {i}. Skipping this batch loss.")
                        continue

                    val_loss += reduced_val_loss
                    val_steps += 1

                except Exception as batch_e:
                    logging.error(f"Error during validation batch {i}: {batch_e}")
                    continue

        if val_steps > 0:
            avg_val_loss = val_loss / val_steps
        else:
            logging.warning("No validation steps were completed successfully.")
            avg_val_loss = 0.0

    except Exception as e:
        logging.error(f"Error during validation data loading or setup: {e}")
        avg_val_loss = -1.0

    val_duration = time.perf_counter() - val_start_time
    epoch_duration = time.perf_counter() - epoch_start_time

    model.train()

    if rank == 0:
        log_message = (f"Epoch: {epoch} | Validation Loss: {avg_val_loss:.7f} | "
                       f"LR: {learning_rate:.6f} | Epoch Time: {epoch_duration:.2f}s | Val Time: {val_duration:.2f}s")

        logging.info(log_message)
        print(log_message)

        if tensorboard_logger is not None: 
            try:
                tensorboard_logger.log_validation(avg_val_loss, model, y, y_pred, iteration)
                logging.info("Logged validation results to TensorBoard.")
            except Exception as tb_e:
                logging.error(f"Error logging validation to TensorBoard: {tb_e}")
    return avg_val_loss


def train(output_directory: Path, log_directory: Path, checkpoint_path: Path, warm_start: bool, n_gpus,
          rank, group_name, hparams, save_interval: int, backup_interval: int, default_pretrained_path: Path):
    """Main training loop."""

    if hparams.distributed_run:
        if not init_distributed(hparams, n_gpus, rank, group_name):
            if rank == 0:
                logging.error("Distributed training initialization failed. Exiting.")
            return

    torch.manual_seed(hparams.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams.seed)

    try:
        model = load_model(hparams)
    except Exception:
        logging.error("Failed to load model. Exiting.")
        return 

    # AdamW might be a better choice in modern PyTorch.
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate,
                                 weight_decay=hparams.weight_decay) # Define optimizer

    # Setup FP16/Mixed Precision using torch.autocast
    scaler = None
    if hparams.fp16_run: 
        if torch.amp.autocast_mode.autocast("cuda") and torch.amp.grad_scaler.GradScaler('cuda'):
            scaler = torch.amp.grad_scaler.GradScaler('cuda')
            logging.info("Using torch.amp.autocast_mode.autocast('cuda') for FP16 training.")
        else:
            logging.warning("torch.amp.autocast_mode.autocast('cuda') not fully available for FP16. FP16 training might be unstable or disabled.")
            hparams.fp16_run = False

    criterion = Tacotron2Loss()

    try:
        tensorboard_logger = prepare_directories_and_logger(output_directory, Path(log_directory.name), rank)
    except Exception:
        logging.error("Failed to prepare directories or logger. Exiting.")
        if rank == 0:
             return
        else:
             time.sleep(5)
             return

    try: 
        train_loader, valset, collate_fn = prepare_dataloaders(hparams)
    except Exception:
        logging.error("Failed to prepare dataloaders. Exiting.")
        if rank == 0:
            return
        else:
            time.sleep(5)
            return

    # Load checkpoint or warm start if specified
    iteration = 0
    epoch_offset = 0
    current_learning_rate = hparams.learning_rate

    # Check if a checkpoint file exists
    if checkpoint_path.is_file(): # Load from checkpoint if it exists
        if warm_start: # Warm start: Load only model weights
            try:
                # Note: Warm start only loads model weights, not optimizer or iteration count.
                model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
                logging.info(f"Warm start successful from {checkpoint_path}. Training starts from iteration 0, epoch 0.")
            except Exception as e:
                logging.error(f"Warm start failed: {e}. Trying to load default pretrained model.")
                # If warm start fails, attempt to load the default pretrained model as a fallback.
                try:
                    pretrained_path = default_pretrained_path # Default path
                    if pretrained_path.is_file():
                         model = warm_start_model(pretrained_path, model, hparams.ignore_layers)
                         logging.info(f"Loaded default pretrained model from {pretrained_path} after warm start failure.")
                    else:
                         logging.error(f"Default pretrained model {pretrained_path} not found after warm start failure. Cannot proceed.")
                         if rank == 0: return
                         else: time.sleep(5); return
                except Exception as pretrain_e:
                     logging.error(f"Failed to load default pretrained model after warm start failure: {pretrain_e}. Cannot proceed.")
                     if rank == 0: return
                     else: time.sleep(5); return
        else: # Full checkpoint load: Load model, optimizer, iteration, LR
            try:
                model, optimizer, loaded_lr, iteration = load_checkpoint(checkpoint_path, model, optimizer)
                if hparams.use_saved_learning_rate:
                    current_learning_rate = loaded_lr # Use learning rate from checkpoint
                    logging.info(f"Using learning rate from checkpoint: {current_learning_rate}")
                else:
                    logging.info(f"Ignoring learning rate from checkpoint. Initial LR: {current_learning_rate}") # Ignore LR from checkpoint

                iteration += 1  # Start from the next iteration
                # Calculate epoch offset based on loaded iteration and dataloader length
                try: # Calculate epoch offset correctly
                     epoch_offset = max(0, iteration // len(train_loader))
                     logging.info(f"Resuming training from iteration {iteration}, epoch offset {epoch_offset}")
                except ZeroDivisionError:
                     logging.warning("Train loader has zero length. Cannot calculate epoch offset.")
                     epoch_offset = 0

            except Exception as e:
                # If loading checkpoint fails, attempt to load default pretrained model
                logging.error(f"Failed to load checkpoint: {e}. Attempting to load default pretrained model.") # Checkpoint load failed
                iteration = 0
                epoch_offset = 0
                try:
                    pretrained_path = default_pretrained_path # Default path
                    if pretrained_path.is_file():
                         model = warm_start_model(pretrained_path, model, hparams.ignore_layers)
                         logging.info(f"Loaded default pretrained model from {pretrained_path} after checkpoint load failure.")
                    else:
                         logging.warning(f"Default pretrained model {pretrained_path} not found. Starting truly from scratch.")
                except Exception as pretrain_e:
                     logging.error(f"Failed to load default pretrained model: {pretrain_e}. Starting truly from scratch.")

    elif default_pretrained_path != None: # No  default pretrained model
        logging.info("No checkpoint found. Attempting to load default pretrained model...") 
        try:
             pretrained_path = default_pretrained_path # Default path
             if pretrained_path.is_file():
                model = warm_start_model(pretrained_path, model, hparams.ignore_layers)
                logging.info(f"Loaded default pretrained model from {pretrained_path}")
             else:
                logging.warning(f"Failed to load default pretrained model from {default_pretrained_path}.")
                logging.warning(f"Are you sure from this path {default_pretrained_path}. Because this path {default_pretrained_path}")
        except:
            logging.error(f"Failed to load default pretrained model: {pretrain_e}. Starting truly from scratch.")

    else:
        logging.warning(f"Failed to load checkpoint from {checkpoint_path}")
        logging.warning(f"Failed to load default pretrained model from {default_pretrained_path}.")
        logging.warning(f"---------------- Starting truly from scratch. ----------------")

    if model == None:
        model = load_model(hparams)
        # AdamW might be a better choice in modern PyTorch.
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
        logging.warning(f"---------------- Starting truly from scratch. ----------------")
        
    model.train() # Set model to train mode

    # ================ MAIN TRAINING LOOP! ===================
    logging.info(f"Starting training for {hparams.epochs - epoch_offset} epochs...")
    global_start_time = time.perf_counter()

    # Loop through epochs
    for epoch in range(epoch_offset, hparams.epochs):
        epoch_start_time = time.perf_counter() # Record epoch start time
        logging.info(f"\n===== Starting Epoch: {epoch}/{hparams.epochs - 1} =====")

        # Set epoch for DistributedSampler (important for shuffling)
        if hparams.distributed_run and train_loader.sampler is not None and hasattr(train_loader.sampler, 'set_epoch'):
           train_loader.sampler.set_epoch(epoch) # Set epoch for distributed sampler

        # Initialize progress bar for the current epoch (only shown on rank 0)
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch}", leave=False, disable=(rank != 0)) # Progress bar for epoch

        # Loop through batches in the training loader
        for i, batch in progress_bar:
            batch_start_time = time.perf_counter() # Record batch start time
            if iteration < hparams.decay_start: # LR schedule: Before decay start
                current_learning_rate = hparams.A_
            else: # LR schedule: After decay start
                iteration_adjusted = iteration - hparams.decay_start
                # Calculate decay factor: e^(-iter/B) = exp(-iter/B)
                decay_factor = math.exp(-iteration_adjusted / hparams.B_) if hparams.B_ != 0 else 0
                current_learning_rate = (hparams.A_ * decay_factor) + hparams.C_

            # Enforce minimum learning rate
            current_learning_rate = max(hparams.min_learning_rate, current_learning_rate) # Clamp LR to minimum

            # Update learning rate in the optimizer
            for param_group in optimizer.param_groups: # Set LR for optimizer
                param_group['lr'] = current_learning_rate

            # Zero gradients before forward pass
            optimizer.zero_grad() # Zero gradients

            # Forward Pass
            try:
                x, y = model.parse_batch(batch) # Prepare batch

                # Run forward pass with Automatic Mixed Precision context if enabled
                if hparams.fp16_run and scaler: # Forward pass with autocast
                  with torch.amp.autocast_mode.autocast("cuda"): # Use autocast for FP16/mixed precision
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                else: # Normal FP32 forward pass
                    y_pred = model(x)
                    loss = criterion(y_pred, y)

            except RuntimeError as e: # Handle specific runtime errors like OOM
                if "out of memory" in str(e): # CUDA Out-of-Memory Error
                    logging.error(f"CUDA out of memory during forward pass at iteration {iteration}. "
                                  f"Batch size: {hparams.batch_size}. Try reducing batch size.")
                    # Exit gracefully on OOM.
                    if rank == 0: return # Exit on OOM
                    else: time.sleep(5); return
                else: # Other runtime errors
                    logging.error(f"Runtime error during forward pass at iteration {iteration}: {e}")
                    logging.debug(traceback.format_exc())
                    # Exit on other critical runtime errors.
                    if rank == 0: return
                    else: time.sleep(5); return
            except Exception as e: # Handle any other exceptions during forward pass
                logging.error(f"Error during forward pass or loss calculation at iteration {iteration}: {e}")
                logging.debug(traceback.format_exc())
                # Exit on unknown errors.
                if rank == 0: return
                else: time.sleep(5); return

            # Loss NaN Check
            if torch.isnan(loss): # Check if loss is NaN
                logging.warning(f"Loss is NaN at iteration {iteration}. Skipping backward pass and optimizer step.")
                # Skip gradient computation and optimizer step if loss is NaN
                iteration += 1
                continue # Skip to the next batch

            # Reduce loss across GPUs if distributed training is enabled
            if hparams.distributed_run: # Average loss across GPUs
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            # Backward Pass (Gradient Calculation)
            try:
                if hparams.fp16_run and scaler: # Backward pass with loss scaling (autocast)
                    # scaler automatically scales the loss before backward
                    scaler.scale(loss).backward()
                else: # Normal FP32 backward pass
                    loss.backward()
            except Exception as e: # Handle errors during backward pass
                logging.error(f"Error during backward pass at iteration {iteration}: {e}")
                logging.debug(traceback.format_exc())
                iteration += 1
                continue # Skip optimizer step if backward fails

            # Gradient Clipping and Optimizer Step (FP16)
            grad_norm = torch.tensor(0.0) # Initialize gradient norm
            if hparams.fp16_run and scaler: # Handle gradients and step for FP16
                try:
                    # Unscale gradients before clipping (required by clip_grad_norm_)
                    scaler.unscale_(optimizer) # Unscale gradients

                    # Clip gradients to prevent explosion
                    # clip_grad_norm_ operates on unscaled gradients.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.grad_clip_thresh) # Clip gradients

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.warning(f"Gradient norm is NaN or Inf ({grad_norm:.4f}) after clipping at iteration {iteration}. Optimizer step will likely be skipped by scaler.")
                    scaler.step(optimizer) # Scaler steps optimizer
                    scaler.update() # Update scaler state

                except RuntimeError as e: # Handle runtime errors during FP16 gradient processing
                    logging.error(f"Runtime error during FP16 grad processing/step at iteration {iteration}: {e}")
                    logging.debug(traceback.format_exc())
                    # Clear gradients and skip step if error occurs
                    optimizer.zero_grad()
                    iteration += 1
                    continue
                except Exception as e: # Handle other unexpected errors
                    logging.error(f"Unexpected error during FP16 grad processing/step at iteration {iteration}: {e}")
                    logging.debug(traceback.format_exc())
                    optimizer.zero_grad()
                    iteration += 1
                    continue

            # Gradient Clipping and Optimizer Step (FP32)
            elif not hparams.fp16_run: # Handle gradients and step for FP32
                try:
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.grad_clip_thresh) # Clip gradients

                    # Check for NaN/Inf gradients
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.warning(f"Gradient norm is NaN or Inf ({grad_norm:.4f}) at iteration {iteration}. Skipping optimizer step.")
                        optimizer.zero_grad()
                    else:
                        optimizer.step() # Normal optimizer step

                except Exception as e: # Handle errors during FP32 gradient processing/step
                    logging.error(f"Error during FP32 gradient clipping or optimizer step at iteration {iteration}: {e}")
                    logging.debug(traceback.format_exc())
                    optimizer.zero_grad()
                    iteration += 1
                    continue

            if rank == 0:
                duration = time.perf_counter() - batch_start_time # Calculate batch duration
                current_lr = optimizer.param_groups[0]['lr'] # Get current LR from optimizer
                # Log metrics to TensorBoard if logger is available
                if tensorboard_logger is not None: # Log to TensorBoard
                    try:
                         tensorboard_logger.log_training(
                             reduced_loss, grad_norm.item(), # Log loss, grad norm, LR, duration
                             current_lr, duration, iteration)
                    except Exception as tb_e:
                         logging.error(f"Error logging training to TensorBoard: {tb_e}")

                # Log metrics to console at specified intervals
                if iteration % hparams.log_interval == 0:
                    logging.info(f"Iter: {iteration} | Loss: {reduced_loss:.5f} | Grad Norm: {grad_norm.item():.4f} | LR: {current_lr:.6f} | Duration: {duration:.3f}s")

            # Increment iteration counter
            iteration += 1

            # Update progress bar postfix (only on rank 0)
            if rank == 0: # Update tqdm progress bar
                 progress_bar.set_postfix(loss=f"{reduced_loss:.4f}", lr=f"{current_learning_rate:.6f}")


        # ===== End of Epoch =====
        epoch_duration = time.perf_counter() - epoch_start_time
        logging.info(f"Epoch {epoch} finished. Duration: {epoch_duration:.2f}s")

        if epoch % hparams.get('validation_interval', 1) == 0: # Run validation periodically
             validate(model, criterion, valset, iteration,
                      hparams.batch_size, n_gpus, collate_fn, tensorboard_logger,
                      hparams.distributed_run, rank, epoch, epoch_start_time, current_learning_rate)

        # Save checkpoint (only on rank 0)
        if rank == 0: # Save checkpoints only on rank 0
             # Save main checkpoint at specified save_interval
             if save_interval > 0 and (epoch + 1) % save_interval == 0: # Save main checkpoint
                 save_filepath = checkpoint_path # Main checkpoint file path
                 save_checkpoint(model, optimizer, current_learning_rate, iteration, save_filepath)

             # Save backup checkpoint at specified backup_interval
             if backup_interval > 0 and (epoch + 1) % backup_interval == 0: # Save backup checkpoint
                 # Create backup filename with epoch number
                 backup_filepath = checkpoint_path.parent / f"{checkpoint_path.stem}_epoch_{epoch+1}{checkpoint_path.suffix}"
                 save_checkpoint(model, optimizer, current_learning_rate, iteration, backup_filepath)

             # Always save checkpoint at the end of the last epoch
             if (epoch + 1) == hparams.epochs: # Save final checkpoint
                 final_filepath = checkpoint_path.parent / f"{checkpoint_path.stem}_final{checkpoint_path.suffix}"
                 logging.info("Saving final model checkpoint...")
                 save_checkpoint(model, optimizer, current_learning_rate, iteration, final_filepath)
                 save_checkpoint(model, optimizer, current_learning_rate, iteration, checkpoint_path)


    # Training finished
    total_training_time = time.perf_counter() - global_start_time
    logging.info(f"Training finished after {hparams.epochs} epochs.")
    logging.info(f"Total training time: {str(datetime.timedelta(seconds=round(total_training_time)))}")


def check_dataset(hparams):
    """
    Checks the training and validation file lists for potential issues.
    """
    logging.info("Checking dataset file lists...")

    def check_filelist(filelist_path: str, is_training: bool):
        """ Helper function to check a specific file list. """
        logging.info(f"Checking {'Training' if is_training else 'Validation'} filelist: {filelist_path}")
        try:
            audiopaths_and_text = load_filepaths_and_text(filelist_path)
        except FileNotFoundError:
            logging.error(f"Filelist not found: {filelist_path}")
            return
        except Exception as e:
            logging.error(f"Error reading filelist {filelist_path}: {e}")
            return

        issues_found = 0
        for i, file_entry in enumerate(audiopaths_and_text):
            line_num = i + 1
            original_line = "|".join(file_entry)
            if len(file_entry) != 2:
                logging.warning(f"L{line_num} in {filelist_path}: Incorrect format '{original_line}'. Expected 'filepath|text'. Skipping entry.")
                issues_found += 1
                continue

            filepath_str, text = file_entry
            filepath = Path(filepath_str)

            expected_suffix = '.npy' if hparams.load_mel_from_disk else '.wav'
            if filepath.suffix != expected_suffix:
                 logging.warning(f"L{line_num} in {filelist_path}: File '{filepath.name}' has suffix '{filepath.suffix}', "
                                 f"but expected '{expected_suffix}' based on hparams.load_mel_from_disk={hparams.load_mel_from_disk}.")
                 issues_found += 1

            if not filepath.exists():
                logging.warning(f"L{line_num} in {filelist_path}: File does not exist: '{filepath}'")
                issues_found += 1
                continue

            if len(text.strip()) < hparams.get('min_text_length', 3):
                logging.warning(f"L{line_num} in {filelist_path}: Text seems too short: '{text}'")
                issues_found += 1

            if hparams.get('check_punctuation', True) and not text.strip().endswith(tuple(r"!?,.;:")):
                 logging.info(f"L{line_num} in {filelist_path}: Text lacks standard ending punctuation: '{text}'")

            if hparams.load_mel_from_disk and filepath.suffix == '.npy':
                try:
                    # Use mmap_mode='r' to read shape without loading the whole file into memory.
                    melspec = np.load(filepath, mmap_mode='r') # Read Mel shape
                    # Assuming shape is (mel_channels, time_steps)
                    mel_length = melspec.shape[1]
                    # Check if Mel length is below a minimum threshold (optional, controlled by hparam)
                    if mel_length < hparams.get('min_mel_length', 5): # Check minimum Mel length
                        logging.warning(f"L{line_num} in {filelist_path}: Mel spectrogram '{filepath.name}' seems too short (length: {mel_length}).")
                        issues_found += 1
                except ValueError: # Handle errors loading/reading the .npy file
                    logging.error(f"L{line_num} in {filelist_path}: Could not load or read shape of .npy file: '{filepath}'")
                    issues_found += 1
                except IndexError: # Handle unexpected array shapes
                     logging.error(f"L{line_num} in {filelist_path}: Mel spectrogram '{filepath.name}' has unexpected shape.")
                     issues_found += 1
                except Exception as e: # Handle other processing errors
                     logging.error(f"L{line_num} in {filelist_path}: Error processing .npy file '{filepath.name}': {e}")
                     issues_found += 1

        # Report summary of checks for the filelist
        if issues_found == 0:
            logging.info(f"No major issues found in {filelist_path}.")
        else:
            logging.warning(f"Found {issues_found} potential issues in {filelist_path}. Please review the warnings above.")

    # Check both training and validation file lists
    check_filelist(hparams.training_files, is_training=True) # Check training files
    # Skip checking validation if it's the same file as training
    if hparams.validation_files != hparams.training_files: # Check validation files (if different)
        check_filelist(hparams.validation_files, is_training=False)
    else:
        logging.info("Validation filelist is the same as training filelist, skipping redundant check.")

    logging.info("Finished checking dataset file lists.")

def confirm_and_create_filelist(meta_file_path: Path, wavs_base_path: Path, output_list_path: Path, use_npy_extension: bool):
    """
    Reads a metadata CSV file, verifies file paths, and creates a filelist
    file with the specified extension (.wav or .npy).

    Args:
        meta_file_path: Path to the metadata.csv file.
        wavs_base_path: Parent directory of the 'wavs' folder (usually the dataset root).
        output_list_path: Path where the generated list.txt file will be saved.
        use_npy_extension: If True, use .npy extension; otherwise, use .wav.

    Returns:
        bool: True if successful, False otherwise.
    """
    logging.info(f"Processing metadata file: {meta_file_path}")
    # Log the expected location of audio files
    logging.info(f"Expecting audio files relative to: {wavs_base_path / 'wavs'}")
    logging.info(f"Output filelist: {output_list_path}")
    logging.info(f"Using extension: {'.npy' if use_npy_extension else '.wav'}")

    output_lines = [] # List to store formatted lines for the output filelist
    expected_suffix = '.npy' if use_npy_extension else '.wav' # Determine the target file extension
    wavs_dir = wavs_base_path / "wavs" # The actual directory where audio files should be

    try:
        # Open and read the metadata file
        with open(meta_file_path, "r", encoding="utf-8") as meta_file:
            # Process each line in the metadata file
            for line_number, line in enumerate(meta_file, 1):
                try:
                    # Split the line by the delimiter '|'
                    cols = line.strip().split("|")
                    # Expect at least 3 columns (e.g., wav_basename | text1 | text2)
                    if len(cols) < 3:
                        logging.warning(f"L{line_number} in {meta_file_path}: Skipping line due to insufficient columns: '{line.strip()}'")
                        continue

                    wav_basename = cols[0]
                    text = cols[2]

                    # Construct the full file path relative to the wavs_dir
                    file_path_with_ext = wavs_dir / f"{wav_basename}{expected_suffix}"

                    if not file_path_with_ext.exists():
                         logging.warning(f"L{line_number}: File '{file_path_with_ext}' referenced in metadata not found on disk. Still adding to list.")
                         
                    # Format the output line as "filepath|text"
                    output_lines.append(f"{file_path_with_ext}|{text}\n") # Format the output line

                except IndexError: # Handle lines with incorrect number of columns
                    logging.warning(f"L{line_number} in {meta_file_path}: Error parsing line (IndexError): '{line.strip()}'")
                except Exception as e: # Handle other unexpected errors during line processing
                    logging.error(f"L{line_number} in {meta_file_path}: Unexpected error processing line '{line.strip()}': {e}")

    except FileNotFoundError: # Handle case where metadata file doesn't exist
        logging.error(f"Metadata file not found: {meta_file_path}")
        return False
    except Exception as e: # Handle other errors reading the metadata file
        logging.error(f"Error reading metadata file {meta_file_path}: {e}")
        return False

    # Write the generated list to the output file
    try:
        # Ensure the output directory exists before writing
        output_list_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        # Write all formatted lines to the filelist
        with open(output_list_path, 'w', encoding='utf-8') as f: # Write the filelist
            f.writelines(output_lines)
        logging.info(f"Successfully created filelist with {len(output_lines)} entries: {output_list_path}")
        return True
    except OSError as e: # Handle OS errors during file writing (permissions, disk space)
        logging.error(f"Error writing filelist to {output_list_path}: {e}")
        return False
    except Exception as e: # Handle other unexpected errors during file writing
        logging.error(f"Unexpected error writing filelist: {e}")
        return False


def calculate_audio_duration(dataset_path: Path):
    """Calculates the total duration of all .wav files in the dataset and logs long files."""
    # Calculate total duration of .wav files in the dataset
    logging.info(f"Calculating total audio duration in: {dataset_path / 'wavs'}")
    wavs_path = dataset_path / "wavs"
    total_duration = 0.0
    long_files_count = 0
    wav_count = 0
    max_duration_threshold = 11.0 # Threshold for logging long files (in seconds)

    try:
        # Find all .wav files
        wav_files = list(wavs_path.glob('*.wav'))
        wav_count = len(wav_files)

        # Handle case where no .wav files are found
        if wav_count == 0:
            logging.warning(f"No .wav files found in {wavs_path}. Cannot calculate duration.")
            return 0.0 # Cannot calculate duration

        # Iterate through WAV files with a progress bar
        for wav_file in tqdm(wav_files, desc="Calculating Duration"):
            try:
                # Open WAV file to read metadata
                with wave.open(str(wav_file), "rb") as wave_file:
                    frames = wave_file.getnframes()
                    rate = wave_file.getframerate()
                    # Handle invalid frame rate
                    if rate == 0: # Invalid frame rate
                        logging.warning(f"Invalid frame rate (0) for file: {wav_file.name}. Skipping duration calculation.")
                        continue
                    # Calculate duration and add to total
                    duration = frames / float(rate)
                    total_duration += duration
                    # Log files exceeding the duration threshold
                    if duration >= max_duration_threshold:
                        logging.warning(f"Audio file '{wav_file.name}' is longer than {max_duration_threshold} seconds (duration: {duration:.2f}s).")
                        long_files_count += 1
            except wave.Error as e: # Handle errors opening/reading WAV files
                logging.error(f"Error opening or reading WAV file '{wav_file.name}': {e}")
            except EOFError: # Handle potentially corrupt WAV files
                 logging.error(f"EOFError reading WAV file (possibly corrupt): '{wav_file.name}'")
            except Exception as e: # Handle other unexpected errors
                logging.error(f"Unexpected error processing file '{wav_file.name}': {e}")

        # Log summary statistics
        total_duration_str = str(datetime.timedelta(seconds=round(total_duration)))
        logging.info(f"Processed {wav_count} audio files.")
        logging.info(f"Total audio duration: {total_duration_str} ({total_duration:.2f} seconds)")
        if long_files_count > 0:
            logging.warning(f"Found {long_files_count} files longer than {max_duration_threshold} seconds.")

        return total_duration

    except FileNotFoundError: # Handle case where wavs directory doesn't exist
        logging.error(f"Wavs directory not found: {wavs_path}")
        return 0.0
    except Exception as e: # Handle other errors during the process
        logging.error(f"An error occurred during duration calculation: {e}")
        return 0.0


# ================== Main Execution Block ==================

if __name__ == "__main__":

    # --- Load Configuration from YAML ---
    config_path = Path("config/config.yaml")
    if not config_path.is_file():
        logging.error(f"Configuration file not found at: {config_path}")
        exit(1)

    with open(config_path, 'r', encoding='utf-8') as f: # Added encoding for safety
        try:
            config = yaml.safe_load(f)
            logging.info(f"Loaded configuration from: {config_path}")
        except yaml.YAMLError as exc:
            logging.error(f"Error parsing YAML configuration file: {exc}")
            exit(1)
        except Exception as e: # Catch other potential file reading errors
             logging.error(f"Error reading configuration file {config_path}: {e}")
             exit(1)

    # --- Setup Paths from Config ---
    try:
      
        BASE_PROJECT_PATH = Path(str(config['base_project_path']))
        DATASET_PATH = BASE_PROJECT_PATH / str(config['dataset_name'])
        OUTPUT_BASE_PATH = BASE_PROJECT_PATH / str(config['output_base_path'])
        OUTPUT_DIR_NAME = str(config['output_dir_name'])
        LOG_DIR_NAME = str(config['log_dir_name'])
        MODEL_FILENAME = str(config['model_filename'])
        
        metadata_file = DATASET_PATH / str(config['metadata_filename'])
        filelist_name = str(config['filelist_name'])
        training_list_file = DATASET_PATH / filelist_name

        output_directory = OUTPUT_BASE_PATH / OUTPUT_DIR_NAME
        log_directory = OUTPUT_BASE_PATH / LOG_DIR_NAME
        checkpoint_path = output_directory / f"{MODEL_FILENAME}.pt"

        # Handle optional paths safely
        default_pretrained_path_str = str(config['default_pretrained_path'])
        default_pretrained_path = Path(default_pretrained_path_str) if default_pretrained_path_str else None
        
    except KeyError as e:
        logging.error(f"Missing required key in config file paths section: {e}")
        exit(1)
    except TypeError as e:
         logging.error(f"Type error accessing config paths, check values: {e}")
         exit(1)


    # --- Create/Update HParams from Config ---
    try:
        hparams = create_hparams() # Start with default hparams

        # Update hparams directly from config
        # Data/Audio related
        hparams.training_files = str(training_list_file)
        hparams.validation_files = str(training_list_file) # Assuming same file for now
        hparams.load_mel_from_disk = bool(config['load_mel_from_disk'])
        hparams.text_cleaners = [str(config['text_cleaners'])]
        hparams.sampling_rate = float(config['sampling_rate'])
        hparams.mel_fmax = float(config['mel_fmax'])

        # Training related
        hparams.epochs = int(config['epochs'])
        hparams.batch_size = int(config['batch_size'])
        hparams.A_ = float(config['lr_schedule_A'])
        hparams.B_ = float(config['lr_schedule_B'])
        hparams.C_ = float(config['lr_schedule_C'])
        hparams.decay_start = int(config['lr_schedule_decay_start'])
        hparams.min_learning_rate = float(config['min_learning_rate'])

        # Dropout
        hparams.p_attention_dropout = float(config['p_attention_dropout'])
        hparams.p_decoder_dropout = float(config['p_decoder_dropout'])

        # Performance/Hardware
        # Check CUDA availability when setting fp16_run
        hparams.fp16_run = bool(config['fp16_run']) and torch.cuda.is_available()
        hparams.cudnn_enabled = bool(config['cudnn_enabled'])
        hparams.cudnn_benchmark = bool(config['cudnn_benchmark'])
        hparams.ignore_layers = []

        # Other hparams
        hparams.num_workers = int(config['num_workers'])
        hparams.log_interval = int(config['log_interval'])
        hparams.validation_interval = int(config['validation_interval'])
        hparams.show_alignments = bool(config['show_alignments'])  # Likely used by the TensorBoard logger
        alignment_graph_height = int(config['alignment_graph_height'])
        alignment_graph_width = int(config['alignment_graph_width'])
        hparams.dynamic_loss_scaling = bool(config['dynamic_loss_scaling'])

    except KeyError as e:
        logging.error(f"Missing required key in config file when setting HParams: {e}")
        exit(1)
    except TypeError as e:
        logging.error(f"Type error accessing config values for HParams, check values: {e}")
        exit(1)


    # --- Get other Training Config values ---
    try:
      
        generate_mels = bool(config['generate_mels'])
        warm_start = bool(config['warm_start'])
        save_interval = int(config['save_interval'])
        backup_interval = int(config['backup_interval'])
        n_gpus = int(config['n_gpus'])
        rank = int(config['rank'])  
        group_name = None
        
    except KeyError as e:
        logging.error(f"Missing required key in config file for training setup: {e}")
        exit(1)

    hparams.distributed_run = (n_gpus > 1)


    # ----- Pre-Training Preparation Steps -----
    logging.info("Starting Tacotron 2 Training Script")
    logging.info(f"Using Base Project Path: {BASE_PROJECT_PATH}")
    logging.info(f"Using Dataset Path: {DATASET_PATH}")
    logging.info(f"Output will be saved to: {output_directory}")
    logging.info(f"Logs will be saved to: {log_directory}")

    # Configure CuDNN settings if CUDA is available
    if torch.cuda.is_available(): # Configure cuDNN
        torch.backends.cudnn.enabled = hparams.cudnn_enabled
        torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
        logging.info(f"cuDNN Enabled: {hparams.cudnn_enabled}, Benchmark: {hparams.cudnn_benchmark}")
        # Verify FP16 capability before enabling
        # Check if torch.autocast components are available
        if hparams.fp16_run and not (torch.amp.autocast_mode.autocast("cuda") and torch.amp.grad_scaler.GradScaler('cuda')):
             logging.warning("FP16 run requested, but torch.amp.autocast_mode.autocast('cuda') is not fully available. Disabling FP16.")
             hparams.fp16_run = False
    else:
        # Disable CUDA-dependent features if CUDA is not available
        logging.warning("CUDA not available. Training will run on CPU.")
        hparams.fp16_run = False # FP16 requires CUDA
        hparams.distributed_run = False # Distributed GPU training requires CUDA


    # Step 1: Create initial filelist with .wav extensions from metadata
    # This might be needed for duration calculation even if loading mels from disk later.
    logging.info("Step 1: Creating initial filelist with .wav extensions from metadata...")
    if not confirm_and_create_filelist(metadata_file, DATASET_PATH, training_list_file, use_npy_extension=False):
        logging.error("Failed to create initial .wav filelist. Exiting.")
        exit(1)

    # Step 2: Calculate total audio duration and check lengths
    logging.info("\nStep 2: Calculating audio durations and checking lengths...")
    calculate_audio_duration(DATASET_PATH) # Calculate and log audio duration stats

    # Step 3: Generate Mel spectrograms if enabled
    if generate_mels: # Generate Mels if enabled
        logging.info("\nStep 3: Generating Mel Spectrograms...")
        create_mels(hparams, DATASET_PATH) # Call the Mel generation function
        if hparams.load_mel_from_disk: # Update filelist to .npy if needed
            logging.info("Updating filelist to use .npy extensions...")
            if not confirm_and_create_filelist(metadata_file, DATASET_PATH, training_list_file, use_npy_extension=True):
                 logging.error("Failed to update filelist to .npy extensions after generating mels. Exiting.")
                 exit(1)
            else:
                 logging.info("Filelist updated successfully with .npy extensions.")
    elif hparams.load_mel_from_disk:
        # If not generating mels but loading from disk, ensure the filelist points to .npy files.
        logging.info("\nStep 3: Skipping Mel generation. Updating filelist to use .npy extensions (assuming they exist)...")
        if not confirm_and_create_filelist(metadata_file, DATASET_PATH, training_list_file, use_npy_extension=True):
            logging.error("Failed to create/update filelist with .npy extensions. Exiting.")
            exit(1)

    # Step 4: Check dataset files based on the final filelist content
    logging.info("\nStep 4: Checking dataset files based on the final filelist...")
    check_dataset(hparams)

    # Step 5: Log final configuration before starting training
    logging.info("\nStep 5: Final Configuration Check & Starting Training...")
    logging.info(f"FP16 Run: {hparams.fp16_run}")
    logging.info(f"Dynamic Loss Scaling: {hparams.dynamic_loss_scaling}")
    logging.info(f"Distributed Run: {hparams.distributed_run}")
    logging.info(f"Number of GPUs: {n_gpus}")
    logging.info(f"Batch Size: {hparams.batch_size}")
    logging.info(f"Epochs: {hparams.epochs}")
    logging.info(f"Save Interval: {save_interval} epochs")
    logging.info(f"Backup Interval: {backup_interval} epochs")
    logging.info(f"Output Directory: {output_directory}")
    logging.info(f"Log Directory: {log_directory}")
    logging.info(f"Checkpoint Path: {checkpoint_path}")
    logging.info(f"Warm Start: {warm_start}")

    # Step 6: Start the training process
    try:       
        train(output_directory,
              log_directory,
              checkpoint_path,
              warm_start,
              n_gpus, rank,
              group_name,
              hparams,
              save_interval,
              backup_interval,
              default_pretrained_path
        )
        logging.info("Training process completed.")
    # --- Error Handling for the main training call ---
    except FileNotFoundError as fnf_err:
         logging.error(f"A required file or directory was not found: {fnf_err}")
         logging.error("Please check your configuration and file paths.")
    except ImportError as imp_err:
         logging.error(f"A required library is missing: {imp_err}")
         logging.error("Please ensure all dependencies are installed correctly.")
    except torch.cuda.OutOfMemoryError as oom_err: # Catch CUDA OOM Error
         logging.error(f"CUDA Out of Memory Error: {oom_err}")
         logging.error("Try reducing the batch size or model size.")
    except Exception as e:
        # Catch any unexpected errors during the training process
        logging.error(f"An unexpected error occurred during training: {e}")
        logging.error("Traceback:")
        # Log the full traceback for detailed debugging
        logging.error(traceback.format_exc())
    finally:
        # Ensure the file handler is closed when training finishes or an error occurs
        if file_handler: # Close the log file handler
             root_logger.removeHandler(file_handler)
             file_handler.close()
        logging.info("Script finished.")