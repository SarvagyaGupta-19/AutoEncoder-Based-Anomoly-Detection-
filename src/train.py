"""
Production training pipeline for the CNN Autoencoder.

Features:
    - Automatic .npy spectrogram conversion (if not already done)
    - Streaming tf.data.Dataset (no full-dataset memory load)
    - Denoising augmentation: corrupted input → clean target
    - Validation split via separate file lists
    - EarlyStopping to prevent overfitting
    - ReduceLROnPlateau for adaptive learning rate
    - ModelCheckpoint to save best model only
    - Training history saved as JSON for later analysis
    - Saves both autoencoder and encoder separately

Usage:
    python -m src.train
    python -m src.train --epochs 50 --batch-size 16
"""

import os
import sys
import json
import glob
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.preprocessing import convert_all_datasets_npy
from src.augmentation import create_training_dataset, create_validation_dataset
from src.model import build_autoencoder


def check_data_ready():
    """Verify .npy data exists; if not, try to convert from raw audio."""
    npy_files = glob.glob(os.path.join(config.NPY_TRAIN_DIR, "*.npy"))

    if len(npy_files) == 0:
        print("\n  No .npy spectrograms found. Checking for raw audio...")
        audio_files = [f for f in os.listdir(config.RAW_TRAIN_DIR)
                       if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

        if len(audio_files) == 0:
            print("\n  ✗ ERROR: No data found!")
            print(f"    Place training audio files (.wav) in:")
            print(f"    {config.RAW_TRAIN_DIR}")
            print(f"\n    Or place pre-generated .npy spectrograms in:")
            print(f"    {config.NPY_TRAIN_DIR}")
            sys.exit(1)

        print(f"  Found {len(audio_files)} audio files. Converting to .npy spectrograms...")
        convert_all_datasets_npy()

        # Re-check after conversion
        npy_files = glob.glob(os.path.join(config.NPY_TRAIN_DIR, "*.npy"))

    print(f"  ✓ Found {len(npy_files)} training .npy spectrograms")
    return len(npy_files)


def split_files_for_validation(npy_dir, val_split=None):
    """
    Split .npy files into training and validation lists by moving
    val files to a temporary subdirectory.

    Returns:
        (train_dir, val_dir) — paths to the two directories
    """
    if val_split is None:
        val_split = config.VALIDATION_SPLIT

    all_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    np.random.seed(42)
    indices = np.random.permutation(len(all_files))

    val_count = int(len(all_files) * val_split)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    # Create validation subdirectory
    val_dir = os.path.join(npy_dir, "_val")
    os.makedirs(val_dir, exist_ok=True)

    # Symlink or copy val files (use symlinks to avoid duplication)
    import shutil
    for idx in val_indices:
        src = all_files[idx]
        dst = os.path.join(val_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Create training subdirectory
    train_dir = os.path.join(npy_dir, "_train")
    os.makedirs(train_dir, exist_ok=True)

    for idx in train_indices:
        src = all_files[idx]
        dst = os.path.join(train_dir, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    return train_dir, val_dir, len(train_indices), len(val_indices)


def train(epochs=None, batch_size=None):
    """
    Full training pipeline using tf.data streaming and denoising augmentation.

    Args:
        epochs: Override config.EPOCHS
        batch_size: Override config.BATCH_SIZE
    """
    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    print("\n" + "=" * 60)
    print("  ANOMALOUS SOUND DETECTION — TRAINING")
    print("=" * 60)

    # ── Step 1: Verify data ──────────────────────────────────
    total_files = check_data_ready()

    # ── Step 2: Split into train / validation ────────────────
    print("\n── Splitting into train / validation ──")
    train_dir, val_dir, n_train, n_val = split_files_for_validation(
        config.NPY_TRAIN_DIR
    )

    print(f"  Training samples:   {n_train}")
    print(f"  Validation samples: {n_val}")

    # ── Step 3: Build tf.data pipelines ──────────────────────
    print("\n── Building data pipelines ──")
    augment_mode = config.AUGMENT_ENABLED
    print(f"  Augmentation: {'ENABLED (denoising)' if augment_mode else 'DISABLED'}")

    train_dataset, _ = create_training_dataset(
        train_dir, batch_size=batch_size, augment=augment_mode
    )
    val_dataset, _ = create_validation_dataset(
        val_dir, batch_size=batch_size
    )

    steps_per_epoch = math.ceil(n_train / batch_size)
    validation_steps = math.ceil(n_val / batch_size)

    print(f"  Steps per epoch:    {steps_per_epoch}")
    print(f"  Validation steps:   {validation_steps}")

    # ── Step 4: Build model ──────────────────────────────────
    print("\n── Building autoencoder ──")
    autoencoder, encoder, decoder = build_autoencoder()
    print(f"  Parameters: {autoencoder.count_params():,}")

    # ── Step 5: Callbacks ────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=config.AUTOENCODER_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Step 6: Train ────────────────────────────────────────
    print(f"\n── Training for up to {epochs} epochs ──")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    if augment_mode:
        print(f"  Augmentation: Gaussian noise (σ={config.AUGMENT_NOISE_STD}), "
              f"time shift (±{config.AUGMENT_TIME_SHIFT_MAX:.0%}), "
              f"SpecAugment (freq={config.AUGMENT_FREQ_MASK_WIDTH}, "
              f"time={config.AUGMENT_TIME_MASK_WIDTH})")
    print()

    history = autoencoder.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Step 7: Save everything ──────────────────────────────
    print("\n── Saving models ──")

    autoencoder.save(config.AUTOENCODER_PATH)
    print(f"  ✓ Autoencoder → {config.AUTOENCODER_PATH}")

    encoder.save(config.ENCODER_PATH)
    print(f"  ✓ Encoder     → {config.ENCODER_PATH}")

    # Save training history as JSON
    history_path = os.path.join(config.MODEL_DIR, "training_history.json")
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  ✓ History     → {history_path}")

    # ── Step 8: Summary ──────────────────────────────────────
    best_val_loss = min(history.history["val_loss"])
    best_epoch = history.history["val_loss"].index(best_val_loss) + 1
    total_epochs = len(history.history["loss"])

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Epochs run:     {total_epochs}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  Best val_loss:  {best_val_loss:.6f}")
    print(f"  Final train_loss: {history.history['loss'][-1]:.6f}")
    print("=" * 60)

    return autoencoder, encoder, history


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CNN Autoencoder")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of epochs (default: {config.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
