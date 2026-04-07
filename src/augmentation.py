"""
Data augmentation and tf.data pipeline for Denoising Autoencoder training.

Strategy:
    The model receives a CORRUPTED spectrogram as input but must reconstruct
    the CLEAN original. This forces the autoencoder to learn the underlying
    structure of normal machine sounds rather than memorizing pixel values.

Augmentations (all safe for anomaly detection):
    - Gaussian noise: simulates background static variations
    - Time shifting:  machines don't always start at t=0
    - SpecAugment frequency masking: forces robustness to missing frequency bands
    - SpecAugment time masking: forces robustness to brief dropouts

NEVER used:
    - Pitch shifting (simulates RPM changes = mechanical failure)
"""

import os
import glob
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ────────────────────────────────────────────────────────────
# Individual augmentation functions (operate on numpy arrays)
# ────────────────────────────────────────────────────────────

def add_gaussian_noise(spec, std=None):
    """
    Add gentle Gaussian noise to a spectrogram.

    Args:
        spec: np.ndarray, shape (H, W), values in [0, 1]
        std: noise standard deviation (default: config.AUGMENT_NOISE_STD)

    Returns:
        Noisy spectrogram clipped to [0, 1]
    """
    if std is None:
        std = config.AUGMENT_NOISE_STD
    noise = np.random.normal(0, std, spec.shape).astype(np.float32)
    return np.clip(spec + noise, 0.0, 1.0)


def time_shift(spec, max_shift_frac=None):
    """
    Roll the spectrogram along the time (horizontal) axis.

    Args:
        spec: np.ndarray, shape (H, W)
        max_shift_frac: max fraction of width to shift

    Returns:
        Time-shifted spectrogram
    """
    if max_shift_frac is None:
        max_shift_frac = config.AUGMENT_TIME_SHIFT_MAX
    max_shift = int(spec.shape[1] * max_shift_frac)
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(spec, shift, axis=1)


def freq_mask(spec, max_width=None):
    """
    Zero out a random contiguous band of frequency bins (SpecAugment).

    Args:
        spec: np.ndarray, shape (H, W)
        max_width: maximum number of mel bins to mask

    Returns:
        Masked spectrogram
    """
    if max_width is None:
        max_width = config.AUGMENT_FREQ_MASK_WIDTH
    h = spec.shape[0]
    width = np.random.randint(1, max_width + 1)
    start = np.random.randint(0, max(1, h - width))
    masked = spec.copy()
    masked[start:start + width, :] = 0.0
    return masked


def time_mask(spec, max_width=None):
    """
    Zero out a random contiguous band of time frames (SpecAugment).

    Args:
        spec: np.ndarray, shape (H, W)
        max_width: maximum number of time frames to mask

    Returns:
        Masked spectrogram
    """
    if max_width is None:
        max_width = config.AUGMENT_TIME_MASK_WIDTH
    w = spec.shape[1]
    width = np.random.randint(1, max_width + 1)
    start = np.random.randint(0, max(1, w - width))
    masked = spec.copy()
    masked[:, start:start + width] = 0.0
    return masked


def apply_augmentation(spec_2d):
    """
    Apply a random combination of augmentations to a single spectrogram.

    Each augmentation is applied independently with 50% probability,
    guaranteeing variety while keeping some samples lightly augmented.

    Args:
        spec_2d: np.ndarray, shape (H, W), values in [0, 1]

    Returns:
        Augmented spectrogram, same shape, values in [0, 1]
    """
    augmented = spec_2d.copy()

    if np.random.random() < 0.5:
        augmented = add_gaussian_noise(augmented)

    if np.random.random() < 0.5:
        augmented = time_shift(augmented)

    if np.random.random() < 0.5:
        augmented = freq_mask(augmented)

    if np.random.random() < 0.5:
        augmented = time_mask(augmented)

    return augmented.astype(np.float32)


# ────────────────────────────────────────────────────────────
# tf.data.Dataset builders
# ────────────────────────────────────────────────────────────

def _load_npy_file(path_tensor):
    """Load a single .npy file from a path tensor (used inside tf.py_function)."""
    import cv2
    path = path_tensor.numpy().decode("utf-8")
    spec = np.load(path).astype(np.float32)
    if spec.shape != (config.IMG_HEIGHT, config.IMG_WIDTH):
        spec = cv2.resize(spec, (config.IMG_WIDTH, config.IMG_HEIGHT),
                          interpolation=cv2.INTER_LINEAR)
    spec = np.clip(spec, 0.0, 1.0)
    return spec


def _load_and_augment(path_tensor):
    """
    Load a .npy file and return (augmented_input, clean_target).
    The augmented input is corrupted; the target is the original clean spectrogram.
    This is the core of the denoising autoencoder approach.
    """
    clean = _load_npy_file(path_tensor)
    augmented = apply_augmentation(clean)

    # Add channel dimension: (H, W) → (H, W, 1)
    clean = np.expand_dims(clean, axis=-1)
    augmented = np.expand_dims(augmented, axis=-1)

    return augmented, clean


def _load_clean(path_tensor):
    """Load a .npy file and return (clean_input, clean_target) — no augmentation."""
    clean = _load_npy_file(path_tensor)
    clean = np.expand_dims(clean, axis=-1)
    return clean, clean


def create_training_dataset(npy_dir, batch_size=None, augment=None):
    """
    Build a tf.data.Dataset that streams training data lazily from disk.

    If augment=True (default when config.AUGMENT_ENABLED):
        Returns (corrupted_input, clean_target) pairs → denoising autoencoder.
    If augment=False:
        Returns (clean_input, clean_target) pairs → standard autoencoder.

    Args:
        npy_dir: Directory containing .npy spectrogram files
        batch_size: Batch size (default: config.BATCH_SIZE)
        augment: Enable augmentation (default: config.AUGMENT_ENABLED)

    Returns:
        tf.data.Dataset yielding (input, target) batches
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if augment is None:
        augment = config.AUGMENT_ENABLED

    # Collect all .npy file paths
    pattern = os.path.join(npy_dir, "*.npy")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")

    print(f"  Found {len(file_paths)} .npy files in {os.path.basename(npy_dir)}")

    output_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    output_sig = (
        tf.TensorSpec(shape=output_shape, dtype=tf.float32),
        tf.TensorSpec(shape=output_shape, dtype=tf.float32),
    )

    # Choose the appropriate loading function
    load_fn = _load_and_augment if augment else _load_clean

    def generator():
        # Shuffle file order each epoch
        indices = np.random.permutation(len(file_paths))
        for idx in indices:
            fpath = file_paths[idx]
            try:
                inp, tgt = load_fn(tf.constant(fpath))
                yield inp, tgt
            except Exception as e:
                print(f"  ⚠ Skipping {fpath}: {e}")
                continue

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_sig)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(file_paths)


def create_validation_dataset(npy_dir, batch_size=None):
    """
    Build a tf.data.Dataset for validation — NO augmentation.
    Input and target are both the clean spectrogram.

    Args:
        npy_dir: Directory containing .npy spectrogram files
        batch_size: Batch size (default: config.BATCH_SIZE)

    Returns:
        tf.data.Dataset yielding (clean_input, clean_target) batches
    """
    return create_training_dataset(npy_dir, batch_size=batch_size, augment=False)
