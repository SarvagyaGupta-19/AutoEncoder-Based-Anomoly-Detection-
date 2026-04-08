"""
CNN Autoencoder model for anomalous sound detection.

Architecture redesigned specifically for audio spectrograms:

Key design decisions:
    1. Strided Conv2D instead of MaxPooling
       MaxPooling discards spatial detail by taking the maximum value, which
       is fine for object recognition but ruins the precise frequency/timing
       patterns that distinguish normal vs anomalous machine sounds.
       Strided convolutions let the network *learn* how to downsample.

    2. LeakyReLU instead of ReLU
       Standard ReLU can cause "dead neurons" (outputs stuck at 0) during
       aggressive downsampling. LeakyReLU allows a small gradient to flow
       through negative activations, keeping all neurons alive and learning.

    3. No Dropout in the encoder bottleneck
       Dropout injects randomness into the latent vector, which directly
       corrupts the Mahalanobis distance calculation in evaluate.py.
       Regularization is handled instead by BatchNormalization.

    4. Smaller, cleaner latent dimension (128)
       A 256-dim latent space fed through PCA(64) means most dimensions
       are discarded anyway. 128 -> PCA(64) is more efficient and forces
       the encoder to learn a tighter, more discriminative representation.

    5. Symmetric decoder with strided Conv2DTranspose
       Mirrors the encoder exactly, ensuring the reconstruction path is
       balanced and gradients flow back cleanly through all layers.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam


# ────────────────────────────────────────────────────────────
# Latent dimension — used by both build_encoder and build_decoder
# ────────────────────────────────────────────────────────────
LATENT_DIM = 128


def _lrelu(x, name=None):
    """LeakyReLU helper (alpha=0.2 — standard for autoencoders)."""
    return layers.LeakyReLU(negative_slope=0.2, name=name)(x)


def build_encoder(input_shape=None):
    """
    Build the encoder half of the autoencoder.

    Downsampling path (strided convolutions, no MaxPooling):
        128×128×1  →  64×64×32   (stride 2)
        64×64×32   →  32×32×64   (stride 2)
        32×32×64   →  16×16×128  (stride 2)
        16×16×128  →   8×8×256   (stride 2)
        Flatten → Dense(LATENT_DIM)

    Returns:
        tf.keras.Model:
            input  → (IMG_HEIGHT, IMG_WIDTH, 1)
            output → (LATENT_DIM,) latent vector
    """
    if input_shape is None:
        input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

    inp = Input(shape=input_shape, name="encoder_input")

    # Block 1: 128×128×1 → 64×64×32
    # Large 5×5 kernel captures wider temporal context in the first layer
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same",
                      use_bias=False, name="enc_conv1")(inp)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = _lrelu(x, name="enc_act1")

    # Block 2: 64×64×32 → 32×32×64
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same",
                      use_bias=False, name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = _lrelu(x, name="enc_act2")

    # Block 3: 32×32×64 → 16×16×128
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same",
                      use_bias=False, name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = _lrelu(x, name="enc_act3")

    # Block 4: 16×16×128 → 8×8×256
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same",
                      use_bias=False, name="enc_conv4")(x)
    x = layers.BatchNormalization(name="enc_bn4")(x)
    x = _lrelu(x, name="enc_act4")

    # Flatten → compact latent vector
    # NO Dropout here — it corrupts Mahalanobis distance in evaluate.py
    x = layers.Flatten(name="enc_flatten")(x)
    latent = layers.Dense(LATENT_DIM, name="latent_vector")(x)

    encoder = Model(inp, latent, name="encoder")
    return encoder


def build_decoder(latent_dim=LATENT_DIM):
    """
    Build the decoder half — symmetric mirror of the encoder.

    Upsampling path (strided Conv2DTranspose):
        Dense → 8×8×256
        8×8×256   →  16×16×128  (stride 2)
        16×16×128 →  32×32×64   (stride 2)
        32×32×64  →  64×64×32   (stride 2)
        64×64×32  → 128×128×1   (stride 2, sigmoid output)

    Returns:
        tf.keras.Model:
            input  → (latent_dim,)
            output → (IMG_HEIGHT, IMG_WIDTH, 1) reconstructed spectrogram
    """
    inp = Input(shape=(latent_dim,), name="decoder_input")

    # Project back to spatial feature map: 8×8×256
    x = layers.Dense(8 * 8 * 256, use_bias=False, name="dec_dense")(inp)
    x = layers.BatchNormalization(name="dec_bn0")(x)
    x = _lrelu(x, name="dec_act0")
    x = layers.Reshape((8, 8, 256), name="dec_reshape")(x)

    # Block 1: 8×8×256 → 16×16×128
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same",
                               use_bias=False, name="dec_convT1")(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = _lrelu(x, name="dec_act1")

    # Block 2: 16×16×128 → 32×32×64
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same",
                               use_bias=False, name="dec_convT2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = _lrelu(x, name="dec_act2")

    # Block 3: 32×32×64 → 64×64×32
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same",
                               use_bias=False, name="dec_convT3")(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)
    x = _lrelu(x, name="dec_act3")

    # Block 4: 64×64×32 → 128×128×1 (output layer — sigmoid squashes to [0,1])
    x = layers.Conv2DTranspose(config.IMG_CHANNELS, (5, 5), strides=(2, 2),
                               padding="same", activation="sigmoid",
                               name="dec_output")(x)

    decoder = Model(inp, x, name="decoder")
    return decoder


def build_autoencoder():
    """
    Build the full autoencoder (encoder + decoder) and compile it.

    Returns:
        tuple: (autoencoder, encoder, decoder) — all tf.keras.Model instances
    """
    encoder = build_encoder()
    decoder = build_decoder(latent_dim=LATENT_DIM)

    # Wire encoder and decoder together
    inp = Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                name="autoencoder_input")
    latent = encoder(inp)
    reconstructed = decoder(latent)

    autoencoder = Model(inp, reconstructed, name="autoencoder")

    autoencoder.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    return autoencoder, encoder, decoder


def print_model_summary():
    """Print model architectures for inspection."""
    autoencoder, encoder, decoder = build_autoencoder()
    print("\n" + "=" * 60)
    print("  ENCODER")
    print("=" * 60)
    encoder.summary()
    print("\n" + "=" * 60)
    print("  DECODER")
    print("=" * 60)
    decoder.summary()
    print("\n" + "=" * 60)
    print("  AUTOENCODER (combined)")
    print("=" * 60)
    autoencoder.summary()

    total_params = autoencoder.count_params()
    print(f"\n  Total parameters: {total_params:,}")
    return autoencoder, encoder, decoder


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_model_summary()
