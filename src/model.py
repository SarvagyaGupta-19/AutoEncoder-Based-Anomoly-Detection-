"""
CNN Autoencoder model for anomalous sound detection.

Architecture:
    Encoder: Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → MaxPool → Dense(latent)
    Decoder: Dense → Reshape → Conv2DT → Conv2DT → Conv2DT → sigmoid output

The encoder output (latent vector) is used for anomaly scoring via
Mahalanobis distance in the latent space.

The full autoencoder is trained to reconstruct normal spectrograms;
high reconstruction error at inference = anomaly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam


def build_encoder(input_shape=None):
    """
    Build the encoder half of the autoencoder.

    Returns:
        tf.keras.Model with:
            input  → (IMG_HEIGHT, IMG_WIDTH, 1)
            output → (latent_dim,) flattened feature vector
    """
    if input_shape is None:
        input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)

    inp = Input(shape=input_shape, name="encoder_input")

    # Block 1: 128×128×1 → 64×64×32
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(inp)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool1")(x)

    # Block 2: 64×64×32 → 32×32×64
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool2")(x)

    # Block 3: 32×32×64 → 16×16×128
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_conv3")(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool3")(x)

    # Block 4: 16×16×128 → 8×8×256
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="enc_conv4")(x)
    x = layers.BatchNormalization(name="enc_bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool4")(x)

    # Flatten to latent vector
    x = layers.Flatten(name="enc_flatten")(x)
    x = layers.Dense(512, activation="relu", name="enc_dense1")(x)
    x = layers.Dropout(0.3, name="enc_dropout")(x)
    latent = layers.Dense(256, activation="relu", name="latent_vector")(x)

    encoder = Model(inp, latent, name="encoder")
    return encoder


def build_decoder(latent_dim=256):
    """
    Build the decoder half of the autoencoder.

    Returns:
        tf.keras.Model with:
            input  → (latent_dim,)
            output → (IMG_HEIGHT, IMG_WIDTH, 1) reconstructed spectrogram
    """
    inp = Input(shape=(latent_dim,), name="decoder_input")

    # Project back to spatial dimensions: 8×8×256
    x = layers.Dense(8 * 8 * 256, activation="relu", name="dec_dense1")(inp)
    x = layers.Reshape((8, 8, 256), name="dec_reshape")(x)

    # Block 1: 8×8×256 → 16×16×128
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), activation="relu",
                               padding="same", name="dec_convT1")(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)

    # Block 2: 16×16×128 → 32×32×64
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu",
                               padding="same", name="dec_convT2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)

    # Block 3: 32×32×64 → 64×64×32
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation="relu",
                               padding="same", name="dec_convT3")(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)

    # Block 4: 64×64×32 → 128×128×1
    x = layers.Conv2DTranspose(config.IMG_CHANNELS, (3, 3), strides=(2, 2),
                               activation="sigmoid", padding="same",
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
    decoder = build_decoder(latent_dim=256)

    # Wire them together
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
