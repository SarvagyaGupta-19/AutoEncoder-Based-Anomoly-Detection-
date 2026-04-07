"""
Utility functions for the Anomalous Sound Detection system.

Contains:
    - Training history plotting
    - Spectrogram visualization
    - Score distribution plots
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def plot_training_history(history_path=None, save_path=None):
    """
    Plot training loss and validation loss curves.

    Args:
        history_path: Path to training_history.json (default: models/training_history.json)
        save_path: If provided, save plot to this path instead of displaying
    """
    import matplotlib.pyplot as plt

    if history_path is None:
        history_path = os.path.join(config.MODEL_DIR, "training_history.json")

    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["loss"], label="Train Loss", color="#3b82f6", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2)
    axes[0].set_title("Reconstruction Loss (MSE)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    if "mae" in history:
        axes[1].plot(history["mae"], label="Train MAE", color="#3b82f6", linewidth=2)
        axes[1].plot(history["val_mae"], label="Val MAE", color="#ef4444", linewidth=2)
        axes[1].set_title("Mean Absolute Error", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved plot → {save_path}")
    else:
        plt.show()


def plot_score_distribution(results, title="Anomaly Score Distribution", save_path=None):
    """
    Plot histogram of Mahalanobis scores with threshold lines.

    Args:
        results: List of result dicts from evaluate_test_set()
        title: Plot title
        save_path: If provided, save instead of display
    """
    import matplotlib.pyplot as plt

    scores = [r["mahalanobis_score"] for r in results]
    classifications = [r["classification"] for r in results]

    colors = []
    for c in classifications:
        if c == "NORMAL":
            colors.append("#10b981")
        elif c == "NEEDS MAINTENANCE":
            colors.append("#f59e0b")
        else:
            colors.append("#ef4444")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(scores)), scores, color=colors, alpha=0.8, width=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Mahalanobis Distance")
    ax.grid(True, alpha=0.3, axis="y")

    # Add threshold lines if available
    if results and "details" in results[0]:
        details = results[0]["details"]
        ax.axhline(y=details["mahal_threshold_warning"], color="#f59e0b",
                    linestyle="--", linewidth=1.5, label="Warning Threshold")
        ax.axhline(y=details["mahal_threshold_critical"], color="#ef4444",
                    linestyle="--", linewidth=1.5, label="Critical Threshold")
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved plot → {save_path}")
    else:
        plt.show()


def visualize_reconstructions(autoencoder, X_samples, n=8, save_path=None):
    """
    Show original spectrograms vs autoencoder reconstructions side by side.

    Args:
        autoencoder: Trained autoencoder model
        X_samples: Array of input spectrograms
        n: Number of samples to show
        save_path: If provided, save instead of display
    """
    import matplotlib.pyplot as plt

    n = min(n, len(X_samples))
    reconstructions = autoencoder.predict(X_samples[:n], verbose=0)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))

    for i in range(n):
        # Original
        axes[0, i].imshow(X_samples[i].squeeze(), cmap="magma", aspect="auto")
        axes[0, i].set_title(f"Original {i+1}", fontsize=9)
        axes[0, i].axis("off")

        # Reconstructed
        axes[1, i].imshow(reconstructions[i].squeeze(), cmap="magma", aspect="auto")
        mse = np.mean((X_samples[i] - reconstructions[i]) ** 2)
        axes[1, i].set_title(f"Recon (MSE: {mse:.5f})", fontsize=9)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Reconstructed", fontsize=11, fontweight="bold")

    plt.suptitle("Autoencoder Reconstructions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved plot → {save_path}")
    else:
        plt.show()


def display_spectrogram(audio_path, save_path=None):
    """
    Generate and display the log-mel spectrogram for a single audio file.

    Args:
        audio_path: Path to .wav file
        save_path: If provided, save instead of display
    """
    import matplotlib.pyplot as plt
    from src.preprocessing import audio_to_log_mel

    log_mel = audio_to_log_mel(audio_path)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(log_mel, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"Log-Mel Spectrogram: {os.path.basename(audio_path)}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Mel Frequency Bin")
    plt.colorbar(img, ax=ax, label="Normalized Amplitude")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved plot → {save_path}")
    else:
        plt.show()
