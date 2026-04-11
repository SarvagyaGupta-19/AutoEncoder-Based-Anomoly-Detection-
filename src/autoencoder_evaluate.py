"""
Evaluation & Anomaly Scoring module.

Implements two complementary scoring methods:
    1. Reconstruction Error:  How well the autoencoder reconstructs the input.
       → High error = the model hasn't seen this pattern = anomaly.

    2. Mahalanobis Distance:  How far the encoder's latent vector is from the
       "normal" distribution learned from training data.
       → Large distance = unusual feature combination = anomaly.

Workflow:
    A) fit_anomaly_detector()   — run once after training
    B) score_audio()            — run on any new audio file
    C) evaluate_test_set()      — batch evaluation with metrics

Usage:
    python -m src.autoencoder_evaluate --fit           # fit detector on training data
    python -m src.autoencoder_evaluate --test          # evaluate test sets
    python -m src.autoencoder_evaluate --score <file>  # score a single audio file
"""

import os
import sys
import json
import argparse
import glob
import numpy as np
from sklearn.decomposition import PCA
import joblib
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from tensorflow.keras.models import load_model

from src.preprocessing import (
    audio_to_model_input,
    load_npy_dataset,
    load_spectrogram_npy,
)


# ────────────────────────────────────────────────────────────
# ANOMALY DETECTOR: Fit on normal training data
# ────────────────────────────────────────────────────────────

def fit_anomaly_detector():
    """
    Fit the anomaly detection pipeline on normal training data.

    Steps:
        1. Load trained encoder
        2. Extract latent features from all training spectrograms
        3. Fit PCA for dimensionality reduction
        4. Compute mean vector and inverse covariance matrix for Mahalanobis
        5. Compute reconstruction error distribution on training data
        6. Derive adaptive thresholds from the training distribution
        7. Save all artifacts to models/ directory

    Must be run AFTER training and BEFORE scoring.
    """
    print("\n" + "=" * 60)
    print("  FITTING ANOMALY DETECTOR")
    print("=" * 60)

    # ── Load encoder ─────────────────────────────────────────
    if not os.path.exists(config.ENCODER_PATH):
        print("  ✗ Encoder not found. Run training first: python -m src.autoencoder_train")
        sys.exit(1)

    encoder = load_model(config.ENCODER_PATH, compile=False)
    print(f"  ✓ Loaded encoder from {config.ENCODER_PATH}")

    # Also load autoencoder for reconstruction error baseline
    autoencoder = load_model(config.AUTOENCODER_PATH, compile=False)

    # ── Load training spectrograms ───────────────────────────
    print("\n── Extracting features from training data ──")
    X_train = load_npy_dataset(config.NPY_TRAIN_DIR)
    print(f"  Training samples: {len(X_train)}")

    # ── Extract latent features ──────────────────────────────
    print("  Computing encoder features...")
    latent_features = encoder.predict(X_train, batch_size=config.BATCH_SIZE, verbose=1)
    print(f"  Latent feature shape: {latent_features.shape}")

    # ── Compute reconstruction errors on training data ───────
    print("  Computing reconstruction errors...")
    reconstructions = autoencoder.predict(X_train, batch_size=config.BATCH_SIZE, verbose=1)
    recon_errors = np.mean((X_train - reconstructions) ** 2, axis=(1, 2, 3))
    print(f"  Recon error range: [{recon_errors.min():.6f}, {recon_errors.max():.6f}]")
    print(f"  Recon error mean:  {recon_errors.mean():.6f} ± {recon_errors.std():.6f}")

    # ── PCA Reduction ────────────────────────────────────────
    n_components = min(config.PCA_COMPONENTS, latent_features.shape[1], len(X_train))
    print(f"\n  Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    latent_pca = pca.fit_transform(latent_features)
    explained = sum(pca.explained_variance_ratio_) * 100
    print(f"  Variance explained: {explained:.1f}%")

    # ── Mahalanobis statistics ───────────────────────────────
    print("  Computing Mahalanobis statistics...")
    mean_vec = np.mean(latent_pca, axis=0)
    cov_matrix = np.cov(latent_pca, rowvar=False)
    # Regularize for numerical stability
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov_matrix)

    # Compute Mahalanobis distances on training data (for threshold calibration)
    train_distances = []
    for feat in latent_pca:
        delta = feat - mean_vec
        dist = float(np.sqrt(delta @ inv_cov @ delta.T))
        train_distances.append(dist)
    train_distances = np.array(train_distances)

    print(f"  Mahalanobis distance range: [{train_distances.min():.2f}, "
          f"{train_distances.max():.2f}]")
    print(f"  Mahalanobis distance mean:  {train_distances.mean():.2f} "
          f"± {train_distances.std():.2f}")

    # ── Adaptive thresholds ──────────────────────────────────
    threshold_warning = float(np.percentile(train_distances,
                                            config.ANOMALY_PERCENTILE_WARNING))
    threshold_critical = float(np.percentile(train_distances,
                                             config.ANOMALY_PERCENTILE_CRITICAL))

    recon_threshold_warning = float(np.percentile(recon_errors,
                                                  config.ANOMALY_PERCENTILE_WARNING))
    recon_threshold_critical = float(np.percentile(recon_errors,
                                                   config.ANOMALY_PERCENTILE_CRITICAL))

    # ── Save all artifacts ───────────────────────────────────
    print("\n── Saving anomaly detection artifacts ──")

    joblib.dump(pca, config.PCA_PATH)
    print(f"  ✓ PCA model → {config.PCA_PATH}")

    anomaly_stats = {
        "mean_vec": mean_vec,
        "inv_cov": inv_cov,
        "threshold_warning": threshold_warning,
        "threshold_critical": threshold_critical,
        "recon_threshold_warning": recon_threshold_warning,
        "recon_threshold_critical": recon_threshold_critical,
        "train_distance_mean": float(train_distances.mean()),
        "train_distance_std": float(train_distances.std()),
        "train_recon_mean": float(recon_errors.mean()),
        "train_recon_std": float(recon_errors.std()),
        "pca_variance_explained": explained,
    }
    joblib.dump(anomaly_stats, config.ANOMALY_STATS_PATH)
    print(f"  ✓ Anomaly stats → {config.ANOMALY_STATS_PATH}")

    print("\n" + "=" * 60)
    print("  ANOMALY DETECTOR FITTED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Mahalanobis thresholds:")
    print(f"    NORMAL:            < {threshold_warning:.2f}")
    print(f"    NEEDS MAINTENANCE: {threshold_warning:.2f} — {threshold_critical:.2f}")
    print(f"    ANOMALY:           > {threshold_critical:.2f}")
    print(f"\n  Reconstruction error thresholds:")
    print(f"    NORMAL:            < {recon_threshold_warning:.6f}")
    print(f"    NEEDS MAINTENANCE: {recon_threshold_warning:.6f} — "
          f"{recon_threshold_critical:.6f}")
    print(f"    ANOMALY:           > {recon_threshold_critical:.6f}")
    print("=" * 60)


# ────────────────────────────────────────────────────────────
# SCORING: Score a single audio file
# ────────────────────────────────────────────────────────────

class AnomalyScorer:
    """
    Stateful scorer — loads all models once, then scores many files efficiently.
    """

    def __init__(self):
        """Load all required models and statistics."""
        self.encoder = load_model(config.ENCODER_PATH, compile=False)
        self.autoencoder = load_model(config.AUTOENCODER_PATH, compile=False)
        self.pca = joblib.load(config.PCA_PATH)
        stats = joblib.load(config.ANOMALY_STATS_PATH)

        self.mean_vec = stats["mean_vec"]
        self.inv_cov = stats["inv_cov"]
        self.threshold_warning = stats["threshold_warning"]
        self.threshold_critical = stats["threshold_critical"]
        self.recon_threshold_warning = stats["recon_threshold_warning"]
        self.recon_threshold_critical = stats["recon_threshold_critical"]

    def score_spectrogram(self, spec_input):
        """
        Score a single model-ready spectrogram.

        Args:
            spec_input: np.ndarray of shape (H, W, 1)

        Returns:
            dict with keys: mahalanobis_score, recon_error, classification, details
        """
        # Add batch dimension
        x = np.expand_dims(spec_input, axis=0)

        # Reconstruction error
        recon = self.autoencoder.predict(x, verbose=0)
        recon_error = float(np.mean((x - recon) ** 2))

        # Latent features → PCA → Mahalanobis
        latent = self.encoder.predict(x, verbose=0).reshape(-1).astype(np.float64)
        latent_pca = self.pca.transform([latent])[0]
        delta = latent_pca - self.mean_vec
        mahal_dist = float(np.sqrt(delta @ self.inv_cov @ delta.T))

        # Classification (combine both signals)
        classification = self._classify(mahal_dist, recon_error)

        return {
            "mahalanobis_score": round(mahal_dist, 4),
            "reconstruction_error": round(recon_error, 8),
            "classification": classification,
            "details": {
                "mahal_threshold_warning": round(self.threshold_warning, 4),
                "mahal_threshold_critical": round(self.threshold_critical, 4),
                "recon_threshold_warning": round(self.recon_threshold_warning, 8),
                "recon_threshold_critical": round(self.recon_threshold_critical, 8),
            }
        }

    def score_audio_file(self, audio_path):
        """
        Score a single audio file end-to-end.

        Args:
            audio_path: Path to .wav file

        Returns:
            dict with scoring results
        """
        spec_input = audio_to_model_input(audio_path)
        result = self.score_spectrogram(spec_input)
        result["file"] = os.path.basename(audio_path)
        return result

    def score_spectrogram_file(self, spec_path):
        """
        Score a single spectrogram file (.npy or .png).

        Args:
            spec_path: Path to .npy or .png spectrogram

        Returns:
            dict with scoring results
        """
        if spec_path.endswith(".npy"):
            spec_input = load_spectrogram_npy(spec_path)
        else:
            from src.preprocessing import load_spectrogram_image
            spec_input = load_spectrogram_image(spec_path)
        result = self.score_spectrogram(spec_input)
        result["file"] = os.path.basename(spec_path)
        return result

    def _classify(self, mahal_dist, recon_error):
        """
        Combined classification using both reconstruction error (primary)
        and Mahalanobis distance (secondary).

        Reconstruction error is used as the primary signal because it is
        the direct output of the autoencoder and consistently outperforms
        Mahalanobis distance (which degrades through PCA compression).

        Mahalanobis is used as a secondary escalation signal only.
        """
        # Primary signal: reconstruction error
        recon_label = "NORMAL"
        if recon_error > self.recon_threshold_critical:
            recon_label = "ANOMALY"
        elif recon_error > self.recon_threshold_warning:
            recon_label = "NEEDS MAINTENANCE"

        # Secondary signal: Mahalanobis can only escalate, never downgrade
        mahal_label = "NORMAL"
        if mahal_dist > self.threshold_critical:
            mahal_label = "ANOMALY"
        elif mahal_dist > self.threshold_warning:
            mahal_label = "NEEDS MAINTENANCE"

        # Take the more severe of the two
        severity = {"NORMAL": 0, "NEEDS MAINTENANCE": 1, "ANOMALY": 2}
        if severity[mahal_label] >= severity[recon_label]:
            return mahal_label
        return recon_label


# ────────────────────────────────────────────────────────────
# BATCH EVALUATION: Evaluate on test sets
# ────────────────────────────────────────────────────────────

def evaluate_test_set(test_spec_dir, scorer=None):
    """
    Evaluate all spectrogram files in a test directory (.npy or .png).

    Args:
        test_spec_dir: Directory containing .npy or .png spectrograms
        scorer: AnomalyScorer instance (created if None)

    Returns:
        list of result dicts
    """
    if scorer is None:
        scorer = AnomalyScorer()

    # Support both .npy and .png files
    files = sorted(glob.glob(os.path.join(test_spec_dir, "*.npy")))
    if not files:
        files = sorted(glob.glob(os.path.join(test_spec_dir, "*.png")))
    if not files:
        print(f"  ⚠ No spectrogram files in {test_spec_dir}")
        return []

    results = []
    for fpath in tqdm(files, desc=f"Evaluating {os.path.basename(test_spec_dir)}"):
        result = scorer.score_spectrogram_file(fpath)
        results.append(result)

    # Summary statistics
    classifications = [r["classification"] for r in results]
    print(f"\n  Results for {os.path.basename(test_spec_dir)}:")
    print(f"    Total:              {len(results)}")
    print(f"    NORMAL:             {classifications.count('NORMAL')}")
    print(f"    NEEDS MAINTENANCE:  {classifications.count('NEEDS MAINTENANCE')}")
    print(f"    ANOMALY:            {classifications.count('ANOMALY')}")

    # If filenames contain ground truth labels (normal/anomaly), compute accuracy
    true_labels = []
    pred_labels = []
    for r in results:
        fname = r["file"].lower()
        if "normal" in fname:
            true_labels.append("NORMAL")
            pred_labels.append("NORMAL" if r["classification"] == "NORMAL" else "ANOMALY")
        elif "anomaly" in fname:
            true_labels.append("ANOMALY")
            pred_labels.append("ANOMALY" if r["classification"] != "NORMAL" else "NORMAL")

    if true_labels:
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        accuracy = correct / len(true_labels) * 100
        print(f"\n    Ground truth analysis (from filenames):")
        print(f"    Accuracy: {correct}/{len(true_labels)} = {accuracy:.1f}%")

        # Count true/false positives/negatives
        tp = sum(1 for t, p in zip(true_labels, pred_labels)
                 if t == "ANOMALY" and p == "ANOMALY")
        fn = sum(1 for t, p in zip(true_labels, pred_labels)
                 if t == "ANOMALY" and p == "NORMAL")
        fp = sum(1 for t, p in zip(true_labels, pred_labels)
                 if t == "NORMAL" and p == "ANOMALY")
        tn = sum(1 for t, p in zip(true_labels, pred_labels)
                 if t == "NORMAL" and p == "NORMAL")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"    Precision: {precision:.3f}")
        print(f"    Recall:    {recall:.3f}")
        print(f"    F1 Score:  {f1:.3f}")

    return results


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Evaluation")
    parser.add_argument("--fit", action="store_true",
                        help="Fit anomaly detector on training data")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate on source_test and target_test sets")
    parser.add_argument("--score", type=str, default=None,
                        help="Score a single audio file")
    args = parser.parse_args()

    if args.fit:
        fit_anomaly_detector()

    elif args.test:
        print("\n" + "=" * 60)
        print("  EVALUATING TEST SETS")
        print("=" * 60)
        scorer = AnomalyScorer()

        # Prefer .npy directories, fall back to .png
        source_dir = config.NPY_SOURCE_TEST_DIR
        if not glob.glob(os.path.join(source_dir, "*.npy")):
            source_dir = config.SPEC_SOURCE_TEST_DIR

        target_dir = config.NPY_TARGET_TEST_DIR
        if not glob.glob(os.path.join(target_dir, "*.npy")):
            target_dir = config.SPEC_TARGET_TEST_DIR

        if os.listdir(source_dir):
            evaluate_test_set(source_dir, scorer)

        if os.listdir(target_dir):
            evaluate_test_set(target_dir, scorer)

    elif args.score:
        if not os.path.exists(args.score):
            print(f"  ✗ File not found: {args.score}")
            sys.exit(1)

        scorer = AnomalyScorer()
        result = scorer.score_audio_file(args.score)

        print("\n" + "=" * 60)
        print(f"  RESULT: {result['file']}")
        print("=" * 60)
        print(f"  Classification:      {result['classification']}")
        print(f"  Mahalanobis Score:   {result['mahalanobis_score']}")
        print(f"  Reconstruction Err:  {result['reconstruction_error']}")
        print("=" * 60)

    else:
        parser.print_help()
