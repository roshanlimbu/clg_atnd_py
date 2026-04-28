"""
========================================================
STEP 2 — Model Conversion (One Time Setup)
========================================================
Converts Teachable Machine TensorFlow.js model (model.json + weights.bin)
into Keras .h5 format that Python/TensorFlow can load directly.

Run this ONCE before starting the main program.
DO NOT re-run unless you retrained your Teachable Machine model.

Usage:
    python convert_model.py

Output:
    converted_model/model.h5
========================================================
"""

import os
import sys
import json
from pathlib import Path


def check_dependencies():
    """Verify all required packages are installed before conversion."""
    missing = []
    try:
        import tensorflowjs  # noqa: F401
    except ImportError:
        missing.append("tensorflowjs")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow")

    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Install them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    print("[OK] All conversion dependencies found.")


def verify_source_model(models_dir: Path):
    """Check that all required Teachable Machine export files are present."""
    required_files = ["model.json", "weights.bin", "metadata.json"]
    missing_files = []

    for f in required_files:
        if not (models_dir / f).exists():
            missing_files.append(f)

    if missing_files:
        print(f"\n[ERROR] Missing model files in '{models_dir}': {missing_files}")
        print("Make sure you exported your Teachable Machine model into the models/ folder.")
        sys.exit(1)

    print(f"[OK] All source model files found in '{models_dir}'.")


def read_class_labels(models_dir: Path) -> list:
    """Read class names from Teachable Machine metadata.json."""
    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Teachable Machine metadata stores labels under 'labels' key
    labels = metadata.get("labels", [])
    print(f"[OK] Found {len(labels)} classes: {labels}")
    return labels


def convert_tfjs_to_keras(models_dir: Path, output_dir: Path) -> Path:
    """
    Core conversion function.
    Uses tensorflowjs.converters to load the TF.js graph model
    and re-save it as a Keras .h5 file.
    """
    import tensorflowjs as tfjs
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_dir / "model.h5"

    print("\n[INFO] Starting model conversion...")
    print(f"       Source : {models_dir / 'model.json'}")
    print(f"       Output : {output_h5_path}")

    # Load the TensorFlow.js layers model into a Keras model object
    keras_model = tfjs.converters.load_keras_model(
        str(models_dir / "model.json")
    )

    # Save as standard Keras HDF5 format
    keras_model.save(str(output_h5_path))

    print(f"\n[SUCCESS] Model converted and saved to: {output_h5_path}")
    return output_h5_path


def validate_converted_model(h5_path: Path, labels: list):
    """
    Validation step — load the converted .h5 model and run a dummy inference
    to confirm the model works correctly before the main program uses it.
    """
    import tensorflow as tf
    import numpy as np

    print("\n[INFO] Validating converted model with a dummy inference...")

    model = tf.keras.models.load_model(str(h5_path))

    # Teachable Machine models expect input shape (1, 224, 224, 3)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    predictions = model.predict(dummy_input, verbose=0)

    print(f"[OK] Model loaded successfully.")
    print(f"[OK] Input shape  : {model.input_shape}")
    print(f"[OK] Output shape : {model.output_shape}")
    print(f"[OK] Dummy prediction shape: {predictions.shape}")
    print(f"[OK] Number of output classes matches labels: {predictions.shape[1] == len(labels)}")

    if predictions.shape[1] != len(labels):
        print(f"\n[WARNING] Output classes ({predictions.shape[1]}) != labels count ({len(labels)})")
        print("          This may cause recognition issues. Re-check your Teachable Machine export.")
    else:
        print(f"\n[SUCCESS] Validation passed. Model outputs {predictions.shape[1]} classes.")
        for i, label in enumerate(labels):
            print(f"          Class {i}: {label}")


def save_conversion_log(models_dir: Path, output_dir: Path, labels: list):
    """Save a log file recording what was converted and when."""
    from datetime import datetime

    log_data = {
        "converted_at": datetime.now().isoformat(),
        "source_model": str(models_dir / "model.json"),
        "output_model": str(output_dir / "model.h5"),
        "classes": labels,
        "num_classes": len(labels),
    }

    log_path = output_dir / "conversion_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"\n[INFO] Conversion log saved to: {log_path}")


def main():
    print("=" * 60)
    print("  Teachable Machine → Keras Model Converter")
    print("=" * 60)

    # Resolve paths relative to this script's location
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    output_dir = base_dir / "converted_model"

    # Pre-flight checks
    check_dependencies()
    verify_source_model(models_dir)

    # Read class labels from metadata
    labels = read_class_labels(models_dir)

    # Check if already converted
    existing_h5 = output_dir / "model.h5"
    if existing_h5.exists():
        print(f"\n[WARNING] Converted model already exists at: {existing_h5}")
        answer = input("  Overwrite it? (y/n): ").strip().lower()
        if answer != "y":
            print("[INFO] Conversion skipped. Using existing model.")
            validate_converted_model(existing_h5, labels)
            return

    # Run the conversion
    h5_path = convert_tfjs_to_keras(models_dir, output_dir)

    # Validate the output
    validate_converted_model(h5_path, labels)

    # Save a log
    save_conversion_log(models_dir, output_dir, labels)

    print("\n" + "=" * 60)
    print("  Conversion complete! You can now run: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()