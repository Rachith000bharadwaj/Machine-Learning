"""Evaluate the saved medical diagnosis model."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "models"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "DATASETS" / "Testing.csv"
CLEANED_DATASET_PATH = PROJECT_ROOT / "DATASETS" / "cleaned" / "Testing_cleaned.csv"


def load_artifacts():
    with (MODEL_DIR / "symptom_vocabulary.json").open("r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    with (MODEL_DIR / "label_encoder.pkl").open("rb") as f:
        label_encoder = pickle.load(f)

    model = tf.keras.models.load_model(MODEL_DIR / "ml_model.h5")
    return model, label_encoder, vocabulary


def main() -> None:
    model, label_encoder, vocabulary = load_artifacts()
    dataset_path = CLEANED_DATASET_PATH if CLEANED_DATASET_PATH.exists() else DEFAULT_DATASET_PATH
    df = pd.read_csv(dataset_path)

    missing_columns = [column for column in vocabulary if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Testing dataset is missing {len(missing_columns)} symptom columns")

    X = df[list(vocabulary.keys())].to_numpy(dtype=np.float32)
    y_true = df["prognosis"].to_numpy()

    probabilities = model.predict(X, verbose=0)
    y_pred = label_encoder.inverse_transform(np.argmax(probabilities, axis=1))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("Saved Model Evaluation")
    print("======================")
    print(f"Dataset: {dataset_path.relative_to(PROJECT_ROOT)}")
    print(f"Samples: {len(df)}")
    print(f"Classes: {len(label_encoder.classes_)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    mismatches = [(actual, predicted) for actual, predicted in zip(y_true, y_pred) if actual != predicted]
    if mismatches:
        print("\nFirst mismatches:")
        for actual, predicted in mismatches[:10]:
            print(f"- actual={actual} predicted={predicted}")
    else:
        print("\nNo mismatches found on this test file.")


if __name__ == "__main__":
    main()
