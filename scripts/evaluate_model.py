"""Evaluate the saved medical diagnosis model on DATASETS/Testing.csv."""

from __future__ import annotations

import json
import math
import os
import pickle
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "models"
DATASET_PATH = PROJECT_ROOT / "DATASETS" / "Testing.csv"

LABEL_FIXES = {
    "Paralysis (brain hemorrhageH": "Paralysis (brain hemorrhage)",
}


def clean_label(value: object) -> str:
    label = str(value).strip()
    return LABEL_FIXES.get(label, label)


def load_artifacts():
    with (MODEL_DIR / "symptom_vocabulary.json").open("r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    with (MODEL_DIR / "label_encoder.pkl").open("rb") as f:
        label_encoder = pickle.load(f)

    model = tf.keras.models.load_model(MODEL_DIR / "ml_model.h5")
    return model, label_encoder, vocabulary


def wilson_interval(successes: int, total: int, confidence_z: float = 1.96) -> dict[str, float]:
    if total == 0:
        return {"lower": 0.0, "upper": 0.0}
    p_hat = successes / total
    denominator = 1 + confidence_z**2 / total
    centre = p_hat + confidence_z**2 / (2 * total)
    margin = confidence_z * math.sqrt((p_hat * (1 - p_hat) + confidence_z**2 / (4 * total)) / total)
    return {
        "lower": float((centre - margin) / denominator),
        "upper": float((centre + margin) / denominator),
    }


def main() -> None:
    model, label_encoder, vocabulary = load_artifacts()
    df = pd.read_csv(DATASET_PATH)
    df["prognosis"] = df["prognosis"].map(clean_label)

    missing_columns = [column for column in vocabulary if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Testing dataset is missing {len(missing_columns)} symptom columns")

    X = df[list(vocabulary.keys())].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
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
    correct = int(np.sum(y_true == y_pred))

    labels = label_encoder.classes_.tolist()
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    mismatches = [
        {
            "row": int(index),
            "actual": str(actual),
            "predicted": str(predicted),
            "confidence": float(np.max(probabilities[index])),
        }
        for index, (actual, predicted) in enumerate(zip(y_true, y_pred))
        if actual != predicted
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "samples": int(len(df)),
        "classes": int(len(labels)),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy_95_ci": wilson_interval(correct, len(df)),
        },
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix,
        },
        "mismatches": mismatches,
    }
    output_path = MODEL_DIR / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved Model Evaluation")
    print("======================")
    print(f"Samples: {len(df)}")
    print(f"Classes: {len(labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(
        "Accuracy 95% CI: "
        f"{summary['metrics']['accuracy_95_ci']['lower']:.4f} - "
        f"{summary['metrics']['accuracy_95_ci']['upper']:.4f}"
    )
    print(f"Evaluation summary saved to {output_path}")

    if mismatches:
        print("\nFirst mismatches:")
        for item in mismatches[:10]:
            print(f"- actual={item['actual']} predicted={item['predicted']}")
    else:
        print("\nNo mismatches found on this test file.")


if __name__ == "__main__":
    main()
