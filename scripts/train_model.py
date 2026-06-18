"""Train and compare symptom-to-disease ML models.

The saved TensorFlow model remains the production model used by the Flask app.
This script also trains interpretable scikit-learn baselines so the project has
a complete ML-process record: preprocessing, model selection, validation, and
artifact generation.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "DATASETS" / "Training.csv"
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "models"
DISEASE_COLUMN = "prognosis"
RANDOM_STATE = 42

LABEL_FIXES = {
    "Paralysis (brain hemorrhageH": "Paralysis (brain hemorrhage)",
}


def clean_label(value: object) -> str:
    label = str(value).strip()
    return LABEL_FIXES.get(label, label)


def load_training_data() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    df = pd.read_csv(DATASET_PATH)
    if DISEASE_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{DISEASE_COLUMN}' in {DATASET_PATH}")

    df.columns = [str(column).strip() for column in df.columns]
    df[DISEASE_COLUMN] = df[DISEASE_COLUMN].map(clean_label)

    symptom_columns = [column for column in df.columns if column != DISEASE_COLUMN]
    for column in symptom_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(0, 1)

    before_rows = len(df)
    df = df.dropna(subset=[DISEASE_COLUMN]).drop_duplicates().reset_index(drop=True)
    duplicate_rows_removed = before_rows - len(df)

    class_counts = df[DISEASE_COLUMN].value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist()
    if rare_classes:
        df = df[~df[DISEASE_COLUMN].isin(rare_classes)].reset_index(drop=True)

    preprocessing = {
        "original_rows": int(before_rows),
        "rows_after_cleaning": int(len(df)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "label_fixes_applied": int(sum(value in LABEL_FIXES for value in pd.read_csv(DATASET_PATH)[DISEASE_COLUMN])),
        "rare_classes_removed": int(len(rare_classes)),
    }
    return df, symptom_columns, preprocessing


def build_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(output_dim, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def train_baselines(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, dict[str, float]]:
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2500, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "support_vector_machine": SVC(
            kernel="linear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    results: dict[str, dict[str, float]] = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_predictions(y_val, model.predict(X_val))
    return results


def save_validation_charts(
    df: pd.DataFrame,
    comparison: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    names = list(comparison.keys())
    accuracies = [comparison[name]["accuracy"] for name in names]
    f1_scores = [comparison[name]["f1_score"] for name in names]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, accuracies, width, label="Accuracy")
    ax.bar(x + width / 2, f1_scores, width, label="F1-score")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("_", " ").title() for name in names], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    counts = df[DISEASE_COLUMN].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 9))
    counts.plot(kind="barh", ax=ax, color="#0f766e")
    ax.set_title("Training Disease Distribution")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Disease")
    fig.tight_layout()
    fig.savefig(output_dir / "disease_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, symptom_columns, preprocessing = load_training_data()
    vocabulary = {symptom: index for index, symptom in enumerate(symptom_columns)}

    X = df[symptom_columns].to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[DISEASE_COLUMN].to_numpy())

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model(input_dim=X.shape[1], output_dim=len(label_encoder.classes_))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        verbose=1,
        callbacks=callbacks,
    )

    train_probabilities = model.predict(X_train, verbose=0)
    val_probabilities = model.predict(X_val, verbose=0)
    train_metrics = evaluate_predictions(y_train, np.argmax(train_probabilities, axis=1))
    val_metrics = evaluate_predictions(y_val, np.argmax(val_probabilities, axis=1))
    baseline_results = train_baselines(X_train, X_val, y_train, y_val)

    model.save(MODEL_DIR / "ml_model.h5", include_optimizer=False)
    with (MODEL_DIR / "label_encoder.pkl").open("wb") as f:
        pickle.dump(label_encoder, f)
    with (MODEL_DIR / "symptom_vocabulary.json").open("w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=2)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "total_diseases": int(len(label_encoder.classes_)),
        "total_symptoms": int(len(symptom_columns)),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "preprocessing": preprocessing,
        "tensorflow_model": {
            "training": train_metrics,
            "validation": val_metrics,
            "final_epoch": int(len(history.history["loss"])),
        },
        "baseline_models": baseline_results,
        "diseases": label_encoder.classes_.tolist(),
    }
    with (MODEL_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    comparison = {
        "tensorflow_neural_network": val_metrics,
        **baseline_results,
    }
    with (MODEL_DIR / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    save_validation_charts(df, comparison, MODEL_DIR)

    print("Training complete")
    print(f"Diseases: {len(label_encoder.classes_)}")
    print(f"Symptoms: {len(symptom_columns)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Validation F1-score: {val_metrics['f1_score']:.4f}")
    print(f"Artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
