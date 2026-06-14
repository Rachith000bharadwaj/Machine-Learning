"""Train the symptom-to-disease TensorFlow model."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "DATASETS" / "Training.csv"
CLEANED_DATASET_PATH = PROJECT_ROOT / "DATASETS" / "cleaned" / "Training_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "models"
DISEASE_COLUMN = "prognosis"
RANDOM_STATE = 42


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


def main() -> None:
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = CLEANED_DATASET_PATH if CLEANED_DATASET_PATH.exists() else DEFAULT_DATASET_PATH
    print(f"Using dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if DISEASE_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{DISEASE_COLUMN}' in {dataset_path}")

    class_counts = df[DISEASE_COLUMN].value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if len(rare_classes):
        df = df[~df[DISEASE_COLUMN].isin(rare_classes)].copy()
        print(f"Removed {len(rare_classes)} class with fewer than 2 samples.")

    symptom_columns = [column for column in df.columns if column != DISEASE_COLUMN]
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
            patience=10,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=16,
        verbose=1,
        callbacks=callbacks,
    )

    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

    model.save(MODEL_DIR / "ml_model.h5")
    with (MODEL_DIR / "label_encoder.pkl").open("wb") as f:
        pickle.dump(label_encoder, f)
    with (MODEL_DIR / "symptom_vocabulary.json").open("w", encoding="utf-8") as f:
        json.dump(vocabulary, f, indent=2)

    summary = {
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "total_diseases": int(len(label_encoder.classes_)),
        "total_symptoms": int(len(symptom_columns)),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "training_accuracy": float(train_accuracy),
        "validation_accuracy": float(val_accuracy),
        "final_epoch": int(len(history.history["loss"])),
        "diseases": label_encoder.classes_.tolist(),
    }
    with (MODEL_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Artifacts saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
