"""Light project-focused dataset cleaning for the diagnosis app.

This intentionally cleans only the symptom-to-disease CSVs used by the Flask
project. It does not deep-clean every downloaded medical dataset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "DATASETS"
OUTPUT_DIR = DATA_DIR / "cleaned"
TRAIN_OUTPUT = OUTPUT_DIR / "Training_cleaned.csv"
TEST_OUTPUT = OUTPUT_DIR / "Testing_cleaned.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "cleaning_summary.json"

RANDOM_STATE = 42
TARGET_COLUMN = "prognosis"
PLACEHOLDER_SYMPTOMS = {"", "0", "0.0", "nan", "none", "null"}

PROJECT_SOURCES = [
    "Training.csv",
    "Testing.csv",
    "dataset.csv",
    "dataset (1).csv",
    "cleaned_dataset.csv",
]

CANONICAL_SYMPTOMS = {
    "dischromic_patches": "dyschromic_patches",
    "dischromic_patches": "dyschromic_patches",
    "foul_smell_of urine": "foul_smell_of_urine",
    "foul_smell_of_urine": "foul_smell_of_urine",
    "scurring": "scurrying",
    "toxic_look_typhus": "toxic_look_typhus",
}

CANONICAL_DISEASES = {
    "covid": "COVID-19",
    "covid19": "COVID-19",
    "covid_19": "COVID-19",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "dimorphic_hemorrhoids_piles": "Dimorphic Hemorrhoids(Piles)",
    "dimorphic_hemorrhoids_piles_": "Dimorphic Hemorrhoids(Piles)",
    "dimorphic_hemmorhoids_piles": "Dimorphic Hemorrhoids(Piles)",
    "dimorphic_hemmorhoids_piles_": "Dimorphic Hemorrhoids(Piles)",
    "osteoarthritis": "Osteoarthritis",
    "osteoarthristis": "Osteoarthritis",
    "paroxysmal_positional_vertigo": "Paroxysmal Positional Vertigo",
    "vertigo_paroymsal_positional_vertigo": "Paroxysmal Positional Vertigo",
    "vertigo_paroxysmal_positional_vertigo": "Paroxysmal Positional Vertigo",
    "peptic_ulcer_disease": "Peptic Ulcer Disease",
    "peptic_ulcer_diseae": "Peptic Ulcer Disease",
    "urinary_tract_infection": "Urinary Tract Infection",
}


def clean_disease(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("_", " ")
    text = text.strip()
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if key in CANONICAL_DISEASES:
        return CANONICAL_DISEASES[key]
    return text.title().replace("Aids", "AIDS").replace("Gerd", "GERD")


def clean_symptom(value: object) -> str:
    text = str(value).strip().lower()
    if text in PLACEHOLDER_SYMPTOMS:
        return ""
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_()]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text in PLACEHOLDER_SYMPTOMS:
        return ""
    return CANONICAL_SYMPTOMS.get(text, text)


def read_binary_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()

    rows = []
    symptom_columns = [column for column in df.columns if column != TARGET_COLUMN]
    for _, row in df.iterrows():
        symptoms = []
        for column in symptom_columns:
            try:
                present = int(row[column]) == 1
            except (TypeError, ValueError):
                present = False
            if present:
                symptom = clean_symptom(column)
                if symptom:
                    symptoms.append(symptom)
        if symptoms:
            rows.append({"prognosis": clean_disease(row[TARGET_COLUMN]), "symptoms": tuple(sorted(set(symptoms)))})
    return pd.DataFrame(rows)


def read_symptom_list_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    disease_column = "Disease"
    if disease_column not in df.columns:
        return pd.DataFrame()

    symptom_columns = [
        column
        for column in df.columns
        if re.fullmatch(r"Symptom_\d+", str(column)) and not str(column).lower().startswith("weight")
    ]
    rows = []
    for _, row in df.iterrows():
        symptoms = []
        for column in symptom_columns:
            value = row[column]
            if pd.isna(value):
                continue
            symptom = clean_symptom(value)
            if symptom:
                symptoms.append(symptom)
        if symptoms:
            rows.append({"prognosis": clean_disease(row[disease_column]), "symptoms": tuple(sorted(set(symptoms)))})
    return pd.DataFrame(rows)


def build_clean_matrix(records: pd.DataFrame) -> pd.DataFrame:
    records = records.drop_duplicates(["prognosis", "symptoms"]).reset_index(drop=True)
    disease_counts = records["prognosis"].value_counts()
    usable_diseases = disease_counts[disease_counts >= 2].index
    records = records[records["prognosis"].isin(usable_diseases)].reset_index(drop=True)

    symptom_columns = sorted({symptom for symptoms in records["symptoms"] for symptom in symptoms})
    matrix = pd.DataFrame(0, index=records.index, columns=symptom_columns, dtype="int8")
    for index, symptoms in records["symptoms"].items():
        matrix.loc[index, list(symptoms)] = 1
    matrix[TARGET_COLUMN] = records["prognosis"]
    return matrix


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    source_rows = {}
    for source in PROJECT_SOURCES:
        path = DATA_DIR / source
        if not path.exists():
            source_rows[source] = 0
            continue

        if source in {"Training.csv", "Testing.csv"}:
            part = read_binary_source(path)
        else:
            part = read_symptom_list_source(path)

        source_rows[source] = len(part)
        if not part.empty:
            parts.append(part)

    if not parts:
        raise RuntimeError("No project symptom datasets were found.")

    records = pd.concat(parts, ignore_index=True)
    matrix = build_clean_matrix(records)

    train_df, test_df = train_test_split(
        matrix,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=matrix[TARGET_COLUMN],
    )

    feature_columns = [column for column in matrix.columns if column != TARGET_COLUMN]
    train_df = train_df[feature_columns + [TARGET_COLUMN]].sort_values(TARGET_COLUMN).reset_index(drop=True)
    test_df = test_df[feature_columns + [TARGET_COLUMN]].sort_values(TARGET_COLUMN).reset_index(drop=True)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    summary = {
        "scope": "project-focused light cleaning only",
        "sources_used": PROJECT_SOURCES,
        "source_rows_loaded": source_rows,
        "clean_records_before_dedup": int(len(records)),
        "training_rows": int(len(train_df)),
        "testing_rows": int(len(test_df)),
        "disease_classes": int(matrix[TARGET_COLUMN].nunique()),
        "symptom_features": int(len(feature_columns)),
        "outputs": {
            "training": str(TRAIN_OUTPUT.relative_to(PROJECT_ROOT)),
            "testing": str(TEST_OUTPUT.relative_to(PROJECT_ROOT)),
        },
    }
    with SUMMARY_OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Light project dataset cleaning complete")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Diseases: {matrix[TARGET_COLUMN].nunique()}")
    print(f"Symptoms: {len(feature_columns)}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
