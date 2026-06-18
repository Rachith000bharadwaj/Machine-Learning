"""Build an aggregate validation index from available clinical-style datasets.

The overview describes Spark-backed EHR validation. The public datasets in this
project are not real private EHR records, so this script builds an aggregate,
privacy-preserving evidence index from the cleaned CSVs. It uses PySpark for
source profiling when available and falls back gracefully for the app runtime.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "DATASETS"
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "models"
OUTPUT_PATH = MODEL_DIR / "ehr_validation_index.json"
DISEASE_COLUMN = "prognosis"

LABEL_FIXES = {
    "Paralysis (brain hemorrhageH": "Paralysis (brain hemorrhage)",
}

PROFILE_SYMPTOM_MAP = {
    "Fever": ["high_fever", "mild_fever"],
    "Cough": ["cough"],
    "Fatigue": ["fatigue"],
    "Difficulty Breathing": ["breathlessness"],
}


def clean_label(value: object) -> str:
    label = str(value).strip()
    return LABEL_FIXES.get(label, label)


def normalize_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def symptom_key(value: object) -> str:
    text = normalize_name(value)
    return text.replace(" ", "_")


def is_positive(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "positive", "present"}


def load_training_shape() -> tuple[list[str], list[str]]:
    with (DATASET_DIR / "Training.csv").open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or DISEASE_COLUMN not in reader.fieldnames:
            raise ValueError("Training.csv must include a prognosis column.")
        symptoms = [field for field in reader.fieldnames if field != DISEASE_COLUMN]
        diseases = sorted({clean_label(row[DISEASE_COLUMN]) for row in reader if row.get(DISEASE_COLUMN)})
    return diseases, symptoms


def empty_record() -> dict[str, object]:
    return {
        "case_count": 0,
        "source_counts": defaultdict(int),
        "symptom_support": defaultdict(int),
        "description": "",
        "precautions": [],
    }


def add_case(
    disease: str,
    source: str,
    positive_symptoms: Iterable[str],
    index: dict[str, dict[str, object]],
    disease_lookup: dict[str, str],
    model_symptoms: set[str],
) -> None:
    canonical = disease_lookup.get(normalize_name(disease))
    if not canonical:
        return

    record = index[canonical]
    record["case_count"] += 1
    record["source_counts"][source] += 1
    symptom_counter = record["symptom_support"]
    for symptom in positive_symptoms:
        key = symptom_key(symptom)
        if key in model_symptoms:
            symptom_counter[key] += 1


def add_binary_symptom_dataset(
    path: Path,
    source: str,
    disease_column: str,
    index: dict[str, dict[str, object]],
    disease_lookup: dict[str, str],
    model_symptoms: set[str],
) -> int:
    count = 0
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or disease_column not in reader.fieldnames:
            return 0
        symptom_columns = [field for field in reader.fieldnames if field != disease_column]
        for row in reader:
            disease = clean_label(row.get(disease_column, ""))
            positives = [column for column in symptom_columns if is_positive(row.get(column, ""))]
            add_case(disease, source, positives, index, disease_lookup, model_symptoms)
            count += 1
    return count


def add_patient_profile_dataset(
    path: Path,
    index: dict[str, dict[str, object]],
    disease_lookup: dict[str, str],
    model_symptoms: set[str],
) -> int:
    count = 0
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positives: list[str] = []
            for source_column, mapped_symptoms in PROFILE_SYMPTOM_MAP.items():
                if is_positive(row.get(source_column, "")):
                    positives.extend(mapped_symptoms)
            add_case(row.get("Disease", ""), "patient_profile", positives, index, disease_lookup, model_symptoms)
            count += 1
    return count


def add_syditriage_dataset(
    path: Path,
    index: dict[str, dict[str, object]],
    disease_lookup: dict[str, str],
    model_symptoms: set[str],
) -> int:
    count = 0
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            add_case(row.get("disease", ""), "syditriage", [row.get("symptom", "")], index, disease_lookup, model_symptoms)
            count += 1
    return count


def add_description_metadata(index: dict[str, dict[str, object]], disease_lookup: dict[str, str]) -> None:
    description_path = DATASET_DIR / "disease_description.csv"
    if description_path.exists():
        with description_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                canonical = disease_lookup.get(normalize_name(row.get("Disease", "")))
                if canonical:
                    index[canonical]["description"] = str(row.get("Symptom_Description", "")).strip()

    precaution_path = DATASET_DIR / "disease_precaution.csv"
    if precaution_path.exists():
        with precaution_path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                canonical = disease_lookup.get(normalize_name(row.get("Disease", "")))
                if canonical:
                    precautions = [
                        str(value).strip()
                        for key, value in row.items()
                        if key != "Disease" and str(value).strip()
                    ]
                    index[canonical]["precautions"] = precautions


def spark_source_profile(paths: list[Path]) -> dict[str, object]:
    profile: dict[str, object] = {
        "engine": "pandas_csv_fallback",
        "available": False,
        "sources": {},
    }
    try:
        from pyspark.sql import SparkSession
    except Exception as exc:
        profile["error"] = str(exc)
        return profile

    spark = None
    try:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("medical-diagnosis-validation-index")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        profile["engine"] = "pyspark"
        profile["available"] = True
        for path in paths:
            if not path.exists():
                continue
            frame = spark.read.option("header", True).csv(str(path))
            profile["sources"][path.name] = {
                "rows": int(frame.count()),
                "columns": int(len(frame.columns)),
            }
    except Exception as exc:
        profile["error"] = str(exc)
    finally:
        if spark is not None:
            spark.stop()
    return profile


def freeze_record(record: dict[str, object]) -> dict[str, object]:
    symptom_support = dict(record["symptom_support"])
    top_symptoms = sorted(symptom_support.items(), key=lambda item: item[1], reverse=True)[:12]
    return {
        "case_count": int(record["case_count"]),
        "source_counts": dict(sorted(dict(record["source_counts"]).items())),
        "symptom_support": dict(sorted(symptom_support.items())),
        "top_symptoms": [{"symptom": symptom, "count": int(count)} for symptom, count in top_symptoms],
        "description": record["description"],
        "precautions": record["precautions"],
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    diseases, symptoms = load_training_shape()
    model_symptoms = set(symptoms)
    disease_lookup = {normalize_name(disease): disease for disease in diseases}
    index = {disease: empty_record() for disease in diseases}

    source_rows: dict[str, int] = {}
    source_rows["Training.csv"] = add_binary_symptom_dataset(
        DATASET_DIR / "Training.csv",
        "training",
        DISEASE_COLUMN,
        index,
        disease_lookup,
        model_symptoms,
    )
    source_rows["Testing.csv"] = add_binary_symptom_dataset(
        DATASET_DIR / "Testing.csv",
        "testing",
        DISEASE_COLUMN,
        index,
        disease_lookup,
        model_symptoms,
    )

    profile_path = DATASET_DIR / "Disease_symptom_and_patient_profile_dataset.csv"
    if profile_path.exists():
        source_rows[profile_path.name] = add_patient_profile_dataset(profile_path, index, disease_lookup, model_symptoms)

    augmented_path = DATASET_DIR / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    if augmented_path.exists():
        source_rows[augmented_path.name] = add_binary_symptom_dataset(
            augmented_path,
            "augmented_symptom_reference",
            "diseases",
            index,
            disease_lookup,
            model_symptoms,
        )

    syditriage_path = DATASET_DIR / "syditriage.csv"
    if syditriage_path.exists():
        source_rows[syditriage_path.name] = add_syditriage_dataset(
            syditriage_path,
            index,
            disease_lookup,
            model_symptoms,
        )

    add_description_metadata(index, disease_lookup)
    spark_profile = spark_source_profile(
        [
            DATASET_DIR / "Training.csv",
            DATASET_DIR / "Testing.csv",
            profile_path,
            augmented_path,
            syditriage_path,
        ]
    )

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Aggregate EHR-style validation evidence for diagnosis outputs.",
        "privacy": "Stores counts and symptom aggregates only; no patient identifiers.",
        "spark_profile": spark_profile,
        "source_rows_processed": source_rows,
        "total_diseases": len(diseases),
        "total_symptoms": len(symptoms),
        "diseases": {disease: freeze_record(record) for disease, record in sorted(index.items())},
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"EHR validation index saved to {OUTPUT_PATH}")
    print(f"Diseases indexed: {len(diseases)}")
    print(f"Spark engine: {spark_profile.get('engine')}")
    print(f"Spark available: {spark_profile.get('available')}")


if __name__ == "__main__":
    main()
