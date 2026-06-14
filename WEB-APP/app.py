"""
Flask application for the ML-Powered Medical Diagnosis Assistant.

The app uses the trained TensorFlow model and label encoder stored in
BACKEND/models. It is intended for educational triage support only and must
not be used as a replacement for a qualified medical professional.
"""

from __future__ import annotations

import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "models"
LOG_DIR = WEB_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-medical-diagnosis-assistant")


COMMON_ALIASES = {
    "fever": ["high_fever", "mild_fever"],
    "high temperature": ["high_fever"],
    "low fever": ["mild_fever"],
    "cough": ["cough"],
    "dry cough": ["cough"],
    "persistent cough": ["cough"],
    "shortness of breath": ["breathlessness"],
    "difficulty breathing": ["breathlessness"],
    "breathing difficulty": ["breathlessness"],
    "chest pain": ["chest_pain"],
    "stomach pain": ["abdominal_pain", "belly_pain"],
    "belly pain": ["belly_pain", "abdominal_pain"],
    "body ache": ["muscle_pain", "joint_pain"],
    "body aches": ["muscle_pain", "joint_pain"],
    "yellow eyes": ["yellowing_of_eyes"],
    "yellow skin": ["yellowish_skin"],
    "diarrhea": ["diarrhoea"],
    "loose motion": ["diarrhoea"],
    "burning urine": ["burning_micturition"],
    "loss appetite": ["loss_of_appetite"],
    "loss of appetite": ["loss_of_appetite"],
    "runny nose": ["runny_nose", "continuous_sneezing"],
    "sore throat": ["throat_irritation"],
    "red eyes": ["redness_of_eyes"],
    "loss of smell": ["loss_of_smell"],
    "fast heart rate": ["fast_heart_rate"],
    "rapid heartbeat": ["fast_heart_rate", "palpitations"],
    "pain behind eyes": ["pain_behind_the_eyes"],
    "red spots": ["red_spots_over_body"],
    "blurred vision": ["blurred_and_distorted_vision"],
    "visual disturbance": ["visual_disturbances"],
    "visual disturbances": ["visual_disturbances"],
    "stiff neck": ["stiff_neck"],
    "skin rash": ["skin_rash"],
}


class DiagnosisEngine:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.label_encoder: Any | None = None
        self.vocabulary: dict[str, int] = {}
        self.status_message = "Model not loaded"
        self.load_artifacts()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.label_encoder is not None and bool(self.vocabulary)

    def load_artifacts(self) -> None:
        vocab_path = MODEL_DIR / "symptom_vocabulary.json"
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        model_path = MODEL_DIR / "ml_model.h5"

        if not vocab_path.exists():
            self.status_message = f"Missing vocabulary: {vocab_path}"
            return
        if not encoder_path.exists():
            self.status_message = f"Missing label encoder: {encoder_path}"
            return
        if not model_path.exists():
            self.status_message = f"Missing model: {model_path}"
            return
        if not TF_AVAILABLE:
            self.status_message = "TensorFlow is not installed"
            return

        with vocab_path.open("r", encoding="utf-8") as f:
            self.vocabulary = json.load(f)

        with encoder_path.open("rb") as f:
            self.label_encoder = pickle.load(f)

        self.model = tf.keras.models.load_model(model_path)
        disease_count = len(getattr(self.label_encoder, "classes_", []))
        self.status_message = f"Loaded {disease_count} diseases and {len(self.vocabulary)} symptoms"

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def contains_phrase(text: str, phrase: str) -> bool:
        return bool(re.search(rf"\b{re.escape(phrase)}\b", text))

    def match_symptoms(self, symptoms_text: str) -> list[str]:
        normalized = self.normalize(symptoms_text)
        matched: set[str] = set()

        for symptom in self.vocabulary:
            phrase = symptom.replace("_", " ")
            if self.contains_phrase(normalized, phrase):
                matched.add(symptom)

        for phrase, aliases in COMMON_ALIASES.items():
            if phrase == "fever" and (
                self.contains_phrase(normalized, "high fever")
                or self.contains_phrase(normalized, "mild fever")
            ):
                continue
            if self.contains_phrase(normalized, phrase):
                for alias in aliases:
                    if alias in self.vocabulary:
                        matched.add(alias)

        return sorted(matched, key=lambda item: self.vocabulary[item])

    def vectorize(self, matched_symptoms: list[str]) -> np.ndarray:
        vector = np.zeros((1, len(self.vocabulary)), dtype=np.float32)
        for symptom in matched_symptoms:
            vector[0, self.vocabulary[symptom]] = 1.0
        return vector

    def predict(self, symptoms_text: str) -> dict[str, Any]:
        matched_symptoms = self.match_symptoms(symptoms_text)
        matched_labels = [symptom.replace("_", " ").title() for symptom in matched_symptoms]

        if not matched_symptoms:
            return {
                "primary_diagnosis": "Insufficient symptom match",
                "confidence": 0.0,
                "top_predictions": [],
                "matched_symptoms": [],
                "urgency_level": "Needs Review",
                "suggested_action": (
                    "Enter clearer symptoms such as fever, cough, chest pain, headache, "
                    "vomiting, skin rash, or breathlessness. Consult a doctor if symptoms are serious."
                ),
                "evidence_count": 0,
                "model_status": self.status_message,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

        if not self.ready:
            return self.rule_based_fallback(matched_symptoms, matched_labels)

        vector = self.vectorize(matched_symptoms)
        probabilities = self.model.predict(vector, verbose=0)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]

        top_predictions = []
        for index in top_indices:
            disease = self.label_encoder.inverse_transform([index])[0]
            top_predictions.append(
                {
                    "disease": disease,
                    "confidence": round(float(probabilities[index]) * 100, 1),
                }
            )

        primary = top_predictions[0]
        urgency = determine_urgency(primary["disease"], primary["confidence"], matched_symptoms)

        return {
            "primary_diagnosis": primary["disease"],
            "confidence": primary["confidence"],
            "top_predictions": top_predictions,
            "matched_symptoms": matched_labels,
            "urgency_level": urgency,
            "suggested_action": generate_action(primary["disease"], primary["confidence"], urgency),
            "evidence_count": len(matched_symptoms),
            "model_status": self.status_message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def rule_based_fallback(self, matched_symptoms: list[str], matched_labels: list[str]) -> dict[str, Any]:
        symptom_set = set(matched_symptoms)
        diagnosis = "General symptom review needed"
        confidence = 45.0

        if {"chest_pain", "breathlessness"} & symptom_set:
            diagnosis = "Respiratory or cardiac warning symptoms"
            confidence = 65.0
        elif {"high_fever", "chills"} <= symptom_set:
            diagnosis = "Fever syndrome"
            confidence = 60.0
        elif {"headache", "nausea"} <= symptom_set:
            diagnosis = "Headache syndrome"
            confidence = 58.0

        urgency = determine_urgency(diagnosis, confidence, matched_symptoms)
        return {
            "primary_diagnosis": diagnosis,
            "confidence": confidence,
            "top_predictions": [{"disease": diagnosis, "confidence": confidence}],
            "matched_symptoms": matched_labels,
            "urgency_level": urgency,
            "suggested_action": generate_action(diagnosis, confidence, urgency),
            "evidence_count": len(matched_symptoms),
            "model_status": self.status_message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


def determine_urgency(diagnosis: str, confidence: float, matched_symptoms: list[str]) -> str:
    diagnosis_lower = diagnosis.lower()
    urgent_symptoms = {"chest_pain", "breathlessness", "palpitations", "coma", "unsteadiness"}

    if urgent_symptoms.intersection(matched_symptoms):
        return "Urgent"
    if any(term in diagnosis_lower for term in ["heart attack", "pneumonia", "paralysis", "tuberculosis"]):
        return "Urgent"
    if confidence < 50:
        return "Needs Review"
    if confidence < 70:
        return "Moderate"
    return "Routine"


def generate_action(diagnosis: str, confidence: float, urgency: str) -> str:
    if urgency == "Urgent":
        return "Seek medical attention as soon as possible, especially if symptoms are severe or worsening."
    if confidence < 50:
        return "The model is not confident. Add more specific symptoms and consult a healthcare professional."

    diagnosis_lower = diagnosis.lower()
    if "diabetes" in diagnosis_lower:
        return "Arrange a clinical check-up and blood glucose testing for confirmation."
    if "hypertension" in diagnosis_lower:
        return "Check blood pressure and consult a healthcare professional for confirmation."
    if any(term in diagnosis_lower for term in ["infection", "malaria", "dengue", "typhoid"]):
        return "Consult a doctor for examination and confirmatory tests."

    return "Monitor symptoms and consult a qualified healthcare professional for confirmation."


def write_log(entry: dict[str, Any]) -> None:
    log_path = LOG_DIR / "diagnosis_logs.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


engine = DiagnosisEngine()


@app.route("/")
def index() -> str:
    return render_template("Index.html")


@app.route("/api/health")
def health() -> Any:
    return jsonify(
        {
            "ready": engine.ready,
            "status": engine.status_message,
            "model_dir": str(MODEL_DIR),
        }
    )


@app.route("/diagnose", methods=["POST"])
def diagnose() -> Any:
    data = request.get_json(silent=True) or {}
    symptoms = str(data.get("symptoms", "")).strip()

    if not symptoms:
        return jsonify({"error": "Please enter your symptoms"}), 400

    result = engine.predict(symptoms)
    write_log(
        {
            "timestamp": result["timestamp"],
            "input": symptoms,
            "diagnosis": result["primary_diagnosis"],
            "confidence": result["confidence"],
            "matched_symptoms": result["matched_symptoms"],
        }
    )
    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback() -> Any:
    data = request.get_json(silent=True) or {}
    feedback_path = LOG_DIR / "feedback.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "diagnosis": data.get("diagnosis"),
        "rating": data.get("rating"),
        "feedback": data.get("feedback"),
    }
    with feedback_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return jsonify({"message": "Thank you for your feedback"})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
