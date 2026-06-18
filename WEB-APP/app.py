"""
Flask application for the ML-Powered Medical Diagnosis Assistant.

The app uses the trained TensorFlow model and label encoder stored in
BACKEND/models. It is intended for educational triage support only and must
not be used as a replacement for a qualified medical professional.
"""

from __future__ import annotations

import base64
import csv
import json
import os
import pickle
import re
import secrets
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    AES_AVAILABLE = True
except ImportError:
    AESGCM = None
    AES_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "BACKEND" / "models"
EHR_INDEX_PATH = MODEL_DIR / "ehr_validation_index.json"
FRACTURE_DATASET_PATH = PROJECT_ROOT / "DATASETS" / "bone_fracture_dataset.csv"
LOG_DIR = WEB_ROOT / "logs"
INSTANCE_DIR = WEB_ROOT / "instance"
LOG_KEY_PATH = INSTANCE_DIR / "log_aes256.key"
LOG_DIR.mkdir(parents=True, exist_ok=True)
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

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

FRACTURE_TRIGGER_TERMS = {
    "fracture",
    "broken bone",
    "bone injury",
    "fall",
    "fell",
    "injury",
    "swelling",
    "deformity",
    "visible bone",
    "unable to bear weight",
    "inability to bear weight",
    "crepitus",
    "grating sensation",
}

FRACTURE_STRONG_TERMS = {
    "visible bone",
    "open fracture",
    "inability to bear weight",
    "unable to bear weight",
    "obvious deformity",
    "visible deformity",
    "grating sensation crepitus",
    "severe pain in hip or groin",
    "sharp pain when breathing deeply",
    "pain when coughing or laughing",
    "inability to lift arm",
    "inability to move the affected part",
}


class DiagnosisEngine:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.label_encoder: Any | None = None
        self.vocabulary: dict[str, int] = {}
        self.ehr_index: dict[str, Any] = {}
        self.fracture_index: dict[str, Any] = {}
        self.ehr_status_message = "EHR validation index not loaded"
        self.fracture_status_message = "Fracture screening dataset not loaded"
        self.status_message = "Model not loaded"
        self.load_artifacts()
        self.load_ehr_index()
        self.load_fracture_index()

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

    def load_ehr_index(self) -> None:
        if not EHR_INDEX_PATH.exists():
            self.ehr_status_message = f"Missing EHR validation index: {EHR_INDEX_PATH}"
            return

        with EHR_INDEX_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.ehr_index = payload.get("diseases", {})
        disease_count = len(self.ehr_index)
        engine = payload.get("spark_profile", {}).get("engine", "aggregate index")
        self.ehr_status_message = f"Loaded aggregate validation index for {disease_count} diseases via {engine}"

    def load_fracture_index(self) -> None:
        if not FRACTURE_DATASET_PATH.exists():
            self.fracture_status_message = f"Missing fracture dataset: {FRACTURE_DATASET_PATH}"
            return

        index: dict[str, Any] = {}
        with FRACTURE_DATASET_PATH.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                diagnosis = str(row.get("Diagnosis", "")).strip()
                if not diagnosis:
                    continue

                record = index.setdefault(
                    diagnosis,
                    {
                        "case_count": 0,
                        "body_parts": Counter(),
                        "bones": Counter(),
                        "symptoms": Counter(),
                    },
                )
                record["case_count"] += 1

                body_part = self.normalize(str(row.get("BodyPart", "")))
                bone = self.normalize(str(row.get("SpecificBone", "")))
                if body_part:
                    record["body_parts"][body_part] += 1
                if bone:
                    record["bones"][bone] += 1

                symptoms = str(row.get("Symptoms", ""))
                for symptom in re.split(r",|;", symptoms):
                    normalized_symptom = self.normalize(symptom)
                    if len(normalized_symptom) >= 4:
                        record["symptoms"][normalized_symptom] += 1

        self.fracture_index = index
        self.fracture_status_message = f"Loaded fracture screening index for {len(index)} fracture types"

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def contains_phrase(text: str, phrase: str) -> bool:
        return bool(re.search(rf"\b{re.escape(phrase)}\b", text))

    def fracture_term_matches(self, normalized_text: str, term: str, trigger_matches: list[str]) -> bool:
        if self.contains_phrase(normalized_text, term):
            return True

        return any(
            len(trigger) >= 4 and self.contains_phrase(term, trigger)
            for trigger in trigger_matches
        )

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

    def fracture_screen(self, symptoms_text: str) -> dict[str, Any] | None:
        if not self.fracture_index:
            return None

        normalized = self.normalize(symptoms_text)
        trigger_matches = [
            self.normalize(term)
            for term in FRACTURE_TRIGGER_TERMS
            if self.contains_phrase(normalized, self.normalize(term))
        ]
        if not trigger_matches:
            return None

        explicit_triggers = {
            "fracture",
            "broken bone",
            "bone injury",
            "visible bone",
            "deformity",
            "unable to bear weight",
            "inability to bear weight",
            "crepitus",
            "grating sensation",
        }
        has_explicit_trigger = bool(set(trigger_matches).intersection(explicit_triggers))
        strong_terms = {self.normalize(term) for term in FRACTURE_STRONG_TERMS}

        candidates = []
        for diagnosis, record in self.fracture_index.items():
            matched_terms: dict[str, int] = {}
            score = 0

            for term, support_count in record["symptoms"].items():
                if self.fracture_term_matches(normalized, term, trigger_matches):
                    matched_terms[term] = int(support_count)
                    score += 3 if term in strong_terms else 1

            for term, support_count in record["body_parts"].items():
                if self.contains_phrase(normalized, term):
                    matched_terms[term] = int(support_count)
                    score += 1

            for term, support_count in record["bones"].items():
                if self.contains_phrase(normalized, term):
                    matched_terms[term] = int(support_count)
                    score += 1

            if score >= 2 or (has_explicit_trigger and score >= 1):
                confidence = min(92.0, 50.0 + (score * 7.0) + min(record["case_count"], 80) * 0.05)
                candidates.append(
                    {
                        "diagnosis": diagnosis,
                        "confidence": round(confidence, 1),
                        "score": score,
                        "case_count": int(record["case_count"]),
                        "matched_terms": matched_terms,
                        "record": record,
                    }
                )

        if not candidates:
            if not has_explicit_trigger:
                return None
            return self.fracture_fallback(trigger_matches)

        candidates.sort(key=lambda item: (item["score"], item["confidence"], item["case_count"]), reverse=True)
        primary = candidates[0]
        urgent_terms = {"visible bone", "open fracture", "visible deformity", "obvious deformity"}
        urgency = "Urgent" if set(primary["matched_terms"]).intersection(urgent_terms) else "Moderate"
        record = primary["record"]

        supporting_symptoms = [
            {
                "symptom": term.title(),
                "support_count": int(count),
            }
            for term, count in sorted(primary["matched_terms"].items(), key=lambda item: item[1], reverse=True)
        ]
        top_reference_symptoms = [
            {"symptom": term.title(), "support_count": int(count)}
            for term, count in record["symptoms"].most_common(6)
        ]

        top_predictions = [
            {"disease": item["diagnosis"], "confidence": item["confidence"]}
            for item in candidates[:3]
        ]

        return {
            "primary_diagnosis": primary["diagnosis"],
            "confidence": primary["confidence"],
            "top_predictions": top_predictions,
            "matched_symptoms": [item["symptom"] for item in supporting_symptoms],
            "urgency_level": urgency,
            "suggested_action": (
                "Immobilize the injured area and get an X-ray or medical evaluation promptly. "
                "Seek emergency care for visible bone, deformity, numbness, severe swelling, or uncontrolled pain."
            ),
            "evidence_count": len(supporting_symptoms),
            "ehr_evidence": {
                "status": self.fracture_status_message,
                "matched_cases": primary["case_count"],
                "source_counts": {"bone_fracture_dataset": primary["case_count"]},
                "supporting_symptoms": supporting_symptoms,
                "top_reference_symptoms": top_reference_symptoms,
                "description": (
                    "Fracture screening matched injury symptoms against the cleaned fracture dataset. "
                    "This is educational triage support, not imaging confirmation."
                ),
                "precautions": ["immobilize the area", "avoid weight bearing", "seek medical review"],
            },
            "model_status": f"{self.status_message}; {self.fracture_status_message}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def fracture_fallback(self, trigger_matches: list[str]) -> dict[str, Any]:
        matched = [term.title() for term in trigger_matches]
        return {
            "primary_diagnosis": "Possible fracture or injury",
            "confidence": 55.0,
            "top_predictions": [{"disease": "Possible fracture or injury", "confidence": 55.0}],
            "matched_symptoms": matched,
            "urgency_level": "Needs Review",
            "suggested_action": (
                "Describe the injured body part and symptoms such as swelling, deformity, visible bone, "
                "tenderness, or inability to bear weight. Get medical review if pain is severe."
            ),
            "evidence_count": len(matched),
            "ehr_evidence": {
                "status": self.fracture_status_message,
                "matched_cases": 0,
                "source_counts": {"bone_fracture_dataset": 0},
                "supporting_symptoms": [
                    {"symptom": symptom, "support_count": 0}
                    for symptom in matched
                ],
                "top_reference_symptoms": [],
                "description": "Injury terms were detected, but the app needs more specific fracture symptoms.",
                "precautions": ["avoid weight bearing", "seek medical review"],
            },
            "model_status": f"{self.status_message}; {self.fracture_status_message}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def vectorize(self, matched_symptoms: list[str]) -> np.ndarray:
        vector = np.zeros((1, len(self.vocabulary)), dtype=np.float32)
        for symptom in matched_symptoms:
            vector[0, self.vocabulary[symptom]] = 1.0
        return vector

    def evidence_for(self, disease: str, matched_symptoms: list[str]) -> dict[str, Any]:
        record = self.ehr_index.get(disease, {})
        support = record.get("symptom_support", {})
        supporting_symptoms = []
        for symptom in matched_symptoms:
            supporting_symptoms.append(
                {
                    "symptom": symptom.replace("_", " ").title(),
                    "support_count": int(support.get(symptom, 0)),
                }
            )

        return {
            "status": self.ehr_status_message,
            "matched_cases": int(record.get("case_count", 0)),
            "source_counts": record.get("source_counts", {}),
            "supporting_symptoms": supporting_symptoms,
            "top_reference_symptoms": record.get("top_symptoms", [])[:6],
            "description": record.get("description", ""),
            "precautions": record.get("precautions", [])[:4],
        }

    def covid_screen(
        self,
        matched_symptoms: list[str],
        matched_labels: list[str],
        top_predictions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        symptom_set = set(matched_symptoms)
        hallmark_symptoms = {"loss_of_smell", "loss_of_taste"}
        if not {"high_fever", "cough"}.issubset(symptom_set):
            return None
        if not hallmark_symptoms.intersection(symptom_set):
            return None

        model_covid_confidence = next(
            (prediction["confidence"] for prediction in top_predictions if prediction["disease"] == "Covid"),
            0.0,
        )
        confidence = round(max(float(model_covid_confidence), 72.0), 1)
        ehr_evidence = self.evidence_for("Covid", matched_symptoms)
        ranked = [{"disease": "Covid", "confidence": confidence}]
        ranked.extend(prediction for prediction in top_predictions if prediction["disease"] != "Covid")

        urgency = "Urgent" if "breathlessness" in symptom_set or "chest_pain" in symptom_set else "Moderate"
        return {
            "primary_diagnosis": "Covid",
            "confidence": confidence,
            "top_predictions": ranked[:3],
            "matched_symptoms": matched_labels,
            "urgency_level": urgency,
            "suggested_action": (
                "Consider COVID testing and reduce close contact until reviewed. "
                "Seek urgent care if breathing difficulty, chest pain, confusion, or worsening fever occurs."
            ),
            "evidence_count": len(matched_symptoms),
            "ehr_evidence": ehr_evidence,
            "model_status": f"{self.status_message}; COVID screening rule active",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def predict(self, symptoms_text: str) -> dict[str, Any]:
        fracture_result = self.fracture_screen(symptoms_text)
        if fracture_result:
            return fracture_result

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
                "ehr_evidence": self.evidence_for("", []),
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

        covid_result = self.covid_screen(matched_symptoms, matched_labels, top_predictions)
        if covid_result:
            return covid_result

        primary = top_predictions[0]
        urgency = determine_urgency(primary["disease"], primary["confidence"], matched_symptoms)
        ehr_evidence = self.evidence_for(primary["disease"], matched_symptoms)

        return {
            "primary_diagnosis": primary["disease"],
            "confidence": primary["confidence"],
            "top_predictions": top_predictions,
            "matched_symptoms": matched_labels,
            "urgency_level": urgency,
            "suggested_action": generate_action(
                primary["disease"],
                primary["confidence"],
                urgency,
                ehr_evidence.get("precautions", []),
            ),
            "evidence_count": len(matched_symptoms),
            "ehr_evidence": ehr_evidence,
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
        ehr_evidence = self.evidence_for(diagnosis, matched_symptoms)
        return {
            "primary_diagnosis": diagnosis,
            "confidence": confidence,
            "top_predictions": [{"disease": diagnosis, "confidence": confidence}],
            "matched_symptoms": matched_labels,
            "urgency_level": urgency,
            "suggested_action": generate_action(diagnosis, confidence, urgency, ehr_evidence.get("precautions", [])),
            "evidence_count": len(matched_symptoms),
            "ehr_evidence": ehr_evidence,
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


def generate_action(diagnosis: str, confidence: float, urgency: str, precautions: list[str] | None = None) -> str:
    precaution_text = ""
    if precautions:
        precaution_text = " Suggested precautions: " + ", ".join(precautions[:2]) + "."

    if urgency == "Urgent":
        return (
            "Seek medical attention as soon as possible, especially if symptoms are severe or worsening."
            + precaution_text
        )
    if confidence < 50:
        return (
            "The model is not confident. Add more specific symptoms and consult a healthcare professional."
            + precaution_text
        )

    diagnosis_lower = diagnosis.lower()
    if "diabetes" in diagnosis_lower:
        return "Arrange a clinical check-up and blood glucose testing for confirmation." + precaution_text
    if "hypertension" in diagnosis_lower:
        return "Check blood pressure and consult a healthcare professional for confirmation." + precaution_text
    if any(term in diagnosis_lower for term in ["infection", "malaria", "dengue", "typhoid"]):
        return "Consult a doctor for examination and confirmatory tests." + precaution_text

    return "Monitor symptoms and consult a qualified healthcare professional for confirmation." + precaution_text


def load_aes_key() -> bytes | None:
    if not AES_AVAILABLE:
        return None

    if LOG_KEY_PATH.exists():
        key = LOG_KEY_PATH.read_bytes()
        if len(key) == 32:
            return key

    key = secrets.token_bytes(32)
    LOG_KEY_PATH.write_bytes(key)
    return key


LOG_KEY = load_aes_key()
SECURE_LOGGING_ENABLED = AES_AVAILABLE and LOG_KEY is not None


def encrypt_entry(entry: dict[str, Any]) -> dict[str, str | int]:
    if AESGCM is None or LOG_KEY is None:
        raise RuntimeError("AES-256-GCM logging is not available")

    nonce = secrets.token_bytes(12)
    plaintext = json.dumps(entry, ensure_ascii=True).encode("utf-8")
    ciphertext = AESGCM(LOG_KEY).encrypt(nonce, plaintext, None)
    return {
        "version": 1,
        "algorithm": "AES-256-GCM",
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def write_event_log(filename: str, entry: dict[str, Any]) -> None:
    if SECURE_LOGGING_ENABLED:
        log_path = LOG_DIR / f"{filename}.jsonl.enc"
        payload: dict[str, Any] = encrypt_entry(entry)
    else:
        log_path = LOG_DIR / f"{filename}.jsonl"
        payload = entry

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_log(entry: dict[str, Any]) -> None:
    write_event_log("diagnosis_logs", entry)


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
            "ehr_ready": bool(engine.ehr_index),
            "ehr_status": engine.ehr_status_message,
            "fracture_ready": bool(engine.fracture_index),
            "fracture_status": engine.fracture_status_message,
            "secure_logging": SECURE_LOGGING_ENABLED,
            "log_security": "AES-256-GCM local event logging" if SECURE_LOGGING_ENABLED else "plain local fallback",
            "offline_ready": True,
            "network_access": "local, LAN, or ngrok tunnel",
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
            "ehr_matched_cases": result.get("ehr_evidence", {}).get("matched_cases", 0),
        }
    )
    return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback() -> Any:
    data = request.get_json(silent=True) or {}
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "diagnosis": data.get("diagnosis"),
        "rating": data.get("rating"),
        "feedback": data.get("feedback"),
    }
    write_event_log("feedback", entry)
    return jsonify({"message": "Thank you for your feedback"})


if __name__ == "__main__":
    host = os.environ.get("MEDAI_HOST", "0.0.0.0")
    port = int(os.environ.get("MEDAI_PORT", "5000"))
    app.run(debug=False, host=host, port=port)
