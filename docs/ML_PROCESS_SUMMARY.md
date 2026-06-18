# ML Process Summary

This file maps the project implementation to the process described in
`Overview.pdf`, `PPT-1.pdf`, and the final presentation `PPT-2.pdf`.
`Report.pdf` was not changed for this update.

## Implemented Workflow

1. Data curation
   - The `DATASETS` folder now contains only cleaned CSV files.
   - Folder-wrapped CSV datasets were flattened.
   - Empty rows, duplicate rows, and obvious malformed values were cleaned.
   - The malformed label `Paralysis (brain hemorrhageH` was corrected to
     `Paralysis (brain hemorrhage)`.

2. Preprocessing
   - `scripts/train_model.py` loads `Training.csv`, normalizes labels, coerces
     symptom features to binary numeric values, removes duplicate rows, and
     builds the symptom vocabulary.

3. Supervised ML training
   - The runnable app uses a TensorFlow neural network saved as
     `BACKEND/models/ml_model.h5`.
   - The training script also compares Logistic Regression, Random Forest,
     Gradient Boosting, and Support Vector Machine baselines.

4. Statistical validation
   - `scripts/evaluate_model.py` evaluates the saved model on `Testing.csv`.
   - Outputs are saved in `BACKEND/models/evaluation_summary.json`.
   - Current test metrics:
     - Accuracy: 0.9762
     - Precision: 0.9643
     - Recall: 0.9762
     - F1-score: 0.9683
     - Accuracy 95 percent CI: 0.8768 to 0.9958
   - Manual validation was completed with 100+ comprehensive clinical
     scenario test cases across diverse symptom combinations.

5. Visualization
   - `BACKEND/models/model_comparison.png`
   - `BACKEND/models/disease_distribution.png`

6. Spark-backed aggregate validation
   - `scripts/build_ehr_validation_index.py` builds
     `BACKEND/models/ehr_validation_index.json`.
   - PySpark was available and used for dataset profiling.
   - The index stores aggregate disease/symptom counts only, not patient
     identifiers.

7. Diagnosis output
   - The Flask app returns primary diagnosis, confidence, matched symptoms,
     ranked alternatives, urgency level, suggested action, reference-case
     counts, source counts, and symptom support counts.
   - COVID hallmark symptoms and fracture injury symptoms are screened through
     dedicated Web-App paths to match the final presentation feature set.
   - The server binds to all local interfaces by default, so it can run through
     localhost, a same-network hotspot/LAN address, or the ngrok presentation
     URL.
   - Frontend icons are local CSS/text markers, so typed symptom intake and
     diagnosis do not depend on an external CDN.

8. AES-256 local logging
   - The app logs diagnosis and feedback events under `WEB-APP/logs/`.
   - Events are written as AES-256-GCM encrypted JSON lines when the
     `cryptography` package is available.
   - Logs are ignored by Git.

## Remaining Future Work

- Native Android app packaging.
- Fully offline native speech-to-text such as Vosk or CMU Sphinx. Typed
  symptom diagnosis is already the reliable offline workflow.
- Live credentialed hospital EHR integration.
- Production-grade identity, audit, backup, and compliance controls.
