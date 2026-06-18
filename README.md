# ML-Powered Medical Diagnosis Assistant

An educational machine learning project that predicts a preliminary diagnosis from patient-reported symptoms. `PPT-2.pdf` is the final presentation reference for the current project version. The runnable system includes a Flask web app, a trained TensorFlow classifier, confidence scores, matched symptom evidence, urgency labels, recommended actions, Spark-backed aggregate validation, AES-256 encrypted local event logs, and printable reports.

This project is not a medical device and must not be used as a replacement for qualified healthcare advice.

## Current Implementation

- Symptom-based disease prediction using a saved TensorFlow model.
- 134 symptom features and 42 disease classes from the training artifacts.
- Flask web app with text input, browser voice input support, ranked predictions, urgency level, EHR-style evidence, and printable diagnosis report.
- Village-point mode that runs on the local laptop or a same-network hotspot without internet-only frontend assets.
- Spark-backed aggregate validation index built from cleaned public clinical-style datasets.
- COVID screening support for high-fever/cough cases with loss of smell or taste.
- Fracture screening support backed by the cleaned fracture dataset.
- AES-256-GCM encrypted local diagnosis and feedback event logs.
- Repeatable evaluation script with accuracy, precision, recall, F1-score, confidence interval, confusion matrix, and mismatch logging.
- Manual validation with 100+ comprehensive clinical scenario test cases.
- Training script to rebuild the model from `DATASETS/Training.csv`, compare Logistic Regression, Random Forest, Gradient Boosting, SVM, and neural-network results, and save validation charts.
- Clean GitHub structure with large raw datasets and local logs ignored.

## What Is Planned or Experimental

Older project material discusses Android deployment and offline native speech recognition. Those remain useful future directions. The current final-PPT implementation focuses on the Flask Web-App, browser voice input, Spark-backed aggregate validation, confidence scoring, encrypted local logging, and visual validation artifacts.

Keeping this distinction clear makes the project stronger and more credible.

## Project Structure

```text
ML Project/
  BACKEND/
    models/
      ml_model.h5
      label_encoder.pkl
      symptom_vocabulary.json
      training_summary.json
      evaluation_summary.json
      ehr_validation_index.json
      model_comparison.json
      model_comparison.png
      disease_distribution.png
    *.ipynb
  DATASETS/
    Training.csv
    Testing.csv
    symptom_Description.csv
    symptom_precaution.csv
  WEB-APP/
    app.py
    templates/Index.html
    static/css/style.css
    static/js/app.js
  scripts/
    train_model.py
    evaluate_model.py
    build_ehr_validation_index.py
  docs/
    ML_PROCESS_SUMMARY.md
    PPT2_ALIGNMENT.md
    PROJECT_AUDIT.md
    GITHUB_UPLOAD_STEPS.md
  Overview.pdf
  Report.pdf
  PPT-1.pdf
  PPT-2.pdf
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Web App

```bash
cd WEB-APP
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The server now binds to `0.0.0.0` by default, so it can also be opened from another phone or laptop on the same hotspot or local network by using the host machine's IP address:

```text
http://<laptop-ip-address>:5000
```

For the village-point offline workflow, run:

```bat
run_village_dashboard.bat
```

This uses the local TensorFlow model, local datasets, local CSS/JS, and local encrypted logs. Internet is not required for typed symptom intake and diagnosis after the Python dependencies are installed. Browser voice input depends on browser support and may not be available offline, so typed symptom entry is the reliable offline path.

## Run on the Public ngrok URL

The presentation website URL is:

```text
https://demetra-varietal-cindi.ngrok-free.dev
```

To publish the local dashboard to that URL, run:

```bat
run_public_ngrok_dashboard.bat
```

The public ngrok link requires internet. The offline village-point mode above does not.

Try example symptoms such as:

```text
High fever, cough, chest pain, chills, fatigue, phlegm and breathlessness
Severe headache, nausea, acidity, stiff neck and visual disturbances
Chest pain, breathlessness, sweating and vomiting
```

## Evaluate the Saved Model

From the project root:

```bash
python scripts/evaluate_model.py
```

Expected current result on `DATASETS/Testing.csv`:

```text
Accuracy: 0.9762
Precision: 0.9643
Recall: 0.9762
F1-score: 0.9683
```

Important note: the test file is small and clean, so these metrics should be presented as classroom dataset performance, not real-world clinical performance.

## Retrain the Model

From the project root:

```bash
python scripts/train_model.py
```

This regenerates the model artifacts in `BACKEND/models/`.

## Build the Validation Index

From the project root:

```bash
python scripts/build_ehr_validation_index.py
```

This creates `BACKEND/models/ehr_validation_index.json`, an aggregate evidence index from the cleaned public datasets. It stores counts and symptom support only, not patient identifiers.

## GitHub Notes

The repository is configured to avoid uploading local logs, databases, `ngrok.exe`, and very large raw datasets. If you want to publish the large raw datasets, use Git LFS, Kaggle links, Google Drive, or a separate dataset release.

## Team

- Rachith Bharadwaj T N - 24BDS062
- Dhanush Gowda N - 24BDS018
- Shreedhar M Kadkol - 24BDS076
- Kishan Kumar Y - 24BDS031

Guided by Dr. Utkarsh Mahadeo Khaire, Department of DSAI, IIIT Dharwad.
