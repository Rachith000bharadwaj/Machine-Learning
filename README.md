# ML-Powered Medical Diagnosis Assistant

An educational machine learning project that predicts a preliminary diagnosis from patient-reported symptoms. The current runnable system includes a Flask web app, a trained TensorFlow classifier, confidence scores, matched symptom evidence, urgency labels, recommended actions, and printable reports.

This project is not a medical device and must not be used as a replacement for qualified healthcare advice.

## Current Implementation

- Symptom-based disease prediction using a saved TensorFlow model.
- 134 symptom features and 42 disease classes from the training artifacts.
- Flask web app with text input, browser voice input support, ranked predictions, urgency level, and printable diagnosis report.
- Repeatable evaluation script for the saved model.
- Training script to rebuild the model from `DATASETS/Training.csv`.
- Clean GitHub structure with large raw datasets and local logs ignored.

## What Is Planned or Experimental

The project report and slides discuss Spark EHR validation, Android deployment, encryption, and large clinical datasets. Those are useful future directions, but the default runnable GitHub demo currently focuses on the trained symptom classifier and web application.

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
  docs/
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
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1-score: 1.0000
```

Important note: the test file is small and clean, so perfect metrics should be presented as classroom dataset performance, not real-world clinical performance.

## Retrain the Model

From the project root:

```bash
python scripts/train_model.py
```

This regenerates the model artifacts in `BACKEND/models/`.

## GitHub Notes

The repository is configured to avoid uploading local logs, databases, `ngrok.exe`, and very large raw datasets. If you want to publish the large raw datasets, use Git LFS, Kaggle links, Google Drive, or a separate dataset release.

## Team

- Rachith Bharadwaj T N - 24BDS062
- Dhanush Gowda N - 24BDS018
- Shreedhar M Kadkol - 24BDS076
- Kishan Kumar Y - 24BDS031

Guided by Dr. Utkarsh Mahadeo Khaire, Department of DSAI, IIIT Dharwad.
