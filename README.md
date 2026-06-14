# ML-Powered Medical Diagnosis Assistant

Repository: [Rachith000bharadwaj/Machine-Learning](https://github.com/Rachith000bharadwaj/Machine-Learning)

An educational machine learning project that predicts a preliminary diagnosis from patient-reported symptoms. The current runnable system includes a Flask web app, a trained TensorFlow classifier, confidence scores, matched symptom evidence, urgency labels, recommended actions, and printable reports.

This project is not a medical device and must not be used as a replacement for qualified healthcare advice.

## Current Implementation

- Symptom-based disease prediction using a saved TensorFlow model.
- 137 symptom features and 42 disease classes from the cleaned project dataset.
- Flask web app with text input, browser voice input support, ranked predictions, urgency level, and printable diagnosis report.
- Repeatable evaluation script for the saved model.
- Light dataset-cleaning script and training script to rebuild the model from project symptom datasets.
- Offline/local-network running without ngrok or internet-loaded page assets.
- Clean GitHub structure with large raw datasets and local logs ignored.

## What Is Planned or Experimental

The original project ideas include Spark EHR validation, Android deployment, encryption, and larger clinical datasets. Those are useful future directions, but the default runnable GitHub demo currently focuses on the trained symptom classifier and Flask web application.

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
    cleaned/
      Training_cleaned.csv
      Testing_cleaned.csv
      cleaning_summary.json
    symptom_Description.csv
    symptom_precaution.csv
  WEB-APP/
    app.py
    templates/Index.html
    static/css/style.css
    static/js/app.js
  scripts/
    clean_datasets.py
    train_model.py
    evaluate_model.py
  docs/
    PROJECT_AUDIT.md
    GITHUB_UPLOAD_STEPS.md
  run_offline.bat
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

For offline village/local use on Windows, double-click:

```text
run_offline.bat
```

Or run manually:

```bash
cd WEB-APP
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

For another phone or laptop, connect it to the same hotspot or Wi-Fi and open the local-network URL printed by the app, for example:

```text
http://10.x.x.x:5000
```

This does not need ngrok. Install Python packages before going offline.

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

Expected current result on `DATASETS/cleaned/Testing_cleaned.csv` after the final cleanup:

```text
Samples: 123
Classes: 42
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1-score: 1.0000
```

Important note: the test file is small and clean, so perfect metrics should be presented as classroom dataset performance, not real-world clinical performance. The training script reported 0.9949 training accuracy and 0.9898 validation accuracy for the saved final model.

## Clean the Project Dataset

From the project root:

```bash
python scripts/clean_datasets.py
```

This performs light project-focused cleaning only. It combines the existing symptom-to-disease CSVs, fixes label spelling variants, removes duplicate disease-symptom rows, and writes:

```text
DATASETS/cleaned/Training_cleaned.csv
DATASETS/cleaned/Testing_cleaned.csv
DATASETS/cleaned/cleaning_summary.json
```

It does not deep-clean unrelated numeric medical datasets.

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
