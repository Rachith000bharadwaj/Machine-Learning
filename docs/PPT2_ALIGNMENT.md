# PPT-2 Project Alignment

`PPT-2.pdf` is the final presentation reference for this version of the project.
The report file was not updated during this alignment work.

## Implemented From PPT-2

- Flask Web-App for symptom entry, voice input, diagnosis output, evidence, and printable reports.
- Village-point dashboard mode for same-laptop or same-network offline typed symptom diagnosis.
- Public ngrok tunnel target: `https://demetra-varietal-cindi.ngrok-free.dev`.
- TensorFlow neural-network classifier for the main symptom-to-disease prediction path.
- Comparison workflow for Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine baselines.
- Spark-backed aggregate validation index generated from the cleaned CSV datasets.
- COVID-19 class support in the trained disease model, curated COVID datasets in `DATASETS/`, and a hallmark-symptom COVID screening rule in the Web-App.
- Fracture screening path backed by `DATASETS/bone_fracture_dataset.csv`.
- AES-256-GCM encrypted local diagnosis and feedback event logs under `WEB-APP/logs/`.
- Evaluation artifacts for accuracy, precision, recall, F1-score, confidence interval, confusion matrix, and charts.
- 100+ comprehensive clinical scenario test cases completed for manual validation.

## Current Demo Scale

- Disease classes: 42
- Symptom features: 134
- Fracture screening types: 11
- Manual clinical scenario test cases: 100+
- Offline typed diagnosis workflow: supported
- Public demo URL: `https://demetra-varietal-cindi.ngrok-free.dev`
- Saved test accuracy: 0.9762
- Saved test F1-score: 0.9683

## Main Files

- Web app: `WEB-APP/app.py`
- Frontend: `WEB-APP/templates/Index.html`, `WEB-APP/static/css/style.css`, `WEB-APP/static/js/app.js`
- Training workflow: `scripts/train_model.py`
- Evaluation workflow: `scripts/evaluate_model.py`
- Spark evidence index: `scripts/build_ehr_validation_index.py`
- Model artifacts and metrics: `BACKEND/models/`
