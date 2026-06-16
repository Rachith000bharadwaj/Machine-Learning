# Project Audit and Improvement Notes

## Main Weaknesses Found

- The runnable Flask backend existed only inside a notebook, so the project was not easy to run from GitHub.
- The web app vocabulary did not match the saved trained model vocabulary.
- The interface used "EHR Evidence" wording even though Spark/MIMIC validation is not wired into the runnable demo.
- Local logs, databases, `ngrok.exe`, and very large raw datasets would make the GitHub repository messy.
- The report and slides were stronger than the implementation, creating a gap between claimed features and runnable features.

## Improvements Made

- Added `WEB-APP/app.py` as a real Flask application.
- Connected the web app to the saved backend TensorFlow model, label encoder, and 134-symptom vocabulary.
- Updated the web interface to show matched symptom evidence instead of unsupported EHR case counts.
- Added `requirements.txt`, `.gitignore`, a professional `README.md`, and reproducible scripts.
- Added `scripts/evaluate_model.py` for saved-model verification.
- Added `scripts/train_model.py` so the model can be rebuilt from `DATASETS/Training.csv`.
- Documented current implementation separately from future/experimental features.

## How to Present the Project

Describe it as a successful educational prototype:

> This project implements a symptom-based disease prediction assistant using a trained TensorFlow classifier and a Flask web interface. It supports ranked diagnosis predictions, confidence scores, matched symptom evidence, urgency labels, and printable reports. Spark EHR validation and Android deployment are planned extensions, not required for the current runnable demo.

This is stronger than claiming every future feature is fully completed.
