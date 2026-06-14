# Model Card: Symptom-to-Disease Classifier

## Model Summary

This model predicts a disease label from binary symptom features. It is used by the Flask web app as an educational preliminary diagnosis assistant.

## Intended Use

- Classroom ML project demonstration.
- Symptom-to-disease classification prototype.
- Support for ranked predictions, confidence display, and urgency labels.

## Not Intended For

- Real clinical diagnosis.
- Emergency decision-making.
- Prescription generation.
- Replacing doctors, nurses, or certified medical systems.

## Inputs

- Text symptoms entered by the user.
- The app maps recognized symptom phrases to the 137 binary symptom features used during training.

## Outputs

- Primary predicted disease.
- Top ranked disease predictions.
- Confidence score from the softmax output.
- Matched symptom evidence count.
- Rule-based urgency and suggested action text.

## Data

The saved model artifacts are based on `DATASETS/cleaned/Training_cleaned.csv`, using `prognosis` as the disease target and symptom columns as binary features. `DATASETS/cleaned/Testing_cleaned.csv` is used by `scripts/evaluate_model.py`.

The cleaned files are produced by `scripts/clean_datasets.py`. This is light project-focused cleaning only: it combines the app-compatible symptom CSVs, fixes disease-name spelling variants, and removes duplicate disease-symptom rows.

## Current Metrics

The saved model currently evaluates at 100 percent accuracy, precision, recall, and F1-score on the included clean test file of 123 samples. Training accuracy is 99.49 percent and validation accuracy is 98.98 percent for the saved final model.

This should be interpreted carefully because the test file is small and structured. It does not prove real-world clinical accuracy.

## Limitations

- User text may not match the exact symptom vocabulary.
- The model does not use laboratory results, imaging, vital signs, or doctor notes.
- Confidence scores are model probabilities, not calibrated medical certainty.
- Spark EHR validation and Android offline deployment are planned extensions, not part of the default runnable app.
- The Flask app can run offline on the local computer or local network, but package installation must be completed beforehand.

## Safety

The interface includes a medical disclaimer and recommends professional consultation. Urgency labels are simple rule-based guidance and should be treated as educational support only.
