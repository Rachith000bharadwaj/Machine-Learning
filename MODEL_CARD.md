# Model Card: Symptom-to-Disease Classifier

## Model Summary

This model predicts a disease label from binary symptom features. It is used by the Flask web app as an educational preliminary diagnosis assistant.

## Intended Use

- Classroom ML project demonstration.
- Symptom-to-disease classification prototype.
- Support for ranked predictions, confidence display, and urgency labels.
- Aggregate EHR-style evidence counts from cleaned public reference datasets.
- Final presentation alignment with the Flask Web-App workflow shown in `PPT-2.pdf`.

## Not Intended For

- Real clinical diagnosis.
- Emergency decision-making.
- Prescription generation.
- Replacing doctors, nurses, or certified medical systems.

## Inputs

- Text symptoms entered by the user.
- The app maps recognized symptom phrases to the 134 binary symptom features used during training.

## Outputs

- Primary predicted disease.
- Top ranked disease predictions.
- Confidence score from the softmax output.
- COVID screening result when hallmark symptoms include high fever, cough, and loss of smell or taste.
- Fracture screening result when injury-specific symptoms match the fracture dataset.
- Matched symptom evidence count.
- Aggregate reference-case evidence and symptom support counts.
- Rule-based urgency and suggested action text.

## Data

The saved model artifacts are based on `DATASETS/Training.csv`, using `prognosis` as the disease target and symptom columns as binary features. `DATASETS/Testing.csv` is used by `scripts/evaluate_model.py`.

The aggregate validation index is built from cleaned CSV datasets in `DATASETS/`, including the symptom-disease training/testing files, patient-profile symptom data, augmented disease-symptom references, and syditriage records. It stores counts and symptom aggregates only.

`scripts/train_model.py` also records comparison metrics for Logistic Regression, Random Forest, Gradient Boosting, and Support Vector Machine baselines.

## Current Metrics

The saved model currently evaluates on the included clean test file at:

- Accuracy: 0.9762
- Precision: 0.9643
- Recall: 0.9762
- F1-score: 0.9683

This should be interpreted carefully because the test file is small and structured. It does not prove real-world clinical accuracy.

## Limitations

- User text may not match the exact symptom vocabulary.
- The model does not use laboratory results, imaging, vital signs, or doctor notes.
- Confidence scores are model probabilities, not calibrated medical certainty.
- The Spark-backed validation index is aggregate public-dataset evidence, not live hospital EHR access.
- Android offline deployment is a planned extension, not part of the default runnable app.

## Safety

The interface includes a medical disclaimer and recommends professional consultation. Urgency labels are simple rule-based guidance and should be treated as educational support only.

Diagnosis and feedback events are stored locally as AES-256-GCM encrypted JSON lines when the project is installed with the listed requirements.
