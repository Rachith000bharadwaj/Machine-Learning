# Dataset Notes

The runnable demo uses these cleaned project files:

- `cleaned/Training_cleaned.csv` for model training.
- `cleaned/Testing_cleaned.csv` for model evaluation.
- `cleaned/cleaning_summary.json` for cleaning details.
- `symptom_Description.csv` and `symptom_precaution.csv` for future explanation features.

The cleaned files are created by:

```bash
python scripts/clean_datasets.py
```

This is light project-focused cleaning only. It combines the app-compatible symptom-to-disease CSVs, fixes disease-name spelling variants, removes duplicate disease-symptom rows, and keeps the model aligned with the current Flask app.

Final cleaned dataset size:

- 488 training rows.
- 123 testing rows.
- 42 disease classes.
- 137 binary symptom features.

Large raw datasets are kept out of Git by `.gitignore` so the repository remains easy to clone and upload:

- `Final_Augmented_dataset_Diseases_and_Symptoms.csv/`
- `Covid Data.csv/`
- `syditriage.csv/`
- `thyroid_cancer_risk_data.csv`

If those datasets are needed, publish them separately and link them in the README.
