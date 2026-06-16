# GitHub Upload Steps

Use these steps from the project root after checking the app and evaluation script.

```bash
git init -b main
git add .
git commit -m "Polish ML medical diagnosis assistant project"
```

Create a new GitHub repository named:

```text
ml-powered-medical-diagnosis-assistant
```

Then connect and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/ml-powered-medical-diagnosis-assistant.git
git push -u origin main
```

Large raw datasets are intentionally ignored. If you need them online, upload them separately through Git LFS, Kaggle, Google Drive, or a dataset release and link them from the README.
