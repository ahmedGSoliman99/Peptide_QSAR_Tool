# Peptide QSAR Prediction Tool (Windows-Friendly)

This is a beginner-friendly peptide-focused QSAR app built with Streamlit.

## Download for Windows

If you only want to use the application, download the latest Windows installer from:

https://github.com/ahmedGSoliman99/Peptide_QSAR_Tool/releases/latest

Recommended file:

`PeptideQSAR_Setup.exe`

Portable single-file version:

`PeptideQSAR.exe`

On first launch, Windows may show SmartScreen because the executable is not code-signed. If you trust the file, choose **More info > Run anyway**.

## Quick Start (No coding)

Double-click:

`run_app.bat`

On first run it will automatically:

- create `.venv`
- install core packages from `requirements.txt`
- open `http://localhost:8501`

## Manual Setup (optional)

1. Install Python 3.10+ (64-bit) from [python.org](https://www.python.org/downloads/windows/).
2. In PowerShell, inside this folder:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional advanced packages:

```powershell
pip install -r requirements-optional.txt
```

## Main Features

- Input: manual sequences, FASTA, CSV, Excel, TXT, SDF
- Validation: invalid amino acid detection + duplicate detection
- Descriptors: AAC, dipeptide (optional), MW, pI, charge, GRAVY, instability, aliphatic, Boman, elemental, motif fingerprints, hydrophobic moment, helix/sheet propensity, charge density
- Modeling: regression, binary classification, multiclass classification
- Models: Linear/Ridge/Lasso/ElasticNet, RF, SVM/SVR, GBM, kNN, Logistic, Naive Bayes, MLP, optional XGBoost/LightGBM
- Non-linear model coverage: SVR/SVM, kNN, tree ensembles (RF/ExtraTrees/Boosting), MLP
- Target flexibility: activity/property columns including docking-style targets (DockingScore/Vina/Affinity)
- Evaluation: R2/RMSE/MAE or Accuracy/Precision/Recall/F1/ROC-AUC + confusion matrix
- Prediction: batch peptide scoring + ranking + structure-quality criteria scoring
- PCA analysis: explained variance, score scatter (PC1/PC2), and loading interpretation
- Explainability: feature importance + optional SHAP
- Export: CSV, Excel, HTML report, model `.joblib`
- Documentation: quick PDF plus manuscript-ready scientific methods file in `docs/`

## App Tabs

1. Home
2. Upload Data
3. Descriptors
4. Train Model
5. Evaluate
6. Predict New Peptides
7. Visualizations
8. Export Report
9. About / Documentation

## Project Structure

```text
Peptide_QSAR_Tool/
|-- app.py
|-- launch_streamlit.py
|-- run_app.bat
|-- requirements.txt
|-- requirements-optional.txt
|-- README.md
|-- peptide_qsar_single_file.py
|-- .streamlit/
|   |-- config.toml
|-- data/
|   |-- example_peptides.csv
|-- docs/
|   |-- Peptide_QSAR_Tool_Documentation.pdf
|   |-- Peptide_QSAR_Tool_Manuscript_Methods.md
|   |-- Peptide_QSAR_Tool_Manuscript_Methods.pdf
|-- modules/
|   |-- constants.py
|   |-- io_utils.py
|   |-- descriptor_engine.py
|   |-- preprocessing.py
|   |-- modeling.py
|   |-- evaluation.py
|   |-- visualization.py
|   |-- reporting.py
|   |-- structure_criteria.py
|-- saved_models/
```

## Scientific Documentation

For manuscript or thesis writing, open:

- `docs/Peptide_QSAR_Tool_Manuscript_Methods.md`
- `docs/Peptide_QSAR_Tool_Manuscript_Methods.pdf`

These files explain each major function, descriptor group, modeling step, scientific basis, limitations, and recommended reporting checklist.

## Windows `.exe` Launcher (optional)

Recommended single-file source for packaging:

`peptide_qsar_single_file.py`

Ready-built executable after packaging:

`dist/PeptideQSAR.exe`

You can rebuild it later by double-clicking:

`build_exe.bat`

Ready-built Windows installer:

`installer/PeptideQSAR_Setup.exe`

You can rebuild the installer later by double-clicking:

`build_installer.bat`

The installer uses Inno Setup and installs the app per-user under:

`%LOCALAPPDATA%\Programs\PeptideQSAR`

After installing optional packaging requirements:

```powershell
pip install -r requirements-optional.txt
```

Build an executable:

```powershell
pyinstaller --noconfirm --onefile --name PeptideQSAR peptide_qsar_single_file.py --collect-all streamlit --collect-all pandas --collect-all numpy --collect-all sklearn --collect-all Bio --collect-all plotly --collect-all scipy --collect-all openpyxl
```

Output:

`dist/PeptideQSAR.exe`

The single-file launcher embeds the app code, modules, example data, and PDF documentation, then extracts them to a local runtime folder when opened.

## Troubleshooting

- If `run_app.bat` says Python not found: install Python 3.10+ and reopen terminal.
- If installation fails on 32-bit Python: install 64-bit Python and rerun.
- If optional methods are unavailable (UMAP/SHAP/XGBoost/LightGBM): install `requirements-optional.txt`.

## Customization Pointers

- Add/change descriptors: `modules/descriptor_engine.py`
- Add/remove models: `modules/modeling.py` (`available_models_for_task`)
- Change preprocessing defaults: `modules/preprocessing.py` and train controls in `app.py`
- Adjust accepted sequence columns: `modules/io_utils.py`
