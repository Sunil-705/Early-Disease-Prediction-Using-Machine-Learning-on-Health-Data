# Early Disease Prediction Using Machine Learning on Health Data

**HCLTech Career Shaper Programme · Project 29 · Digital-Data-AI / AIML**  
**Project duration:** 4 weeks · **Format:** Individual project

This repository is a runnable educational prototype and reproducible data-analysis scaffold based on the supplied Project 29 brief, the uploaded eight-column CSV, and the Dataset Generation Prompt.

## What is included

- A fixed-schema clinical-feature dataset (`data/dataset.csv`) with 5,000 rows.
- The 500-row uploaded seed dataset (`data/seed_dataset.csv`) preserved separately.
- 4,500 generated synthetic rows (`data/synthetic_4500.csv`).
- A separate 2,000-row balanced synthetic sample (`data/synthetic_balanced_2000.csv`) and a 2,500-row seed-plus-2,000 file (`data/dataset_seed_plus_2000.csv`) to satisfy the prompt's separate 2,000-record instruction.
- Row-level provenance (`data/record_provenance.csv`) identifying original vs synthetic rows, synthetic parent IDs, and 3% flagged boundary/outlier test cases.
- A completed data dictionary (`data/data_dictionary.csv`).
- Data validation, EDA visualisations, model comparison, SHAP explainability (where supported), and a Streamlit interface.
- An optional source-fetch script for the Pima Indians Diabetes and UCI Heart Disease datasets. Those source datasets are stored separately and are not silently merged into the generic eight-column file.

## Important data and scope notes

### 1. Meaning of the current target

The uploaded CSV contains a binary `label` column. The prompt describes it as `disease/no disease`, but it does not specify the exact disease, diagnostic criteria, or how the labels were assigned. Accordingly, the supplied dataset powers a **generic binary-label educational model**. It must not be described as a clinically validated diabetes, heart-disease, or hypertension diagnosis model unless those target definitions are supplied and separately validated.

### 2. Public source datasets are separate

The Pima Indians Diabetes dataset and UCI Heart Disease dataset use different source features and different target definitions. For example, Pima includes pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age, and outcome; UCI Heart Disease includes age, sex, chest-pain type, resting blood pressure, cholesterol, fasting-blood-sugar indicator, ECG/heart-test fields, and a heart-disease target. These source schemas do not match the uploaded eight-column schema. They are therefore downloaded to `data/external/` by `scripts/fetch_public_sources.py` and kept separate. No absent feature is invented to force a merge.

The modern UCI repository's dataset ID 45 is Heart Disease. The current UCI page for dataset ID 34 is a diabetes time-series dataset, not the Pima Indians Diabetes table. The fetch script therefore tries the legacy UCI Pima archive URL and then a commonly used public CSV mirror; check the source citation and licence before external distribution.

### 3. Synthetic row count conflict in the supplied prompt

The prompt text asks for 2,000 synthetic balanced records, while the screenshot also says to add 4,500 rows to reach 5,000 total from 500 current rows. To preserve both instructions without ambiguity, the package contains:

- `dataset.csv`: 500 uploaded seed rows + 4,500 synthetic rows = 5,000 total.
- `synthetic_balanced_2000.csv`: a separate balanced sample of 1,000 rows per class.

The generated rows are class-conditioned bootstrap perturbations of the uploaded seed data. They inherit their parent's label; they are not new independent patients and do not add independent medical evidence. A provenance file records the parent seed ID. Do not randomly split the augmented file and claim the synthetic test rows are independent.

### 4. Evaluation design

`scripts/train_models.py` splits the original 500 seed rows first (stratified 80/20). Synthetic rows are eligible for training only when their recorded parent seed ID belongs to the training fold. The held-out test fold contains only original seed rows. This prevents synthetic variants of test parents from leaking into training. The resulting metrics describe this small uploaded dataset and should not be generalized to clinical populations.

## Quick start

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/validate_data.py
python scripts/run_eda.py
python scripts/train_models.py
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_data.py
python scripts/run_eda.py
python scripts/train_models.py
streamlit run app.py
```

Open the local URL printed by Streamlit (typically `http://localhost:8501`).

If you replace or edit the seed data, keep the required column order and 500-row count if you plan to rerun the bundled generation script. Then run:

```bash
python scripts/generate_dataset.py
python scripts/validate_data.py
python scripts/run_eda.py
python scripts/train_models.py
```

## Project workflow

1. **Data validation:** schema, data types, missing values, binary labels, unique IDs, and provenance.
2. **Data generation:** reproducible synthetic perturbation, with labels inherited from parent rows and 3% anomaly flags for robustness exercises.
3. **EDA:** class distribution, feature distributions, feature-by-label boxplots, correlation heatmap, and seed-vs-augmented comparisons.
4. **Modeling:** Logistic Regression, Random Forest, Extra Trees, and XGBoost (if available); comparison with accuracy, balanced accuracy, precision, recall, macro-F1, and ROC-AUC.
5. **Explainability:** SHAP global feature importance where the installed SHAP/model combination supports it; a permutation-importance fallback is generated if not.
6. **Interface:** Streamlit form for the six input features, displaying the model's prediction for the supplied binary label and a probability estimate.

## Project Screenshot

![Early Disease Prediction](outputs)



## Repository structure

```text
Early_Disease_Prediction_Project/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── seed_dataset.csv
│   ├── dataset.csv
│   ├── synthetic_4500.csv
│   ├── synthetic_balanced_2000.csv
│   ├── dataset_seed_plus_2000.csv
│   ├── data_dictionary.csv
│   ├── record_provenance.csv
│   ├── generation_manifest.json
│   └── external/ (optional downloaded public source datasets)
├── docs/
│   ├── DATA_AND_LABEL_LIMITATIONS.md
│   ├── DATASET_GENERATION_PROMPT.md
│   ├── REFERENCES.md
│   └── reference/ (supplied brief, synopsis template, prompt image)
├── scripts/
│   ├── generate_dataset.py
│   ├── validate_data.py
│   ├── run_eda.py
│   ├── train_models.py
│   └── fetch_public_sources.py
└── outputs/
    ├── figures/
    ├── metrics/
    └── models/
```

## Medical and privacy disclaimer

This prototype is for academic and educational demonstration only. It is not medically validated, is not a diagnostic device, and must not be used to diagnose, treat, screen, or rule out a disease. Seek advice from a qualified healthcare professional. Do not enter real patient-identifiable data. The `patient_id` values in the supplied file are treated as anonymised dataset IDs, but this has not been independently verified.

## References

See [`docs/REFERENCES.md`](docs/REFERENCES.md) and [`docs/DATA_AND_LABEL_LIMITATIONS.md`](docs/DATA_AND_LABEL_LIMITATIONS.md).
