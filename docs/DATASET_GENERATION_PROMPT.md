# Dataset Generation Prompt (transcribed from the supplied prompt image)

**Project:** Early Disease Prediction Using Machine Learning on Health Data  
**Domain:** Digital-Data-AI / AIML

> Use PIMA Indians Diabetes Dataset (UCI ML Repository) and Heart Disease Dataset. Supplement with 2,000 synthetic balanced records.

## Additional instructions

- Generate result as a CSV with the exact same columns as `dataset.csv`.
- Use realistic value ranges (not all zeroes or random text).
- Maintain consistent data types per column.
- Label anomalies / faults at approximately 2–5% of rows, if applicable.
- Keep patient identifiers sequential and consistent.

## Required columns

`patient_id, age, bmi, blood_pressure, glucose, cholesterol, smoking, label`

## Counts stated in prompt

- Current row count: 500.
- Suggested additional rows: 4,500 (to reach 5,000 total).
- Separate narrative instruction: supplement with 2,000 synthetic balanced records.

The package preserves both stated targets: `dataset_seed_plus_2000.csv` is 2,500 rows; `dataset.csv` is 5,000 rows. Both use the same eight-column schema. `synthetic_balanced_2000.csv` contains only the additional 2,000 balanced synthetic rows.

## Ethical note

Disease prediction models must not be used for clinical diagnosis without medical supervision. Include a disclaimer and handle class imbalance carefully.
