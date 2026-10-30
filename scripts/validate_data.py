"""Validate the project's fixed eight-column clinical dataset schema."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
EXPECTED = ['patient_id','age','bmi','blood_pressure','glucose','cholesterol','smoking','label']


def main():
    p = DATA / 'dataset.csv'
    df = pd.read_csv(p)
    errors = []
    if list(df.columns) != EXPECTED:
        errors.append(f'Column mismatch: {list(df.columns)}')
    if len(df) != 5000:
        errors.append(f'Expected 5,000 rows; got {len(df)}')
    if df['patient_id'].isna().any() or df['patient_id'].duplicated().any():
        errors.append('patient_id must be non-null and unique')
    if df.isna().any().any():
        errors.append('Dataset contains null cells')
    for c in EXPECTED[1:]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f'{c} must be numeric')
    if not set(df['label'].unique()).issubset({0,1}):
        errors.append('label must contain only 0 and 1')
    if not set(df['smoking'].unique()).issubset({0,1}):
        errors.append('smoking must contain only 0 and 1')
    prov = pd.read_csv(DATA/'record_provenance.csv')
    if len(prov) != len(df) or prov['patient_id'].duplicated().any():
        errors.append('Provenance must contain one row per patient_id')
    flagged = int(prov['anomaly_flag'].fillna(0).astype(int).sum())
    if errors:
        print('VALIDATION FAILED')
        for e in errors: print(' -',e)
        return 1
    print('VALIDATION PASSED')
    print(f'Rows: {len(df):,}')
    print(f'Columns: {len(df.columns)}')
    print('Label counts:', df['label'].value_counts().sort_index().to_dict())
    print('Unique IDs:', df['patient_id'].nunique())
    print('Missing cells:', int(df.isna().sum().sum()))
    print(f'Anomaly-flagged records in provenance: {flagged:,} ({flagged/len(df):.2%})')
    print('Note: flagged records are synthetic boundary/outlier test cases, not confirmed clinical anomalies.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
