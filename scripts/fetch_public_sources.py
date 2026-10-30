"""Optional download helper for public reference datasets.

Downloads are stored separately under data/external and are NOT merged into the
fixed eight-column dataset because the source schemas and target definitions differ.
"""
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data'/'external'
OUT.mkdir(parents=True, exist_ok=True)

PIMA_COLUMNS = ['pregnancies','glucose','blood_pressure','skin_thickness','insulin','bmi','diabetes_pedigree_function','age','outcome']
PIMA_URLS = [
 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
]


def fetch_pima():
    last = None
    for url in PIMA_URLS:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path = OUT/'pima_indians_diabetes_raw.csv'
            path.write_bytes(response.content)
            df = pd.read_csv(path, header=None, names=PIMA_COLUMNS)
            if len(df.columns) != 9 or len(df) < 500:
                raise ValueError('Downloaded file did not match expected Pima schema')
            df.to_csv(path,index=False)
            (OUT/'pima_source_url.txt').write_text(url+'\n',encoding='utf-8')
            print('Pima dataset saved:', path, 'rows=',len(df))
            return
        except Exception as exc:
            last = exc
    raise RuntimeError(f'Could not fetch Pima data from configured sources: {last}')


def fetch_heart():
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError('Install ucimlrepo with: pip install ucimlrepo') from exc
    heart = fetch_ucirepo(id=45)
    X = heart.data.features.copy()
    y = heart.data.targets.copy()
    out = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    path = OUT/'uci_heart_disease_raw.csv'
    out.to_csv(path,index=False)
    (OUT/'heart_source.txt').write_text('UCI Machine Learning Repository, Heart Disease, dataset ID 45; https://archive.ics.uci.edu/dataset/45/heart+disease\n',encoding='utf-8')
    print('Heart Disease dataset saved:',path,'rows=',len(out))

if __name__ == '__main__':
    fetch_pima()
    fetch_heart()
