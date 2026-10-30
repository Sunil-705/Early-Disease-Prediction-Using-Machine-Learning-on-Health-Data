"""Generate the documented 5,000-row dataset from data/seed_dataset.csv.

The uploaded 500-row seed data is preserved. The 4,500 generated records are
class-conditioned bootstrap perturbations with inherited labels (not independent
observations). A separate balanced 2,000-row synthetic file is also exported.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
SEED_PATH = DATA/'seed_dataset.csv'
SEED = 20261030
COLUMNS = ['patient_id','age','bmi','blood_pressure','glucose','cholesterol','smoking','label']
NOISE = {'age':2.5,'bmi':1.15,'blood_pressure':5.5,'glucose':8.0,'cholesterol':9.0}
BOUNDS = {'age':(18,90),'bmi':(15.0,50.0),'blood_pressure':(50,190),'glucose':(50,350),'cholesterol':(100,400)}


def main():
    if not SEED_PATH.exists(): raise FileNotFoundError('Place the original 500-row file at data/seed_dataset.csv first.')
    seed = pd.read_csv(SEED_PATH)
    if list(seed.columns) != COLUMNS: raise ValueError(f'Expected columns {COLUMNS}; found {list(seed.columns)}')
    if len(seed) != 500: raise ValueError(f'Expected 500 seed rows, got {len(seed)}')
    if seed.patient_id.duplicated().any(): raise ValueError('Seed patient_id values must be unique')
    rng = np.random.default_rng(SEED)
    rows=[]; provenance=[]; next_id=501
    for label in (0,1):
        pool=seed[seed.label==label].reset_index(drop=True)
        if pool.empty: raise ValueError(f'No seed rows for label {label}')
        for parent_idx in rng.integers(0,len(pool),size=2250):
            p=pool.iloc[int(parent_idx)]
            r={'patient_id':f'P{next_id:04d}',
               'age':int(round(float(p.age)+rng.normal(0,NOISE['age']))),
               'bmi':round(float(p.bmi)+rng.normal(0,NOISE['bmi']),1),
               'blood_pressure':int(round(float(p.blood_pressure)+rng.normal(0,NOISE['blood_pressure']))),
               'glucose':int(round(float(p.glucose)+rng.normal(0,NOISE['glucose']))),
               'cholesterol':int(round(float(p.cholesterol)+rng.normal(0,NOISE['cholesterol']))),
               'smoking':int(p.smoking),'label':int(p.label)}
            for c,(lo,hi) in BOUNDS.items(): r[c]=max(lo,min(hi,r[c]))
            rows.append(r)
            provenance.append({'patient_id':r['patient_id'],'record_origin':'synthetic_bootstrap_perturbation',
                'parent_patient_id':str(p.patient_id),'anomaly_flag':0,'anomaly_description':'','generation_seed':SEED})
            next_id+=1
    synth=pd.DataFrame(rows,columns=COLUMNS)
    # Flag 3% of synthetic rows with a boundary/outlier feature; do not alter labels.
    n_anom=round(len(synth)*0.03)
    for i in rng.choice(len(synth),size=n_anom,replace=False):
        c=str(rng.choice(list(BOUNDS)))
        lo,hi=BOUNDS[c]
        if c=='age': value=int(lo+rng.integers(0,4)) if rng.random()<.5 else int(hi-rng.integers(0,4))
        elif c=='bmi': value=round(float(lo+rng.uniform(0,1)),1) if rng.random()<.5 else round(float(hi-rng.uniform(0,1)),1)
        else: value=int(lo+rng.integers(0,6)) if rng.random()<.5 else int(hi-rng.integers(0,6))
        synth.loc[i,c]=value
        provenance[i]['anomaly_flag']=1
        provenance[i]['anomaly_description']=f'Boundary/outlier value injected in {c}; retained for robustness testing'
    synth.to_csv(DATA/'synthetic_4500.csv',index=False)
    balanced=pd.concat([synth[synth.label==0].head(1000),synth[synth.label==1].head(1000)],ignore_index=True)
    balanced.to_csv(DATA/'synthetic_balanced_2000.csv',index=False)
    seed_plus_2k=pd.concat([seed,balanced],ignore_index=True)
    seed_plus_2k.to_csv(DATA/'dataset_seed_plus_2000.csv',index=False)
    full=pd.concat([seed,synth],ignore_index=True)
    full.to_csv(DATA/'dataset.csv',index=False)
    p0=pd.DataFrame({'patient_id':seed.patient_id,'record_origin':'user_uploaded_seed','parent_patient_id':'',
                     'anomaly_flag':0,'anomaly_description':'','generation_seed':''})
    pd.concat([p0,pd.DataFrame(provenance)],ignore_index=True).to_csv(DATA/'record_provenance.csv',index=False)
    manifest={'seed_rows':len(seed),'synthetic_rows':len(synth),'total_rows':len(full),
      'synthetic_balanced_2000_rows':len(balanced),'seed_plus_2000_total_rows':len(seed_plus_2k),
      'anomaly_rows_flagged':int(n_anom),'random_seed':SEED,
      'method':'Class-conditioned bootstrap perturbation of supplied seed rows; labels inherited from parent rows.',
      'warning':'Synthetic rows are not independent patient observations and must not be used as a held-out test set.'}
    (DATA/'generation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
