"""Train and compare binary classifiers without synthetic-to-test leakage.

The original 500 uploaded records are split first. Synthetic rows are used for training
only when their recorded parent seed row belongs to the training fold. The test set
contains original uploaded records only.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'
OUT_M = ROOT/'outputs'/'metrics'; OUT_M.mkdir(parents=True, exist_ok=True)
OUT_MODELS = ROOT/'outputs'/'models'; OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_FIG = ROOT/'outputs'/'figures'; OUT_FIG.mkdir(parents=True, exist_ok=True)
FEATURES = ['age','bmi','blood_pressure','glucose','cholesterol','smoking']


def get_models(seed=42):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=350, max_depth=7, min_samples_leaf=4,
            class_weight='balanced_subsample', random_state=seed, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=350, max_depth=7, min_samples_leaf=4,
            class_weight='balanced', random_state=seed, n_jobs=-1)
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0, min_child_weight=4,
            objective='binary:logistic', eval_metric='logloss', random_state=seed, n_jobs=2)
    except Exception as exc:
        warnings.warn(f'XGBoost unavailable; skipping: {exc}')
    return models


def main():
    seed = pd.read_csv(DATA/'seed_dataset.csv')
    synth = pd.read_csv(DATA/'synthetic_4500.csv')
    prov = pd.read_csv(DATA/'record_provenance.csv')
    if len(seed) != 500: raise ValueError('Expected original 500-row seed file')
    X = seed[FEATURES].astype(float)
    y = seed['label'].astype(int)
    train_idx, test_idx = train_test_split(np.arange(len(seed)), test_size=0.20,
        stratify=y, random_state=42)
    train_seed = seed.iloc[train_idx].copy()
    test_seed = seed.iloc[test_idx].copy()
    train_ids = set(train_seed['patient_id'].astype(str))

    synth_prov = prov[prov['record_origin']=='synthetic_bootstrap_perturbation'][['patient_id','parent_patient_id']]
    synth_train_ids = set(synth_prov.loc[synth_prov['parent_patient_id'].astype(str).isin(train_ids),'patient_id'].astype(str))
    synth_train = synth[synth['patient_id'].astype(str).isin(synth_train_ids)].copy()
    train_df = pd.concat([train_seed, synth_train], ignore_index=True)
    X_train = train_df[FEATURES].astype(float); y_train = train_df['label'].astype(int)
    X_test = test_seed[FEATURES].astype(float); y_test = test_seed['label'].astype(int)

    rows, fitted = [], {}
    for name, estimator in get_models().items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:,1] if hasattr(pipe,'predict_proba') else None
        try: auc = roc_auc_score(y_test, prob) if prob is not None else float('nan')
        except Exception: auc = float('nan')
        result = {
            'model': name,
            'accuracy': accuracy_score(y_test,pred),
            'balanced_accuracy': balanced_accuracy_score(y_test,pred),
            'precision_positive': precision_score(y_test,pred,zero_division=0),
            'recall_positive': recall_score(y_test,pred,zero_division=0),
            'f1_positive': f1_score(y_test,pred,zero_division=0),
            'macro_f1': f1_score(y_test,pred,average='macro',zero_division=0),
            'roc_auc': auc,
            'train_rows_including_synthetic': len(train_df),
            'test_rows_original_only': len(test_seed)
        }
        rows.append(result); fitted[name] = pipe
        print(name, {k:round(v,4) if isinstance(v,(int,float,np.floating)) else v for k,v in result.items()})

    metrics = pd.DataFrame(rows).sort_values(['macro_f1','balanced_accuracy'],ascending=False)
    metrics.to_csv(OUT_M/'model_comparison.csv', index=False)
    best_name = str(metrics.iloc[0]['model'])
    best = fitted[best_name]
    best_path = OUT_MODELS/'best_model.joblib'
    joblib.dump(best, best_path)
    metadata = {
        'selected_model': best_name,
        'features': FEATURES,
        'positive_label': 1,
        'negative_label': 0,
        'test_method': 'Stratified 80/20 split of original 500 uploaded records; synthetic rows linked to training-fold parents only; test rows are original-only.',
        'seed': 42,
        'n_original_records': int(len(seed)),
        'n_train_original': int(len(train_seed)),
        'n_train_synthetic': int(len(synth_train)),
        'n_test_original': int(len(test_seed)),
        'warning': 'The generic binary label is not tied to a specified disease and source provenance of the uploaded seed file is unknown. Educational demonstration only.'
    }
    (OUT_MODELS/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

    pred = best.predict(X_test)
    prob = best.predict_proba(X_test)[:,1]
    prediction_df = test_seed[['patient_id','label']].copy()
    prediction_df['predicted_label'] = pred
    prediction_df['positive_probability'] = prob
    prediction_df.to_csv(OUT_M/'best_model_test_predictions.csv', index=False)
    cm = confusion_matrix(y_test,pred,labels=[0,1])
    fig, ax = plt.subplots(figsize=(5.6,4.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1], labels=['Predicted 0','Predicted 1'])
    ax.set_yticks([0,1], labels=['Actual 0','Actual 1'])
    ax.set_title(f'Confusion matrix — {best_name}')
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center',fontsize=13)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout(); fig.savefig(OUT_FIG/'06_model_confusion_matrix.png',dpi=180); plt.close(fig)

    # SHAP global importance, best-effort and optional. Fall back to permutation importance.
    explain_path = OUT_FIG/'07_shap_global_importance.png'
    try:
        import shap
        model = best.named_steps['model']
        Xt = best.named_steps['scaler'].transform(X_test)
        Xtr = best.named_steps['scaler'].transform(X_train.sample(min(100,len(X_train)), random_state=42))
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(Xt)
            if isinstance(vals,list): vals = vals[-1]
            vals = np.asarray(vals)
            if vals.ndim == 3: vals = vals[:,:,1]
        else:
            explainer = shap.LinearExplainer(model, Xtr)
            vals = np.asarray(explainer.shap_values(Xt))
            if vals.ndim == 3: vals = vals[:,:,1]
        importance = np.abs(vals).mean(axis=0)
        order = np.argsort(importance)
        fig, ax = plt.subplots(figsize=(7,4.5))
        ax.barh(np.array(FEATURES)[order], importance[order], color='#5470c6')
        ax.set_title('SHAP global importance (mean absolute value)')
        ax.set_xlabel('Mean |SHAP value|')
        fig.tight_layout(); fig.savefig(explain_path,dpi=180); plt.close(fig)
        pd.DataFrame({'feature':FEATURES,'mean_absolute_shap':importance}).sort_values('mean_absolute_shap',ascending=False).to_csv(OUT_M/'shap_feature_importance.csv',index=False)
    except Exception as exc:
        print('SHAP visualization not produced:', repr(exc))
        try:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(best, X_test, y_test, n_repeats=20, random_state=42, scoring='f1_macro')
            vals = pi.importances_mean
            order = np.argsort(vals)
            fig, ax = plt.subplots(figsize=(7,4.5))
            ax.barh(np.array(FEATURES)[order], vals[order], color='#5470c6')
            ax.set_title('Permutation feature importance (SHAP unavailable)')
            ax.set_xlabel('Mean decrease in macro-F1')
            fig.tight_layout(); fig.savefig(OUT_FIG/'07_permutation_importance_fallback.png',dpi=180); plt.close(fig)
            pd.DataFrame({'feature':FEATURES,'permutation_importance':vals}).sort_values('permutation_importance',ascending=False).to_csv(OUT_M/'permutation_importance.csv',index=False)
        except Exception as exc2: print('Fallback importance not produced:',repr(exc2))

    print('\nSelected:', best_name)
    print('Saved:', best_path)
    print('Metrics:', OUT_M/'model_comparison.csv')
    print('Test report:\n', classification_report(y_test,pred,zero_division=0))

if __name__ == '__main__': main()
