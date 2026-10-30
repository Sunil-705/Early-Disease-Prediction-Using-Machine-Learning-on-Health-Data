"""Streamlit interface for the educational binary-label prediction prototype."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT/'outputs'/'models'/'best_model.joblib'
META_PATH = ROOT/'outputs'/'models'/'model_metadata.json'

st.set_page_config(page_title='Early Disease Prediction — Project 29', page_icon='🩺', layout='centered')
st.title('Early Disease Prediction')
st.caption('HCLTech Career Shaper Programme · Project 29 · Educational ML prototype')
st.info('This application predicts the generic binary label in the supplied project dataset. The uploaded label does not identify a specific disease or diagnostic definition.')

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning('The trained model artifact is not present. Run `python scripts/train_models.py` from the project folder, then refresh this page.')
    st.stop()

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text(encoding='utf-8'))
st.subheader('Enter clinical feature values')
with st.form('patient_form'):
    c1,c2 = st.columns(2)
    with c1:
        age = st.number_input('Age (years)', min_value=18, max_value=90, value=45)
        bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        blood_pressure = st.number_input('Blood pressure (as recorded)', min_value=50, max_value=190, value=120)
    with c2:
        glucose = st.number_input('Glucose (unit not specified in source)', min_value=50, max_value=350, value=110)
        cholesterol = st.number_input('Cholesterol (type/unit not specified)', min_value=100, max_value=400, value=200)
        smoking = st.selectbox('Smoking recorded', options=[0,1], format_func=lambda x: 'No (0)' if x==0 else 'Yes (1)')
    submitted = st.form_submit_button('Predict supplied dataset label', use_container_width=True)

if submitted:
    X = pd.DataFrame([{'age':age,'bmi':bmi,'blood_pressure':blood_pressure,'glucose':glucose,'cholesterol':cholesterol,'smoking':smoking}])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0,1]) if hasattr(model,'predict_proba') else None
    st.subheader('Model output')
    st.metric('Predicted label', 'Positive (1)' if pred==1 else 'Negative (0)')
    if prob is not None:
        st.metric('Model probability for label 1', f'{prob:.1%}')
    st.caption(f"Model: {meta.get('selected_model','unknown')} · Features: {', '.join(meta.get('features',[]))}")
    st.warning('Educational demonstration only. This model is not clinically validated and must not be used to diagnose, treat, or rule out any disease. Consult a qualified healthcare professional for health concerns.')

with st.expander('About this prototype'):
    st.write('The training script splits the original 500 uploaded seed rows before augmentation, uses only synthetic records linked to training-fold parents during training, and evaluates on original seed rows held out from training.')
    st.write('Synthetic records are derived from the supplied seed data and are not independent patients. The dataset provenance, units, and exact meaning of the binary label need confirmation before any clinical interpretation.')
