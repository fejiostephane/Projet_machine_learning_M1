import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ajouter la racine du projet au path pour que 'src' soit importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import load_dataset, get_X_y, load_artifacts, predict_cycle, top_features

st.set_page_config(page_title="Valve Condition Monitor", layout="centered")
st.title("🔧 Valve Condition Monitoring — Random Forest")
st.caption("Prédiction de l'état de la valve (100/90/80/73)")

# Charger dataset et artefacts
@st.cache_data(show_spinner=False)
def _load_df():
    return load_dataset()

@st.cache_resource(show_spinner=False)
def _load_model_scaler():
    return load_artifacts()

try:
    df = _load_df()
    X, y = get_X_y(df)
    model, scaler = _load_model_scaler()
except Exception as e:
    st.error(f"Erreur de chargement: {e}")
    st.stop()

# Entrée utilisateur
max_cycle = len(df)
cycle_id = st.number_input(
    "Numéro de cycle",
    min_value=1,
    max_value=max_cycle,
    value=min(1500, max_cycle),
    step=1,
)

if st.button("Predict", type="primary"):
    with st.spinner("Prédiction en cours…"):
        info = predict_cycle(int(cycle_id))

    st.subheader("Résultat")
    st.write(f"Classe prédite : **{info['pred']}**")

    # Probas
    if info["proba"] is not None:
        st.subheader("Probabilités par classe")
        proba_df = pd.DataFrame({
            "Classe": info["classes"],
            "Probabilité": info["proba"],
        })
        st.bar_chart(proba_df.set_index("Classe"))
        st.caption("La somme des probas doit être ≈ 1")

    # Top features
    st.subheader("Top 5 features importantes")
    feats = top_features(model, info["feature_names"], k=5)
    if feats:
        top_df = pd.DataFrame(feats, columns=["Feature", "Importance"]).copy()
        st.table(top_df)

        # Valeurs pour ce cycle
        st.subheader("Valeurs de ces features pour ce cycle")
        vals = {f: info["x_row"][f] for f, _ in feats}
        vals_df = pd.DataFrame.from_dict(vals, orient="index", columns=["Valeur"]).reset_index()
        vals_df.rename(columns={"index": "Feature"}, inplace=True)
        st.table(vals_df)
    else:
        st.info("Le modèle ne fournit pas d'importances (non RandomForest)")

st.divider()
st.caption("Models: Random Forest | Scaler: StandardScaler | Données: FINAL_DATASET_READY_FOR_ML.csv")
