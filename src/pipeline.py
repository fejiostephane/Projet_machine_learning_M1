import os
import json
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import joblib

DATA_FILE_CANDIDATES = [
    "FINAL_DATASET_READY_FOR_ML.csv",
    "FINAL_dataset_with_labels.csv",
    "FINAL_dataset_with_labels.txt",
]
MODELS_DIR = "models"
MODEL_FILE = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")

EXPECTED_CLASSES = [100, 90, 80, 73]
EXPECTED_N_FEATURES = 1945


def _find_data_file() -> str:
    for f in DATA_FILE_CANDIDATES:
        if os.path.isfile(f):
            return f
    raise FileNotFoundError(
        f"Aucun fichier dataset trouvé parmi: {DATA_FILE_CANDIDATES}"
    )


def load_dataset() -> pd.DataFrame:
    """Charge le dataset final prêt ML."""
    data_file = _find_data_file()
    if data_file.endswith(".csv"):
        df = pd.read_csv(data_file)
    else:
        # Fallback for .txt (assume CSV with separators)
        df = pd.read_csv(data_file, sep=",", header=0)
    return df


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Retourne X (features) et y (label).
    Exclut : la colonne label ET la colonne 'cycle' (identifiant, pas une feature).
    Vérifie que X a exactement EXPECTED_N_FEATURES colonnes.
    """
    label_col_candidates = [
        "valve_condition",  # recommandé
        "label",
        "target",
    ]
    label_col: Optional[str] = None
    for c in label_col_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        # supposer que la dernière colonne est le label si elle existe
        if df.shape[1] >= EXPECTED_N_FEATURES + 2:  # cycle + 1945 features + label
            label_col = df.columns[-1]
        else:
            raise ValueError(f"Impossible de trouver la colonne label. Colonnes: {df.columns.tolist()}")
    
    # Colonnes à exclure : label + cycle (identifiant, pas une feature)
    cols_to_drop = [label_col]
    if "cycle" in df.columns:
        cols_to_drop.append("cycle")
    
    X = df.drop(columns=cols_to_drop)
    y = df[label_col]
    
    # Vérifier que X a exactement le nombre de features attendu
    if X.shape[1] != EXPECTED_N_FEATURES:
        raise ValueError(
            f"X a {X.shape[1]} features, {EXPECTED_N_FEATURES} attendues. "
            f"Colonnes exclues : {cols_to_drop}. Colonnes totales : {df.shape[1]}"
        )
    
    return X, y


def sequential_split(X: pd.DataFrame, y: pd.Series, train_end_idx: int = 2000):
    """Split séquentiel: train = 1..train_end_idx, test = le reste."""
    X_train = X.iloc[:train_end_idx].copy()
    y_train = y.iloc[:train_end_idx].copy()
    X_test = X.iloc[train_end_idx:].copy()
    y_test = y.iloc[train_end_idx:].copy()
    return X_train, X_test, y_train, y_test


def load_artifacts():
    """Charge le modèle et le scaler depuis models/."""
    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_FILE}")
    if not os.path.isfile(SCALER_FILE):
        raise FileNotFoundError(f"Scaler introuvable: {SCALER_FILE}")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler


def get_cycle_row(df: pd.DataFrame, cycle_id: int) -> pd.Series:
    """Récupère la ligne correspondant au cycle.
    Si une colonne 'cycle' existe, on l'utilise, sinon index = cycle_id-1.
    """
    cycle_col_candidates = ["cycle", "Cycle", "cycle_id"]
    cycle_col = None
    for c in cycle_col_candidates:
        if c in df.columns:
            cycle_col = c
            break
    if cycle_col is not None:
        row = df.loc[df[cycle_col] == cycle_id]
        if row.empty:
            raise ValueError(f"Cycle {cycle_id} introuvable dans la colonne {cycle_col}")
        return row.iloc[0]
    else:
        idx = cycle_id - 1
        if not (0 <= idx < len(df)):
            raise IndexError(f"cycle_id {cycle_id} en dehors de 1..{len(df)}")
        return df.iloc[idx]


def predict_cycle(cycle_id: int):
    """Prédit la classe et les probas pour un cycle donné."""
    df = load_dataset()
    X, y = get_X_y(df)
    model, scaler = load_artifacts()

    # Récupérer la ligne du cycle
    row = get_cycle_row(df, cycle_id)
    
    # Récupérer les features (exclure cycle et label)
    # X.columns contient déjà les bonnes colonnes (sans cycle, sans label)
    label_col = y.name
    x_row = row[X.columns]  # Prendre seulement les colonnes de X
    X_row_df = pd.DataFrame([x_row.values], columns=X.columns)

    # Standardiser
    X_row_scaled = scaler.transform(X_row_df.values)

    # Prédiction
    pred = model.predict(X_row_scaled)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row_scaled)[0]
        class_order = list(model.classes_)
    else:
        class_order = EXPECTED_CLASSES

    return {
        "cycle_id": cycle_id,
        "pred": int(pred),
        "proba": (proba.tolist() if proba is not None else None),
        "classes": class_order,
        "feature_names": X.columns.tolist(),
        "x_row": X_row_df.iloc[0].to_dict(),
    }

    # Prédiction
    pred = model.predict(X_row_scaled)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row_scaled)[0]
        class_order = list(model.classes_)
    else:
        class_order = EXPECTED_CLASSES

    return {
        "cycle_id": cycle_id,
        "pred": int(pred),
        "proba": (proba.tolist() if proba is not None else None),
        "classes": class_order,
        "feature_names": X.columns.tolist(),
        "x_row": X_row_df.iloc[0].to_dict(),
    }


def top_features(model, feature_names: List[str], k: int = 5):
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    idxs = np.argsort(importances)[-k:][::-1]
    return [(feature_names[i], float(importances[i])) for i in idxs]


if __name__ == "__main__":
    # Petit test manuel
    info = predict_cycle(1500)
    print(json.dumps(info, indent=2, ensure_ascii=False))
