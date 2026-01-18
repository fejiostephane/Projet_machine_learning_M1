import os
import numpy as np
import pandas as pd
import pytest

from src.pipeline import load_dataset, get_X_y, sequential_split, load_artifacts, predict_cycle, EXPECTED_CLASSES, EXPECTED_N_FEATURES


def test_dataset_alignment():
    df = load_dataset()
    X, y = get_X_y(df)
    assert len(X) == len(y), "X et y doivent avoir le même nombre de lignes"
    # cycle cohérent si colonne présente
    cycle_col = None
    for c in ["cycle", "Cycle", "cycle_id"]:
        if c in df.columns:
            cycle_col = c
            break
    if cycle_col:
        cycles = df[cycle_col].values
        assert len(cycles) == len(df)
        assert cycles.min() == 1
        assert cycles.max() == len(df)
    # nb de features attendu
    assert X.shape[1] == EXPECTED_N_FEATURES, f"Nombre de features attendu: {EXPECTED_N_FEATURES}"


def test_features_no_nan_inf():
    df = load_dataset()
    X, y = get_X_y(df)
    X_values = X.values
    assert not np.isnan(X_values).any(), "Pas de NaN dans les features"
    assert not np.isinf(X_values).any(), "Pas d'Inf dans les features"


def test_split_sequential():
    df = load_dataset()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = sequential_split(X, y, train_end_idx=2000)
    assert len(X_train) == 2000
    assert len(y_train) == 2000
    assert (X_train.index.max() == 1999)
    assert (X_test.index.min() == 2000)


def test_standardization_transform():
    df = load_dataset()
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = sequential_split(X, y, train_end_idx=2000)
    model, scaler = load_artifacts()
    # transform doit marcher et conserver dimensions
    Xt = scaler.transform(X_train.values)
    Xv = scaler.transform(X_test.values)
    assert Xt.shape == (len(X_train), X_train.shape[1])
    assert Xv.shape == (len(X_test), X_test.shape[1])
    # moyenne approx 0 sur train (quelques features uniquement)
    mean_approx = Xt[:, :50].mean()
    assert abs(mean_approx) < 0.2, "La standardisation devrait centrer ~0 sur train"


def test_prediction_single_cycle():
    # choisir un cycle valide
    info = predict_cycle(1500)
    assert info["pred"] in EXPECTED_CLASSES, "La prédiction doit être parmi les classes autorisées"
    if info["proba"] is not None:
        s = sum(info["proba"])
        assert abs(s - 1.0) < 1e-6, "La somme des probas doit être ≈ 1"


def test_model_artifacts_exist():
    assert os.path.isfile("models/random_forest_model.pkl"), "Modèle RF manquant"
    assert os.path.isfile("models/scaler.pkl"), "Scaler manquant"
