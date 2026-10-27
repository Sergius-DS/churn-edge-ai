"""
src/data_loading.py
───────────────────
Carga del CSV, eliminación de columnas de fuga,
conversión del target y train/test split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    TARGET,
    SEED,
    TEST_SIZE,
    LEAKAGE_COLUMNS,
    NUMERIC_LIKE_COLUMNS,
    SYNTHETIC_IOT_CSV,
    IOT_FEATURES,
)


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el CSV y verifica que existe."""
    assert path.exists(), f"No se encontró el archivo: {path}"
    df = pd.read_csv(path)
    print(f"[data_loading] Shape inicial: {df.shape}")
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas de fuga (Naive Bayes, CLIENTNUM)
    y cualquier columna que contenga 'naive_bayes_classifier'.
    """
    nb_cols = [
        col for col in df.columns
        if "naive_bayes_classifier" in col.lower()
    ]
    cols_to_drop = list(set(LEAKAGE_COLUMNS + nb_cols))
    cols_present = [col for col in cols_to_drop if col in df.columns]

    df_clean = df.drop(columns=cols_present).copy()
    print(f"[data_loading] Columnas eliminadas por leakage/ID: {cols_present}")
    print(f"[data_loading] Shape después de limpiar: {df_clean.shape}")
    return df_clean


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa X e y. Convierte Attrition_Flag a binario:
      'Existing Customer' → 0
      'Attrited Customer' → 1
    """
    assert TARGET in df.columns, f"No se encontró la columna objetivo: {TARGET}"

    y_raw = df[TARGET].copy()
    X = df.drop(columns=[TARGET])

    if y_raw.dtype == "object":
        y_clean = y_raw.astype(str).str.strip()
        class_map = {
            "Existing Customer": 0,
            "Attrited Customer": 1,
        }
        y = y_clean.map(class_map)

        unmapped = y_clean[y.isna()].unique()
        if len(unmapped) > 0:
            raise ValueError(f"Categorías sin mapear en {TARGET}: {unmapped}")
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        if y.isna().any():
            raise ValueError("Hay valores nulos o no numéricos en la variable objetivo")
        y = y.astype(int)

    assert y.notna().all(), "La variable objetivo tiene valores nulos"
    print(f"[data_loading] Target convertido: {y.value_counts().to_dict()}")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split estratificado."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    print(f"[data_loading] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def convert_numeric_like(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte columnas que parecen numéricas pero vienen como object."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in NUMERIC_LIKE_COLUMNS:
        if col in X_train.columns and col in X_test.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
            print(f"[data_loading] Convertida a numérica: {col}")

    return X_train, X_test


def convert_iot_numeric(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convierte explícitamente las features IoT a numéricas.
    Esto previene errores si el CSV sintético guardó algún conteo como string.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in IOT_FEATURES:
        if col in X_train.columns:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")

        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return X_train, X_test


def verify_no_leakage(X_train: pd.DataFrame) -> None:
    """Assert final: que no quedaron columnas con fuga."""
    bad_cols = [
        col for col in X_train.columns
        if "naive_bayes" in col.lower() or "attrition_flag" in col.lower()
    ]
    assert len(bad_cols) == 0, f"Posible fuga de datos en features: {bad_cols}"
    print("[data_loading] ✅ OK: no se detectan columnas con fuga aparente")


def load_and_prepare(
    path: Path = None,
    with_iot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline completo de carga y preparación.

    Si with_iot=True:
        carga data/processed/credit_card_churn_iot_synthetic.csv

    Si with_iot=False:
        carga data/raw/Credit_card_churn.csv
    """
    if path is not None and with_iot:
        raise ValueError("No uses path y with_iot=True al mismo tiempo.")

    if path is None:
        path = SYNTHETIC_IOT_CSV if with_iot else DATA_PATH

    path = Path(path)

    if with_iot and not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset IoT: {path}. "
            "Ejecuta primero: python scripts/generate_synthetic_iot.py"
        )

    df = load_raw_data(path)
    df_clean = drop_leakage_columns(df)

    X, y = encode_target(df_clean)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test = convert_numeric_like(X_train, X_test)
    X_train, X_test = convert_iot_numeric(X_train, X_test)

    if with_iot:
        missing_train = [c for c in IOT_FEATURES if c not in X_train.columns]
        missing_test = [c for c in IOT_FEATURES if c not in X_test.columns]

        assert not missing_train, f"Faltan IoT features en train: {missing_train}"
        assert not missing_test, f"Faltan IoT features en test: {missing_test}"

    verify_no_leakage(X_train)

    # Assertions de consistencia
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    return X_train, X_test, y_train, y_test