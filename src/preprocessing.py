"""
src/preprocessing.py
─────────────────────
Pipeline completo: imputación → ordinal → one-hot → SMOTE → escalado.
Todo se ajusta SOLO con train para evitar data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    SEED,
    INCOME_ORDER,
    EDUCATION_ORDER,
    ORDINAL_SOURCE_COLUMNS,
)


# ═══════════════════════════════════════════════════════════
# IMPUTACIÓN
# ═══════════════════════════════════════════════════════════

def impute_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer, SimpleImputer]:
    """
    Imputación segura usando solo train:
      - Numéricas: media
      - Categóricas: moda
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"[preprocessing] Columnas numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if len(num_cols) > 0:
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    if len(cat_cols) > 0:
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    assert X_train.isna().sum().sum() == 0, "NaN en X_train tras imputación"
    assert X_test.isna().sum().sum() == 0, "NaN en X_test tras imputación"

    print("[preprocessing] ✅ Imputación completada sin NaN")
    return X_train, X_test, num_imputer, cat_imputer


# ═══════════════════════════════════════════════════════════
# TRANSFORMACIÓN ORDINAL
# ═══════════════════════════════════════════════════════════

def _add_ordinal_feature(
    df: pd.DataFrame,
    col: str,
    mapping: dict,
) -> pd.DataFrame:
    """Crea columna ordinal + flag is_unknown."""
    ordinal_col = f"{col}_ordinal"
    unknown_col = f"{col}_is_unknown"

    if col in df.columns:
        df[ordinal_col] = df[col].map(mapping)
        df[unknown_col] = df[ordinal_col].isna().astype(int)

    return df


def apply_ordinal_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """
    Transforma Income_Category y Education_Level a ordinales.
    Crea flags is_unknown. Imputa con mediana de train.
    Elimina las columnas categóricas originales.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Aplicar transformación
    X_train = _add_ordinal_feature(X_train, "Income_Category", INCOME_ORDER)
    X_test = _add_ordinal_feature(X_test, "Income_Category", INCOME_ORDER)

    X_train = _add_ordinal_feature(X_train, "Education_Level", EDUCATION_ORDER)
    X_test = _add_ordinal_feature(X_test, "Education_Level", EDUCATION_ORDER)

    # Columnas ordinales creadas
    ordinal_cols = [
        "Income_Category_ordinal",
        "Education_Level_ordinal",
    ]
    ordinal_cols = [col for col in ordinal_cols if col in X_train.columns]

    # Imputación con mediana de train
    ordinal_imputer = SimpleImputer(strategy="median")

    if len(ordinal_cols) > 0:
        X_train[ordinal_cols] = ordinal_imputer.fit_transform(X_train[ordinal_cols])
        X_test[ordinal_cols] = ordinal_imputer.transform(X_test[ordinal_cols])

    # Eliminar las categóricas originales
    cols_to_drop = [c for c in ORDINAL_SOURCE_COLUMNS if c in X_train.columns]
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

    print("[preprocessing] ✅ Transformación ordinal terminada")
    return X_train, X_test, ordinal_imputer


# ═══════════════════════════════════════════════════════════
# ONE-HOT ENCODING
# ═══════════════════════════════════════════════════════════

def apply_onehot_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    OneHot encoding de las categóricas restantes.
    El encoder aprende solo de train.
    """
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    print(f"[preprocessing] OneHot sobre: {cat_cols}")

    if len(cat_cols) > 0:
        X_train_cat_enc = encoder.fit_transform(X_train[cat_cols])
        X_test_cat_enc = encoder.transform(X_test[cat_cols])

        if hasattr(encoder, "get_feature_names_out"):
            cat_feature_names = encoder.get_feature_names_out(cat_cols)
        else:
            cat_feature_names = encoder.get_feature_names(cat_cols)

        X_train_cat = pd.DataFrame(
            X_train_cat_enc,
            columns=cat_feature_names,
            index=X_train.index,
        )
        X_test_cat = pd.DataFrame(
            X_test_cat_enc,
            columns=cat_feature_names,
            index=X_test.index,
        )

        X_train_ready = pd.concat([X_train[num_cols], X_train_cat], axis=1)
        X_test_ready = pd.concat([X_test[num_cols], X_test_cat], axis=1)
    else:
        X_train_ready = X_train[num_cols].copy()
        X_test_ready = X_test[num_cols].copy()

    # Reset index y validaciones
    X_train_ready = X_train_ready.reset_index(drop=True)
    X_test_ready = X_test_ready.reset_index(drop=True)

    assert list(X_train_ready.columns) == list(X_test_ready.columns)
    assert X_train_ready.columns.is_unique

    print(f"[preprocessing] ✅ Encoding terminado. Shape: {X_train_ready.shape}")
    return X_train_ready, X_test_ready, encoder


# ═══════════════════════════════════════════════════════════
# SMOTE (solo train)
# ═══════════════════════════════════════════════════════════

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica SMOTE SOLO al set de entrenamiento."""
    smote = SMOTE(random_state=seed)

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
    y_train_smote = pd.Series(y_train_smote, name=y_train.name)

    print(
        f"[preprocessing] ✅ SMOTE aplicado. "
        f"Antes: {X_train.shape[0]} → Después: {X_train_smote.shape[0]}"
    )
    return X_train_smote, y_train_smote


# ═══════════════════════════════════════════════════════════
# ESCALADO
# ═══════════════════════════════════════════════════════════

def apply_scaling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """StandardScaler ajustado solo con train."""
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    assert X_train_scaled.isna().sum().sum() == 0
    assert X_test_scaled.isna().sum().sum() == 0

    print("[preprocessing] ✅ Escalado terminado")
    return X_train_scaled, X_test_scaled, scaler


# ═══════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════

def run_full_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Ejecuta todo el pipeline de preprocessing.
    Retorna un diccionario con todos los artefactos necesarios.
    """
    # 1. Imputación
    X_train, X_test, num_imputer, cat_imputer = impute_data(X_train, X_test)

    # 2. Ordinal encoding
    X_train, X_test, ordinal_imputer = apply_ordinal_encoding(X_train, X_test)

    # 3. OneHot encoding
    X_train_ready, X_test_ready, encoder = apply_onehot_encoding(X_train, X_test)

    # Reset targets
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # 4. SMOTE (solo train)
    X_train_smote, y_train_smote = apply_smote(X_train_ready, y_train)

    # 5. Escalado
    X_train_scaled, X_test_scaled, scaler = apply_scaling(X_train_smote, X_test_ready)

    # Validaciones finales anti-leakage
    assert X_test_scaled.shape == X_test_ready.shape, "X_test cambió de shape"
    assert y_test.shape[0] == X_test_scaled.shape[0], "y_test inconsistente"
    assert "Attrition_Flag" not in X_train_scaled.columns

    print("[preprocessing] 🎉 Pipeline completo ejecutado exitosamente")

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_smote": y_train_smote,
        "y_test": y_test,
        "feature_order": list(X_train_scaled.columns),
        # Artefactos para exportar
        "num_imputer": num_imputer,
        "cat_imputer": cat_imputer,
        "ordinal_imputer": ordinal_imputer,
        "encoder": encoder,
        "scaler": scaler,
    }