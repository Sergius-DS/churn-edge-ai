"""
src/export_edge.py
──────────────────
Exporta el modelo a JSON + contrato de inferencia para WASM/Rust.
Genera parity_sample.json para validar Python ↔ Rust.
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone

from src.config import (
    EDGE_DIR,
    SEED,
    TARGET,
    BEST_THRESHOLD,
    INCOME_ORDER,
    EDUCATION_ORDER,
    PARITY_SAMPLE_SIZE,
)


def _to_list(x):
    return np.asarray(x).tolist()


def _clean_mapping(mapping: dict) -> dict:
    """Convierte mappings Python a JSON-safe. None → null."""
    out = {}
    for k, v in mapping.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[str(k)] = None
        else:
            out[str(k)] = int(v)
    return out


def export_edge_artifacts(
    model: xgb.Booster,
    X_train_scaled: pd.DataFrame,
    scaler,
    encoder,
    num_imputer,
    cat_imputer,
    ordinal_imputer,
    threshold: float = BEST_THRESHOLD,
) -> None:
    """
    Exporta todos los artefactos necesarios para inferencia en Rust/WASM:
      - xgb_model.json
      - inference_contract.json
      - parity_sample.json
      - expected_proba.npy / expected_margin.npy
    """
    EDGE_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # 1) Modelo XGBoost portable
    # ─────────────────────────────────────────────────────────
    model.save_model(EDGE_DIR / "xgb_model.json")
    print(f"[export_edge] Modelo guardado: {EDGE_DIR / 'xgb_model.json'}")

    # ─────────────────────────────────────────────────────────
    # 2) Feature order
    # ─────────────────────────────────────────────────────────
    feature_order = list(X_train_scaled.columns)

    # ─────────────────────────────────────────────────────────
    # 3) Imputadores
    # ─────────────────────────────────────────────────────────
    num_cols_imputer = list(getattr(num_imputer, "feature_names_in_", []))
    cat_cols_imputer = list(getattr(cat_imputer, "feature_names_in_", []))

    if len(num_cols_imputer) == 0:
        raise ValueError("num_imputer no tiene feature_names_in_.")

    if len(cat_cols_imputer) == 0:
        cat_cols_imputer = ["Gender", "Education_Level", "Marital_Status",
                            "Income_Category", "Card_Category"]

    # ─────────────────────────────────────────────────────────
    # 4) OneHotEncoder
    # ─────────────────────────────────────────────────────────
    if hasattr(encoder, "feature_names_in_"):
        onehot_cols = list(encoder.feature_names_in_)
    else:
        onehot_cols = ["Gender", "Marital_Status", "Card_Category"]

    if hasattr(encoder, "get_feature_names_out"):
        onehot_encoded_names = list(encoder.get_feature_names_out(onehot_cols))
    else:
        onehot_encoded_names = list(encoder.get_feature_names(onehot_cols))

    onehot_categories = {
        str(col): [str(v) for v in cats]
        for col, cats in zip(onehot_cols, encoder.categories_)
    }

    category_pairs = []
    idx = 0
    for col, cats in zip(onehot_cols, encoder.categories_):
        for cat in cats:
            category_pairs.append({
                "column": str(col),
                "category": str(cat),
                "feature": onehot_encoded_names[idx],
            })
            idx += 1

    # ─────────────────────────────────────────────────────────
    # 5) Medianas ordinales
    # ─────────────────────────────────────────────────────────
    ord_cols_imputer = list(
        getattr(
            ordinal_imputer,
            "feature_names_in_",
            ["Income_Category_ordinal", "Education_Level_ordinal"],
        )
    )
    ordinal_stats = _to_list(ordinal_imputer.statistics_)

    income_median = float(
        ordinal_stats[ord_cols_imputer.index("Income_Category_ordinal")]
    )
    education_median = float(
        ordinal_stats[ord_cols_imputer.index("Education_Level_ordinal")]
    )

    # ─────────────────────────────────────────────────────────
    # 6) Scaler
    # ─────────────────────────────────────────────────────────
    scaler_mean = _to_list(scaler.mean_)
    scaler_scale_raw = _to_list(scaler.scale_)
    scaler_scale = [
        float(s) if abs(float(s)) > 1e-12 else 1.0
        for s in scaler_scale_raw
    ]

    # ─────────────────────────────────────────────────────────
    # 7) Contrato de inferencia
    # ─────────────────────────────────────────────────────────
    contract = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "best_threshold": float(threshold),
        "seed": int(SEED),
        "feature_order": feature_order,
        "raw_input": {
            "numeric_columns": num_cols_imputer,
            "categorical_columns": cat_cols_imputer,
        },
        "numeric_imputer": {
            "strategy": "mean",
            "columns": num_cols_imputer,
            "values": _to_list(num_imputer.statistics_),
        },
        "categorical_imputer": {
            "strategy": "most_frequent",
            "columns": cat_cols_imputer,
            "values": [str(v) for v in cat_imputer.statistics_],
        },
        "ordinal": {
            "income_column": "Income_Category",
            "education_column": "Education_Level",
            "income_mapping": _clean_mapping(INCOME_ORDER),
            "education_mapping": _clean_mapping(EDUCATION_ORDER),
            "income_ordinal_feature": "Income_Category_ordinal",
            "education_ordinal_feature": "Education_Level_ordinal",
            "income_unknown_feature": "Income_Category_is_unknown",
            "education_unknown_feature": "Education_Level_is_unknown",
            "income_median": income_median,
            "education_median": education_median,
        },
        "onehot": {
            "columns": onehot_cols,
            "categories": onehot_categories,
            "encoded_feature_names": onehot_encoded_names,
            "category_feature_pairs": category_pairs,
            "handle_unknown": "ignore",
        },
        "scaler": {
            "method": "standard",
            "columns": feature_order,
            "mean": scaler_mean,
            "scale": scaler_scale,
        },
    }

    # Leer base_score del JSON de XGBoost
    model_json_obj = json.loads((EDGE_DIR / "xgb_model.json").read_text())
    base_score_param = (
        model_json_obj
        .get("learner", {})
        .get("learner_model_param", {})
        .get("base_score", None)
    )

    contract["xgboost"] = {
        "base_score_param": base_score_param,
        "base_raw_hint": 0.0,
        "note": "Validar base_score con parity_sample.json.",
    }

    with open(EDGE_DIR / "inference_contract.json", "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)

    print(f"[export_edge] Contrato guardado: {EDGE_DIR / 'inference_contract.json'}")

    # ─────────────────────────────────────────────────────────
    # 8) Parity sample (Python ↔ Rust)
    # ─────────────────────────────────────────────────────────
    n_sample = min(PARITY_SAMPLE_SIZE, len(X_train_scaled))
    X_sample = X_train_scaled.iloc[:n_sample].copy()

    d_sample = xgb.DMatrix(X_sample)
    expected_prob = model.predict(d_sample)
    expected_margin = model.predict(d_sample, output_margin=True)

    parity = {
        "description": "Features ya escaladas. Rust debe predecir probabilidad y margin aproximadamente iguales.",
        "feature_order": feature_order,
        "features": X_sample.to_numpy(dtype=np.float64).tolist(),
        "expected_proba": expected_prob.tolist(),
        "expected_margin": expected_margin.tolist(),
    }

    with open(EDGE_DIR / "parity_sample.json", "w", encoding="utf-8") as f:
        json.dump(parity, f)

    np.save(EDGE_DIR / "expected_proba.npy", expected_prob.astype(np.float32))
    np.save(EDGE_DIR / "expected_margin.npy", expected_margin.astype(np.float32))

    # ─────────────────────────────────────────────────────────
    # 9) Validaciones
    # ─────────────────────────────────────────────────────────
    if model.feature_names is not None:
        assert list(model.feature_names) == feature_order

    assert len(contract["scaler"]["mean"]) == len(feature_order)
    assert len(contract["scaler"]["scale"]) == len(feature_order)
    assert X_sample.shape[1] == len(feature_order)

    print(f"[export_edge] ✅ Export completo. Features: {len(feature_order)}")
    print(f"[export_edge] Sample shape: {X_sample.shape}")
    print(f"[export_edge] Expected proba[:5]: {expected_prob[:5]}")
    print(f"[export_edge] Expected margin[:5]: {expected_margin[:5]}")

    files = sorted([p.name for p in EDGE_DIR.glob("*")])
    print(f"[export_edge] Archivos generados: {files}")