#!/usr/bin/env python
"""
scripts/export_edge_lite.py
────────────────────────────
Exporta el modelo EDGE a JSON + contrato WASM.
Genera parity_sample_edge.json para validar Python ↔ Rust.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timezone

from src.config import (
    ARTIFACTS_DIR, EDGE_DIR, SEED, TARGET,
    BEST_THRESHOLD, INCOME_ORDER, EDUCATION_ORDER,
    PARITY_SAMPLE_SIZE, PARITY_TOLERANCE,
)


def _to_list(x):
    return np.asarray(x).tolist()


def _clean_mapping(mapping: dict) -> dict:
    out = {}
    for k, v in mapping.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[str(k)] = None
        else:
            out[str(k)] = int(v)
    return out


def main():
    print("=" * 60)
    print("📦 CHURN EDGE AI — Export Edge Lite (WASM/Rust)")
    print("=" * 60)

    # ─── Verificar artefactos ───
    edge_model_path = ARTIFACTS_DIR / "xgb_model_edge.joblib"
    assert edge_model_path.exists(), (
        "Primero ejecuta: python scripts/train_edge_model.py"
    )

    # ─── Cargar artefactos ───
    print("\n📂 Cargando artefactos edge...")
    edge_model = joblib.load(edge_model_path)
    edge_config = joblib.load(ARTIFACTS_DIR / "model_config_edge.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    encoder = joblib.load(ARTIFACTS_DIR / "onehot_encoder.joblib")
    num_imputer = joblib.load(ARTIFACTS_DIR / "num_imputer.joblib")
    cat_imputer = joblib.load(ARTIFACTS_DIR / "cat_imputer.joblib")
    ordinal_imputer = joblib.load(ARTIFACTS_DIR / "ordinal_imputer.joblib")

    edge_features = edge_config["feature_names"]
    threshold = edge_config["best_threshold"]

    print(f"[export_lite] Features edge: {len(edge_features)}")
    print(f"[export_lite] Features: {edge_features}")

    # ─── Cargar muestra real ───
    sample_path = ARTIFACTS_DIR / "X_train_edge_sample.csv"
    assert sample_path.exists(), (
        f"No se encontró {sample_path}. Ejecuta train_edge_model.py primero."
    )
    X_sample = pd.read_csv(sample_path)
    print(f"[export_lite] ✅ Muestra cargada: {sample_path}")
    print(f"[export_lite]    Shape: {X_sample.shape}")

    # ─── Guardar modelo JSON ───
    EDGE_DIR.mkdir(parents=True, exist_ok=True)
    edge_json_path = EDGE_DIR / "xgb_model_edge.json"
    edge_model.save_model(edge_json_path)
    print(f"[export_lite] Modelo guardado: {edge_json_path}")

    # ─── Feature order completo (para mapear scaler) ───
    full_model = joblib.load(ARTIFACTS_DIR / "xgb_model.joblib")
    full_feature_order = list(full_model.feature_names) if full_model.feature_names else []
    full_indices = {name: i for i, name in enumerate(full_feature_order)}

    # Extraer solo los parámetros del scaler para las features edge
    edge_scaler_indices = [full_indices[f] for f in edge_features if f in full_indices]
    edge_scaler_mean = _to_list(scaler.mean_[edge_scaler_indices])
    edge_scaler_scale_raw = _to_list(scaler.scale_[edge_scaler_indices])
    edge_scaler_scale = [
        float(s) if abs(float(s)) > 1e-12 else 1.0
        for s in edge_scaler_scale_raw
    ]

    # ─── Imputadores (solo columnas relevantes para edge) ───
    num_cols_imputer = list(getattr(num_imputer, "feature_names_in_", []))
    cat_cols_imputer = list(getattr(cat_imputer, "feature_names_in_", []))

    # Filtrar imputadores numéricos a los que usan las features edge
    edge_num_cols = [c for c in num_cols_imputer if c in edge_features]
    edge_num_indices = [num_cols_imputer.index(c) for c in edge_num_cols]
    edge_num_values = _to_list(num_imputer.statistics_[edge_num_indices])

    # ─── OneHot (solo categorías relevantes) ───
    if hasattr(encoder, "feature_names_in_"):
        onehot_cols = list(encoder.feature_names_in_)
    else:
        onehot_cols = ["Gender", "Marital_Status", "Card_Category"]

    if hasattr(encoder, "get_feature_names_out"):
        onehot_encoded_names = list(encoder.get_feature_names_out(onehot_cols))
    else:
        onehot_encoded_names = list(encoder.get_feature_names(onehot_cols))

    # Filtrar solo las features one-hot que están en edge
    edge_onehot_pairs = []
    idx = 0
    for col, cats in zip(onehot_cols, encoder.categories_):
        for cat in cats:
            feat_name = onehot_encoded_names[idx]
            if feat_name in edge_features:
                edge_onehot_pairs.append({
                    "column": str(col),
                    "category": str(cat),
                    "feature": feat_name,
                })
            idx += 1

    edge_onehot_encoded = [p["feature"] for p in edge_onehot_pairs]
    edge_onehot_cols = list(set(p["column"] for p in edge_onehot_pairs))

    onehot_categories = {
        str(col): [str(v) for v in cats]
        for col, cats in zip(onehot_cols, encoder.categories_)
        if str(col) in edge_onehot_cols
    }

    # ─── Medianas ordinales ───
    ord_cols = list(getattr(
        ordinal_imputer, "feature_names_in_",
        ["Income_Category_ordinal", "Education_Level_ordinal"],
    ))
    ordinal_stats = _to_list(ordinal_imputer.statistics_)

    income_median = float(ordinal_stats[ord_cols.index("Income_Category_ordinal")])
    education_median = float(ordinal_stats[ord_cols.index("Education_Level_ordinal")])

    # ─── Contrato de inferencia EDGE ───
    contract = {
        "version": 1,
        "model_type": "edge_lite",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "best_threshold": float(threshold),
        "seed": int(SEED),
        "feature_order": edge_features,
        "n_features": len(edge_features),
        "raw_input": {
            "numeric_columns": edge_num_cols,
            "categorical_columns": cat_cols_imputer,
        },
        "numeric_imputer": {
            "strategy": "mean",
            "columns": edge_num_cols,
            "values": edge_num_values,
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
            "columns": edge_onehot_cols,
            "categories": onehot_categories,
            "encoded_feature_names": edge_onehot_encoded,
            "category_feature_pairs": edge_onehot_pairs,
            "handle_unknown": "ignore",
        },
        "scaler": {
            "method": "standard",
            "columns": edge_features,
            "mean": edge_scaler_mean,
            "scale": edge_scaler_scale,
        },
    }

    # Leer base_score del JSON edge
    model_json_obj = json.loads(edge_json_path.read_text())
    base_score_param = (
        model_json_obj
        .get("learner", {})
        .get("learner_model_param", {})
        .get("base_score", None)
    )
    contract["xgboost"] = {
        "base_score_param": base_score_param,
        "base_raw_hint": 0.0,
        "note": "Validar base_score con parity_sample_edge.json.",
    }

    contract_path = EDGE_DIR / "inference_contract_edge.json"
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, ensure_ascii=False)
    print(f"[export_lite] Contrato guardado: {contract_path}")

    # ─── Parity sample edge ───
    n_sample = min(PARITY_SAMPLE_SIZE, len(X_sample))
    X_parity = X_sample.iloc[:n_sample].copy()
    d_parity = xgb.DMatrix(X_parity, feature_names=edge_features)

    expected_prob = edge_model.predict(d_parity)
    expected_margin = edge_model.predict(d_parity, output_margin=True)

    parity = {
        "description": "Features edge escaladas. Rust debe predecir proba y margin iguales.",
        "model_type": "edge_lite",
        "feature_order": edge_features,
        "features": X_parity.to_numpy(dtype=np.float64).tolist(),
        "expected_proba": expected_prob.tolist(),
        "expected_margin": expected_margin.tolist(),
    }

    parity_path = EDGE_DIR / "parity_sample_edge.json"
    with open(parity_path, "w", encoding="utf-8") as f:
        json.dump(parity, f)

    np.save(EDGE_DIR / "sample_scaled_edge.npy", X_parity.to_numpy(dtype=np.float32))
    np.save(EDGE_DIR / "expected_proba_edge.npy", expected_prob.astype(np.float32))
    np.save(EDGE_DIR / "expected_margin_edge.npy", expected_margin.astype(np.float32))

    print(f"[export_lite] Parity sample: {parity_path}")
    print(f"[export_lite] Sample shape: {X_parity.shape}")
    print(f"[export_lite] Expected proba[:5]: {expected_prob[:5]}")

    # ─── Validaciones ───
    assert len(contract["scaler"]["mean"]) == len(edge_features)
    assert len(contract["scaler"]["scale"]) == len(edge_features)
    assert X_parity.shape[1] == len(edge_features)

    # ─── Validación de paridad inmediata ───
    print("\n🔍 Validación rápida de paridad edge...")
    model_check = xgb.Booster()
    model_check.load_model(str(edge_json_path))

    df_check = pd.DataFrame(
        parity["features"][:10],
        columns=edge_features,
    )
    dmat_check = xgb.DMatrix(df_check)
    check_proba = model_check.predict(dmat_check)
    expected_check = np.array(parity["expected_proba"][:10])

    try:
        np.testing.assert_allclose(check_proba, expected_check, atol=PARITY_TOLERANCE)
        max_diff = np.max(np.abs(check_proba - expected_check))
        print(f"[export_lite] ✅ Paridad edge verificada. Max diff: {max_diff:.2e}")
    except AssertionError as e:
        print(f"[export_lite] ❌ Paridad edge FALLÓ: {e}")
        sys.exit(1)

    # ─── Resumen de archivos ───
    edge_files = sorted([p.name for p in EDGE_DIR.glob("*edge*")])
    print(f"\n[export_lite] Archivos edge generados: {edge_files}")
    print("\n✅ Export Edge Lite completado")


if __name__ == "__main__":
    main()