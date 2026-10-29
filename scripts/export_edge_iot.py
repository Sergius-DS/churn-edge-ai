#!/usr/bin/env python
"""
scripts/export_edge_iot.py
──────────────────────────
Exporta el modelo Edge IoT y su contrato ENRIQUECIDO (schema v2).

Genera:
    edge/xgb_model_edge_iot.json
    edge/inference_contract_edge_iot.json       ← Schema rico para Rust
    edge/sample_scaled_edge_iot.npy
    edge/expected_proba_edge_iot.npy
    edge/parity_sample_edge_iot.json            ← Para demo.js
    edge/raw_parity_case_iot.json               ← Para tests/parity_raw_iot.rs
"""

import json
import hashlib
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    EDGE_DIR,
    EDGE_IOT_ARTIFACTS_DIR,
    EDGE_IOT_MODEL_JSON,
    EDGE_IOT_CONTRACT_JSON,
    IOT_FEATURES,
    FORBIDDEN_FIELDS,
    IOT_FEATURE_BOUNDS,
    INCOME_ORDER,
    EDUCATION_ORDER,
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_raw_case(raw: dict, contract: dict) -> np.ndarray:
    """
    Replica EXACTAMENTE el preprocessing de Rust (preprocessing.rs).
    Se usa para generar raw_parity_case_iot.json.
    """
    # 1) Imputación numérica (media)
    numeric = {}
    for col, fill in zip(contract["numeric_imputer"]["columns"],
                         contract["numeric_imputer"]["values"]):
        v = raw.get(col)
        numeric[col] = float(v) if v is not None else float(fill)

    # 2) Imputación categórica (moda)
    categorical = {}
    for col, fill in zip(contract["categorical_imputer"]["columns"],
                         contract["categorical_imputer"]["values"]):
        v = raw.get(col)
        categorical[col] = str(v) if v is not None else str(fill)

    # 3) Ordinal + flags is_unknown + mediana
    ordinal = {}
    o = contract["ordinal"]
    for src_col, map_key, ord_feat, unk_feat, med_key in [
        (o["income_column"], "income_mapping",
         o["income_ordinal_feature"], o["income_unknown_feature"], "income_median"),
        (o["education_column"], "education_mapping",
         o["education_ordinal_feature"], o["education_unknown_feature"], "education_median"),
    ]:
        mapped = o[map_key].get(categorical.get(src_col))
        if mapped is None:
            ordinal[ord_feat] = float(o[med_key])
            ordinal[unk_feat] = 1.0
        else:
            ordinal[ord_feat] = float(mapped)
            ordinal[unk_feat] = 0.0

    # 4) One-hot
    onehot = {}
    for pair in contract["onehot"]["category_feature_pairs"]:
        onehot[pair["feature"]] = (
            1.0 if categorical.get(pair["column"]) == pair["category"] else 0.0
        )

    # 5) Ensamblar en el orden exacto del modelo
    feats = []
    for name in contract["feature_order"]:
        if name in numeric:
            feats.append(numeric[name])
        elif name in ordinal:
            feats.append(ordinal[name])
        elif name in onehot:
            feats.append(onehot[name])
        else:
            feats.append(0.0)

    # 6) StandardScaler
    mean = np.array(contract["scaler"]["mean"], dtype=np.float64)
    scale = np.array(contract["scaler"]["scale"], dtype=np.float64)
    return (np.array(feats, dtype=np.float64) - mean) / scale


def main() -> None:
    print("=" * 70)
    print("📦 Exportando modelo Edge IoT (contrato enriquecido v2)")
    print("=" * 70)

    EDGE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Rutas de artefactos ──
    model_path = EDGE_IOT_ARTIFACTS_DIR / "xgb_model.joblib"
    config_path = EDGE_IOT_ARTIFACTS_DIR / "model_config.joblib"
    scaler_path = EDGE_IOT_ARTIFACTS_DIR / "scaler.joblib"
    sample_path = EDGE_IOT_ARTIFACTS_DIR / "X_test_scaled_sample.csv"
    num_imputer_path = EDGE_IOT_ARTIFACTS_DIR / "num_imputer.joblib"
    cat_imputer_path = EDGE_IOT_ARTIFACTS_DIR / "cat_imputer.joblib"
    ordinal_imputer_path = EDGE_IOT_ARTIFACTS_DIR / "ordinal_imputer.joblib"
    encoder_path = EDGE_IOT_ARTIFACTS_DIR / "onehot_encoder.joblib"

    # ── Verificar que todos los artefactos existen ──
    all_paths = [
        model_path, config_path, scaler_path, sample_path,
        num_imputer_path, cat_imputer_path,
        ordinal_imputer_path, encoder_path,
    ]
    for p in all_paths:
        assert p.exists(), f"❌ Falta artefacto: {p}"

    # ── Cargar todo ──
    model = joblib.load(model_path)
    config = joblib.load(config_path)
    scaler = joblib.load(scaler_path)
    num_imputer = joblib.load(num_imputer_path)
    cat_imputer = joblib.load(cat_imputer_path)
    ordinal_imputer = joblib.load(ordinal_imputer_path)
    encoder = joblib.load(encoder_path)

    feature_order = config.get("feature_order", [])
    threshold = config.get("best_threshold", 0.5)

    print(f"[export_edge_iot] Modelo cargado: {model.num_boosted_rounds()} árboles")
    print(f"[export_edge_iot] Features: {len(feature_order)}")
    print(f"[export_edge_iot] Threshold: {threshold}")

    # ── Exportar modelo XGBoost JSON ──
    model.save_model(EDGE_IOT_MODEL_JSON)
    print(f"[export_edge_iot] ✅ Modelo → {EDGE_IOT_MODEL_JSON}")
    model_sha256 = sha256_file(EDGE_IOT_MODEL_JSON)

    # ── Muestra de paridad (escalada) ──
    sample = pd.read_csv(sample_path)
    if feature_order:
        sample = sample[feature_order]
    else:
        feature_order = list(sample.columns)

    X_sample = sample.values.astype(np.float32)

    dmat_sample = xgb.DMatrix(X_sample, feature_names=feature_order)
    expected_proba = model.predict(dmat_sample)

    np.save(EDGE_DIR / "sample_scaled_edge_iot.npy", X_sample)
    np.save(EDGE_DIR / "expected_proba_edge_iot.npy", expected_proba)
    print(f"[export_edge_iot] ✅ Paridad escalada: {X_sample.shape[0]} muestras")

    # ── parity_sample_edge_iot.json (para demo.js) ──
    parity_sample = {
        "features": X_sample.tolist(),
        "expected_proba": expected_proba.tolist(),
        "feature_order": feature_order,
        "n_samples": int(X_sample.shape[0]),
        "n_features": int(X_sample.shape[1]),
    }
    parity_sample_path = EDGE_DIR / "parity_sample_edge_iot.json"
    parity_sample_path.write_text(
        json.dumps(parity_sample, indent=2), encoding="utf-8"
    )
    print(f"[export_edge_iot] ✅ Parity sample JSON → {parity_sample_path}")

    # ── Extraer info de los imputadores ──
    raw_numeric_columns = list(
        getattr(num_imputer, "feature_names_in_", [])
    )
    raw_categorical_columns = list(
        getattr(cat_imputer, "feature_names_in_", [])
    )

    numeric_imputer_cfg = {
        "strategy": "mean",
        "columns": raw_numeric_columns,
        "values": num_imputer.statistics_.tolist(),
    }

    categorical_imputer_cfg = {
        "strategy": "most_frequent",
        "columns": raw_categorical_columns,
        "values": [str(v) for v in cat_imputer.statistics_],
    }

    # ── Ordinal ──
    ordinal_medians = ordinal_imputer.statistics_.tolist()
    ordinal_cfg = {
        "income_column": "Income_Category",
        "education_column": "Education_Level",
        "income_mapping": INCOME_ORDER,
        "education_mapping": EDUCATION_ORDER,
        "income_ordinal_feature": "Income_Category_ordinal",
        "education_ordinal_feature": "Education_Level_ordinal",
        "income_unknown_feature": "Income_Category_is_unknown",
        "education_unknown_feature": "Education_Level_is_unknown",
        "income_median": float(ordinal_medians[0]) if len(ordinal_medians) > 0 else 2.0,
        "education_median": float(ordinal_medians[1]) if len(ordinal_medians) > 1 else 3.0,
    }

    # ── OneHot (robusto, desde categories_) ──
    categories_dict = {}
    category_feature_pairs = []
    encoded_feature_names = []

    if hasattr(encoder, "categories_"):
        input_cols = list(getattr(encoder, "feature_names_in_", []))
        for col, cats in zip(input_cols, encoder.categories_):
            categories_dict[col] = [str(c) for c in cats]

        for col, cats in zip(input_cols, encoder.categories_):
            for cat in cats:
                feat_name = f"{col}_{cat}"
                encoded_feature_names.append(feat_name)
                category_feature_pairs.append({
                    "column": col,
                    "category": str(cat),
                    "feature": feat_name,
                })

    onehot_cfg = {
        "columns": list(categories_dict.keys()),
        "categories": categories_dict,
        "encoded_feature_names": encoded_feature_names,
        "category_feature_pairs": category_feature_pairs,
        "handle_unknown": "ignore",
    }

    # ── Scaler con nombres ──
    scaler_cfg = {
        "method": "standard",
        "columns": feature_order,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    # ── IOT bounds serializables ──
    iot_bounds_serializable = {
        k: [float(lo), float(hi)]
        for k, (lo, hi) in IOT_FEATURE_BOUNDS.items()
    }

    # ── Contrato completo ──
    contract = {
        "version": 1,
        "model_type": "edge_iot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "Attrition_Flag",
        "best_threshold": float(threshold),
        "seed": 44,
        "feature_order": feature_order,
        "n_features": len(feature_order),

        # Metadata IoT
        "iot_features": IOT_FEATURES,
        "iot_bounds": iot_bounds_serializable,
        "forbidden_fields": FORBIDDEN_FIELDS,
        "consent_required": True,

        # Input raw
        "raw_input": {
            "numeric_columns": raw_numeric_columns,
            "categorical_columns": raw_categorical_columns,
        },

        # Preprocessing
        "numeric_imputer": numeric_imputer_cfg,
        "categorical_imputer": categorical_imputer_cfg,
        "ordinal": ordinal_cfg,
        "onehot": onehot_cfg,
        "scaler": scaler_cfg,

        # XGBoost metadata
        "xgboost": {
            "base_score_param": "[5E-1]",
            "base_raw_hint": 0.0,
            "note": "Validar base_score con parity_sample_edge_iot.json.",
        },

        # Output
        "output": {
            "type": "churn_probability",
            "range": [0.0, 1.0],
        },

        # Privacy
        "privacy_mode": "edge_local_inference",
        "model_sha256": model_sha256,
    }

    EDGE_IOT_CONTRACT_JSON.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[export_edge_iot] ✅ Contrato v2 → {EDGE_IOT_CONTRACT_JSON}")

    # ── Validación de consistencia ──
    n_scaler = len(scaler_cfg["mean"])
    n_features = len(feature_order)
    n_onehot = len(encoded_feature_names)
    assert n_scaler == n_features, (
        f"Scaler tiene {n_scaler} valores pero feature_order tiene {n_features}"
    )
    print(f"[export_edge_iot] ✅ Consistencia: scaler={n_scaler}, "
          f"features={n_features}, onehot={n_onehot}")

    # ── raw_parity_case_iot.json (para tests/parity_raw_iot.rs) ──
    RAW_IOT_INPUT = {
        "Customer_Age": 45,
        "Gender": "F",
        "Dependent_count": 3,
        "Education_Level": "Graduate",
        "Marital_Status": "Married",
        "Income_Category": "$60K - $80K",
        "Card_Category": "Blue",
        "Months_on_book": 39,
        "Total_Relationship_Count": 5,
        "Months_Inactive_12_mon": 2,
        "Contacts_Count_12_mon": 3,
        "Credit_Limit": 12691.0,
        "Total_Revolving_Bal": 777,
        "Avg_Open_To_Buy": 11914.0,
        "Total_Amt_Chng_Q4_Q1": 0.8,
        "Total_Trans_Amt": 3500,
        "Total_Trans_Ct": 45,
        "Total_Ct_Chng_Q4_Q1": 0.69,
        "Avg_Utilization_Ratio": 0.22,
        # 6 features IoT (valores dentro de bounds)
        "app_open_count_30d": 76,
        "contactless_usage_count_30d": 58,
        "payment_flow_abandon_count_30d": 5,
        "nfc_failed_attempts_30d": 3,
        "app_crash_count_30d": 2,
        "pos_latency_p95_ms": 1500,
    }

    scaled_raw = preprocess_raw_case(RAW_IOT_INPUT, contract)
    dmat_raw = xgb.DMatrix(
        scaled_raw.reshape(1, -1),
        feature_names=feature_order,
    )
    raw_proba = float(model.predict(dmat_raw)[0])

    raw_parity_case = {
        "description": (
            "Golden case RAW→proba para Edge IoT. "
            "Rust debe replicar el preprocessing completo."
        ),
        "raw": RAW_IOT_INPUT,
        "expected_proba": raw_proba,
        "expected_proba_pct": raw_proba * 100.0,
    }

    raw_parity_path = EDGE_DIR / "raw_parity_case_iot.json"
    raw_parity_path.write_text(
        json.dumps(raw_parity_case, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[export_edge_iot] ✅ Raw parity case → {raw_parity_path}")
    print(f"[export_edge_iot]    Proba esperada: {raw_proba * 100:.4f}%")

    print("\n" + "=" * 70)
    print("🎉 Exportación Edge IoT completada")
    print("=" * 70)
    print(f"   Modelo:     {EDGE_IOT_MODEL_JSON}")
    print(f"   Contrato:   {EDGE_IOT_CONTRACT_JSON}")
    print(f"   Features:   {n_features}")
    print(f"   IoT:        {len(IOT_FEATURES)}")
    print(f"   Threshold:  {threshold}")
    print(f"   SHA256:     {model_sha256[:16]}...")


if __name__ == "__main__":
    main()