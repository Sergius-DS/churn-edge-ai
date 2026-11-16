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
from pathlib import Path

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
    SEED,
    SYNTHETIC_IOT_CSV,
)
from src.train import run_fingerprint


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_xgboost_metadata(model_json_path) -> dict:
    """Lee base_score del JSON de XGBoost. Falla si no existe."""
    model_obj = json.loads(model_json_path.read_text(encoding="utf-8"))
    base_score = (
        model_obj.get("learner", {})
        .get("learner_model_param", {})
        .get("base_score")
    )
    if base_score is None:
        raise ValueError(
            "base_score ausente en el JSON del modelo. "
            "No se puede hardcodear sin romper reproducibilidad."
        )
    return {
        "base_score_param": base_score,
        "base_raw_hint": 0.0,
        "note": "Leído del learner_model_param del modelo exportado.",
    }


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
            raise KeyError(
                f"Feature '{name}' no cubierta por numeric, ordinal ni onehot. "
                f"Revisar feature_order vs contrato."
            )

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

    if not feature_order:
        raise ValueError("model_config.joblib sin feature_order: no se puede verificar linaje")

    print(f"[export_edge_iot] Modelo cargado: {model.num_boosted_rounds()} árboles")
    print(f"[export_edge_iot] Features: {len(feature_order)}")
    print(f"[export_edge_iot] Threshold: {threshold}")

    # ── Gate de procedencia (Fix C1): verificar linaje del sample ──
    fp = run_fingerprint(feature_order, scaler, threshold)
    prov_path = EDGE_IOT_ARTIFACTS_DIR / "sample_provenance.json"
    if not prov_path.exists():
        raise FileNotFoundError(
            f"Falta {prov_path.name}: la muestra de paridad no tiene procedencia. "
            "Re-ejecutar scripts/train_edge_iot_model.py."
        )
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if prov.get("run_fingerprint") != fp:
        raise ValueError(
            "X_test_scaled_sample.csv pertenece a OTRO linaje: "
            f"sample={str(prov.get('run_fingerprint'))[:16]} vs run={fp[:16]}. "
            "Re-ejecutar train_edge_iot_model.py."
        )
    print(f"[export_edge_iot] ✅ Linaje del sample verificado ({fp[:16]})")
    
    # ── Publicar métricas: el umbral del contrato necesita su justificación trackeada ──
    # model_artifacts/ está ignorado por git. Un metrics.json que vive ahí no es
    # evidencia: es un archivo local. Se publica en edge/, junto al contrato.
    metrics_src = EDGE_IOT_ARTIFACTS_DIR / "metrics.json"
    if not metrics_src.exists():
        raise FileNotFoundError(
            "Falta metrics.json: el contrato declararía un best_threshold sin "
            "justificación registrada. Re-ejecutar train_edge_iot_model.py.")
    metrics_obj = json.loads(metrics_src.read_text(encoding="utf-8"))

    # Tercera verificación de huella: las métricas deben ser del mismo run.
    if metrics_obj.get("run_fingerprint") != fp:
        raise ValueError(
            "metrics.json pertenece a OTRO linaje: "
            f"metrics={str(metrics_obj.get('run_fingerprint'))[:16]} vs run={fp[:16]}. "
            "Re-ejecutar train_edge_iot_model.py.")

    # Y del mismo dataset. Dos runs con config idéntica comparten huella;
    # el hash del dato es lo que los distingue.
    if metrics_obj.get("training_dataset_sha256") != prov["training_dataset_sha256"]:
        raise ValueError(
            "metrics.json y sample_provenance.json apuntan a datasets distintos. "
            "Re-ejecutar train_edge_iot_model.py.")

    metrics_pub = EDGE_DIR / "metrics_edge_iot.json"
    metrics_pub.write_text(
        json.dumps(metrics_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[export_edge_iot] ✅ Métricas publicadas → {metrics_pub.name} "
          f"(test_auc={metrics_obj['test_auc']:.6f}, thr={metrics_obj['threshold']})")

    # ── Exportar modelo XGBoost JSON ──
    model.save_model(EDGE_IOT_MODEL_JSON)
    print(f"[export_edge_iot] ✅ Modelo → {EDGE_IOT_MODEL_JSON}")
    model_sha256 = sha256_file(EDGE_IOT_MODEL_JSON)

    # ── Muestra de paridad (escalada) ──
    sample = pd.read_csv(sample_path)
    sample = sample[feature_order]

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
        "dtype": "float32",
        "parity_tolerance": {"proba_abs": 1e-6},
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

    # ── OneHot (nombres desde el encoder, no re-derivados) ──
    categories_dict = {}
    category_feature_pairs = []

    if hasattr(encoder, "categories_"):
        input_cols = list(getattr(encoder, "feature_names_in_", []))
        for col, cats in zip(input_cols, encoder.categories_):
            categories_dict[col] = [str(c) for c in cats]
        
        # Fix #7: usar get_feature_names_out(), no f-string manual
        encoded_feature_names = list(encoder.get_feature_names_out(input_cols))
        assert set(encoded_feature_names) <= set(feature_order), (
            f"OneHot produce nombres fuera de feature_order: "
            f"{set(encoded_feature_names) - set(feature_order)}"
        )
        
        # Reconstruir pairs usando los nombres reales del encoder
        for feat_name, col, cat in zip(
            encoded_feature_names,
            (c for c in input_cols for _ in categories_dict[c]),
            (cat for c in input_cols for cat in categories_dict[c]),
        ):
            category_feature_pairs.append({
                "column": col,
                "category": str(cat),
                "feature": feat_name,
            })
    else:
        encoded_feature_names = []

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

    # ── Contrato completo (Fix C2: sin created_at para byte-determinismo) ──
    contract = {
        "version": 2,                                    # Fix #3: era 1
        "model_type": "edge_iot",
        "target": "Attrition_Flag",
        "best_threshold": float(threshold),
        "seed": int(SEED),                               # Fix #4: era 44 hardcodeado
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
        "xgboost": _extract_xgboost_metadata(EDGE_IOT_MODEL_JSON),

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

    # ── Checkpoint 3: la réplica del contrato debe reproducir el pipeline sklearn ──
    raw_sample_path = EDGE_IOT_ARTIFACTS_DIR / "X_test_raw_sample.csv"
    if not raw_sample_path.exists():
        raise FileNotFoundError(
            f"Falta {raw_sample_path.name}: sin él no se puede verificar que "
            "preprocess_raw_case replica el pipeline de entrenamiento. "
            "Re-ejecutar train_edge_iot_model.py.")
    raw_sample = pd.read_csv(raw_sample_path)
    
    # FIX: Alineación de longitud (la huella no certifica que ambos CSV tengan el mismo N)
    if len(raw_sample) != len(X_sample):
        raise ValueError(
            f"raw/scaled desalineados: raw={len(raw_sample)} vs scaled={len(X_sample)}. "
            "Re-ejecutar train_edge_iot_model.py."
        )

    n_chk = min(10, len(raw_sample))
    worst = 0.0
    for k in range(n_chk):
        row = {c: (None if pd.isna(v) else v)
               for c, v in raw_sample.iloc[k].to_dict().items()}
        rep = preprocess_raw_case(row, contract).astype(np.float32)
        worst = max(worst, float(np.max(np.abs(rep - X_sample[k]))))
    if worst > 1e-5:
        raise ValueError(
            f"preprocess_raw_case NO replica el pipeline sklearn: max|Δ|={worst:.3e}. "
            "El golden sería auto-consistente con Rust pero distinto del entrenamiento.")
    print(f"[export_edge_iot] ✅ Réplica≡sklearn ({n_chk} filas, max|Δ|={worst:.3e})")

    # ── raw_parity_case_iot.json (para tests/parity_raw_iot.rs) ──
    # Fix #1: fila REAL del dataset IoT, no un cliente ensamblado a mano.
    # Fix #2: doble checkpoint (expected_scaled + expected_proba).
    # Fix C3: alinear dtypes a f32 para consistencia con XGBoost
    iot_df = pd.read_csv(SYNTHETIC_IOT_CSV)
    golden_row = iot_df.iloc[len(iot_df) // 2]  # determinista: fila central

    # Extraer solo las columnas que el contrato espera como input raw
    raw_cols = raw_numeric_columns + raw_categorical_columns
    RAW_IOT_INPUT = {}
    for c in raw_cols:
        val = golden_row[c]
        if hasattr(val, "item"):
            RAW_IOT_INPUT[c] = val.item()
        else:
            RAW_IOT_INPUT[c] = val

    # El golden DEBE cumplir las identidades algebraicas del contrato.
    assert abs(
        RAW_IOT_INPUT["Avg_Open_To_Buy"]
        - (RAW_IOT_INPUT["Credit_Limit"] - RAW_IOT_INPUT["Total_Revolving_Bal"])
    ) < 0.01, (
        f"Golden viola INV-1: Open_To_Buy={RAW_IOT_INPUT['Avg_Open_To_Buy']} "
        f"pero Limit-Revolving="
        f"{RAW_IOT_INPUT['Credit_Limit'] - RAW_IOT_INPUT['Total_Revolving_Bal']}"
    )
    rebuilt_util = round(
        RAW_IOT_INPUT["Total_Revolving_Bal"] / RAW_IOT_INPUT["Credit_Limit"], 3
    )
    assert abs(RAW_IOT_INPUT["Avg_Utilization_Ratio"] - rebuilt_util) < 0.0011, (
        f"Golden viola INV-2: Utilización={RAW_IOT_INPUT['Avg_Utilization_Ratio']} "
        f"pero Reconstruida={rebuilt_util}"
    )

    scaled_raw = preprocess_raw_case(RAW_IOT_INPUT, contract)
    scaled_raw_f32 = scaled_raw.astype(np.float32)     # ← Fix C3: alinear con XGBoost
    dmat_raw = xgb.DMatrix(
        scaled_raw_f32.reshape(1, -1),                 # ← f32
        feature_names=feature_order,
    )
    raw_proba = float(model.predict(dmat_raw)[0])

    raw_parity_case = {
        "description": (
            "Golden RAW→scaled→proba. Fila real del curated+IoT. "
            "Rust debe replicar el preprocessing completo."
        ),
        "source_clientnum": int(golden_row["CLIENTNUM"]),
        "raw": RAW_IOT_INPUT,
        "expected_scaled": scaled_raw_f32.tolist(),    # ← f32 (consistente)
        "scaled_dtype": "float32",                     # ← Fix C3: declarado
        "parity_tolerance": {                          # ← Fix C3: tolerancia explícita
            "scaled_abs": 1e-5,
            "proba_abs": 1e-6,
        },
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

    # ── Export manifest (Fix C4: procedencia volcada fuera del contrato) ──
    export_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "export_script_sha256": sha256_file(Path(__file__)),
        "run_fingerprint": fp,
        "training_dataset_sha256": prov["training_dataset_sha256"],
        "model_sha256": model_sha256,
        "contract_sha256": sha256_file(EDGE_IOT_CONTRACT_JSON),
        "parity_sample_sha256": sha256_file(parity_sample_path),
        "raw_parity_sha256": sha256_file(raw_parity_path),
        "metrics_sha256": sha256_file(metrics_pub),
        "volatile_fields_relocated": ["created_at"],
    }
    manifest_path = EDGE_DIR / "export_manifest_edge_iot.json"
    manifest_path.write_text(
        json.dumps(export_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[export_edge_iot] ✅ Manifest → {manifest_path}")

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