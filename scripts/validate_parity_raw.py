#!/usr/bin/env python
"""
scripts/validate_parity_raw.py
──────────────────────────────
Paridad END-TO-END desde input RAW:
replica en Python el preprocessing de Rust (impute → ordinal →
onehot → scale) usando el contrato edge, y genera un golden file
que el test de Rust consume como regresión.

Uso: python scripts/validate_parity_raw.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import xgboost as xgb

from src.config import EDGE_DIR

# ─── El MISMO input crudo que tiene la demo en la captura ───
RAW_INPUT = {
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
}


def preprocess_raw(raw: dict, contract: dict) -> np.ndarray:
    """Replica EXACTAMENTE src/preprocessing.rs (Rust)."""
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
    contract = json.loads((EDGE_DIR / "inference_contract_edge.json").read_text())
    scaled = preprocess_raw(RAW_INPUT, contract)

    model = xgb.Booster()
    model.load_model(str(EDGE_DIR / "xgb_model_edge.json"))
    dmat = xgb.DMatrix(
        scaled.reshape(1, -1),
        feature_names=contract["feature_order"],
    )
    proba = float(model.predict(dmat)[0])

    # ─── Golden file para el test de Rust ───
    case = {
        "description": "Golden case RAW→proba generado por Python. Rust debe replicarlo.",
        "raw": RAW_INPUT,
        "expected_proba": proba,
        "expected_proba_pct": proba * 100.0,
    }

    out_path = EDGE_DIR / "raw_parity_case.json"
    payload = json.dumps(case, indent=2, ensure_ascii=False)
    out_path.write_text(payload, encoding="utf-8")

    print("=" * 60)
    print("🔍 PARIDAD END-TO-END (RAW → proba)")
    print("=" * 60)
    print(f"🐍 Python (edge):  {proba * 100:.4f}%")
    print(f"🦀 WASM (demo):    59.67%  ← lo que mostró tu navegador")
    print(f"📉 Diff:           {abs(proba * 100 - 59.67):.4f}%")
    print(f"💾 Golden file:    {out_path}")
    print("=" * 60)
    print("▶ Siguiente paso: cd edge/churn-wasm && cargo test --test parity_raw")


if __name__ == "__main__":
    main()