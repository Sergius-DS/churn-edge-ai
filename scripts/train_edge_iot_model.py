#!/usr/bin/env python
"""
scripts/train_edge_iot_model.py
───────────────────────────────
Entrena el modelo Edge IoT usando features bancarias + features IoT sintéticas.
"""

import hashlib
import pandas as pd
import json as _json

from src.config import (
    SEED,
    XGB_EDGE_PARAMS,
    EDGE_CV_NUM_BOOST_ROUND,
    EDGE_EARLY_STOPPING_ROUNDS,
    EDGE_IOT_ARTIFACTS_DIR,
    PARITY_SAMPLE_SIZE,
    SYNTHETIC_IOT_CSV,
)

from src.data_loading import load_and_prepare
from src.preprocessing import run_full_pipeline

from src.train import (
    train_with_cv,
    optimize_threshold,
    evaluate_on_test,
    get_feature_importance,
    save_artifacts,
    run_fingerprint,
)


def main() -> None:
    print("=" * 70)
    print("🤖 Entrenando modelo EDGE IoT")
    print("=" * 70)

    # 1. Cargar dataset con IoT
    X_train, X_test, y_train, y_test = load_and_prepare(with_iot=True)

    # 2. Preprocessing completo
    artifacts = run_full_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    X_train_scaled = artifacts["X_train_scaled"]
    X_test_scaled = artifacts["X_test_scaled"]
    y_train_smote = artifacts["y_train_smote"]
    feature_order = artifacts["feature_order"]

    # 3. Entrenar con parámetros edge
    model, best_round, best_auc_cv = train_with_cv(
        X_train_scaled=X_train_scaled,
        y_train_smote=y_train_smote,
        params=XGB_EDGE_PARAMS,
        num_boost_round=EDGE_CV_NUM_BOOST_ROUND,
        early_stopping_rounds=EDGE_EARLY_STOPPING_ROUNDS,
    )

    # 4. Optimizar threshold usando datos antes de SMOTE
    threshold = optimize_threshold(
        X_train_ready=artifacts["X_train_ready"],
        y_train=y_train,
        best_round=best_round,
        seed=SEED,
        params=XGB_EDGE_PARAMS,
    )

    # 5. Evaluar en test — PERSISTIR resultado (Fix #8)
    metrics = evaluate_on_test(
        model=model,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        threshold=threshold,
    )

    # 6. Feature importance — PERSISTIR (Fix #8)
    importance = get_feature_importance(model, top_n=25)

    # 7. Guardar artefactos
    EDGE_IOT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    save_artifacts(
        model=model,
        scaler=artifacts["scaler"],
        encoder=artifacts["encoder"],
        num_imputer=artifacts["num_imputer"],
        cat_imputer=artifacts["cat_imputer"],
        ordinal_imputer=artifacts["ordinal_imputer"],
        threshold=threshold,
        seed=SEED,
        feature_order=feature_order,
        artifact_dir=EDGE_IOT_ARTIFACTS_DIR,
    )

    # 8. Guardar muestra de test para export/parity — NO OPCIONAL (Fix #9)
    sample = X_test_scaled.head(PARITY_SAMPLE_SIZE)
    sample_path = EDGE_IOT_ARTIFACTS_DIR / "X_test_scaled_sample.csv"
    sample.to_csv(sample_path, index=False)
    
    # 8b. Sample RAW alineado posicionalmente con el escalado (para Checkpoint 3)
    #     X_test viene de load_and_prepare (sin target, sin leakage, pero sin escalar).
    X_test.head(PARITY_SAMPLE_SIZE).to_csv(
        EDGE_IOT_ARTIFACTS_DIR / "X_test_raw_sample.csv", index=False
    )

    # Procedencia del sample: huella del run + hash del dataset
    fp = run_fingerprint(feature_order, artifacts["scaler"], threshold)
    dataset_hash = hashlib.sha256(SYNTHETIC_IOT_CSV.read_bytes()).hexdigest()

    (EDGE_IOT_ARTIFACTS_DIR / "sample_provenance.json").write_text(
        _json.dumps({
            "run_fingerprint": fp,
            "n_rows": int(len(sample)),
            "n_features": len(feature_order),
            "training_dataset_sha256": dataset_hash,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[train_edge_iot] ✅ Sample + procedencia ({fp[:16]})")

    # ── Fix #8: metrics.json — el número del post deja de existir
    #    solo en el scrollback de una terminal ──

    metrics_payload = {
    "best_round": int(best_round),
    "best_round_at_cap": bool(best_round >= EDGE_CV_NUM_BOOST_ROUND),
    "cv_num_boost_round_cap": int(EDGE_CV_NUM_BOOST_ROUND),
    "early_stopping_caveat": (
        f"early_stopping_rounds={EDGE_EARLY_STOPPING_ROUNDS} es estructuralmente "
        f"inerte sobre esta curva: el CV post-SMOTE es monotono, el contador se "
        f"resetea en cada iteracion y el unico terminator es num_boost_round."
    ),
    "best_auc_cv": float(best_auc_cv),
    "cv_caveat": (
        "CV sobre datos post-SMOTE: AUC de validación optimista. "
        "El número publicable es test_auc."
    ),
    "test_auc": float(metrics["test_auc"]),
    "threshold": float(threshold),
    "threshold_criterion": "F2 sobre split interno de train (pre-SMOTE)",
    "confusion_matrix": metrics["confusion_matrix"].tolist(),
    "tp": int(metrics["tp"]),
    "fn": int(metrics["fn"]),
    "fp": int(metrics["fp"]),
    "tn": int(metrics["tn"]),
    "training_dataset_sha256": dataset_hash,   # Fix #12
    "run_fingerprint": fp,                      # Fix #12
    "best_round_caveat": (                      # Fix #12
        "best_round proviene del CV exterior (post-SMOTE); "
        "el modelo interno del threshold lo reutiliza. Sesgo leve, documentado."
    ),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": len(feature_order),
    "contract_report": artifacts.get("contract_report", {}),
    "feature_importance_gain": importance.to_dict(orient="records"),
    }
    (EDGE_IOT_ARTIFACTS_DIR / "metrics.json").write_text(
        _json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[train_edge_iot] ✅ Métricas persistidas en {EDGE_IOT_ARTIFACTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()