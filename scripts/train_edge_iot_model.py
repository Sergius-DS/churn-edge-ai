#!/usr/bin/env python
"""
scripts/train_edge_iot_model.py
───────────────────────────────
Entrena el modelo Edge IoT usando features bancarias + features IoT sintéticas.
"""

import pandas as pd

from src.config import (
    SYNTHETIC_IOT_CSV,
    SEED,
    XGB_EDGE_PARAMS,
    EDGE_CV_NUM_BOOST_ROUND,
    EDGE_EARLY_STOPPING_ROUNDS,
    EDGE_IOT_ARTIFACTS_DIR,
    PARITY_SAMPLE_SIZE,
)

from src.data_loading import load_and_prepare
from src.preprocessing import run_full_pipeline

from src.train import (
    train_with_cv,
    optimize_threshold,
    evaluate_on_test,
    get_feature_importance,
    save_artifacts,
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

    # 5. Evaluar en test
    evaluate_on_test(
        model=model,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        threshold=threshold,
    )

    # 6. Feature importance
    get_feature_importance(model, top_n=25)

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

    # 8. Guardar muestra de test para export/parity
    sample = X_test_scaled.head(PARITY_SAMPLE_SIZE)

    sample_path = EDGE_IOT_ARTIFACTS_DIR / "X_test_scaled_sample.csv"
    sample.to_csv(sample_path, index=False)

    print(f"[train_edge_iot] ✅ Muestra de paridad guardada en {sample_path}")
    print("[train_edge_iot] ✅ Modelo Edge IoT entrenado correctamente")


if __name__ == "__main__":
    main()