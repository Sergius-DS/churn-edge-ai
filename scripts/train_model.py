#!/usr/bin/env python
"""
scripts/train_model.py
──────────────────────
Entry point principal: python scripts/train_model.py

Ejecuta el pipeline completo:
  1. Carga y limpieza de datos
  2. Preprocessing (impute → ordinal → onehot → SMOTE → scale)
  3. Entrenamiento XGBoost con CV
  4. Evaluación en test
  5. Guardado de artefactos
"""

import sys
from pathlib import Path

# Asegurar que el proyecto está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_and_prepare
from src.preprocessing import run_full_pipeline
from src.train import (
    train_with_cv,
    evaluate_on_test,
    get_feature_importance,
    save_artifacts,
)
from src.config import BEST_THRESHOLD


def main():
    print("=" * 60)
    print("🚀 CHURN EDGE AI — Pipeline de Entrenamiento")
    print("=" * 60)

    # ─── Fase 1: Carga de datos ───
    print("\n📂 FASE 1: Carga y preparación de datos")
    print("-" * 40)
    X_train, X_test, y_train, y_test = load_and_prepare()

    # ─── Fase 2: Preprocessing ───
    print("\n🔧 FASE 2: Preprocessing completo")
    print("-" * 40)
    pipeline_result = run_full_pipeline(X_train, X_test, y_train, y_test)

    X_train_scaled = pipeline_result["X_train_scaled"]
    X_test_scaled = pipeline_result["X_test_scaled"]
    y_train_smote = pipeline_result["y_train_smote"]
    y_test = pipeline_result["y_test"]

    # ─── Fase 3: Entrenamiento ───
    print("\n🤖 FASE 3: Entrenamiento XGBoost con CV")
    print("-" * 40)
    model, best_round, best_auc_cv = train_with_cv(X_train_scaled, y_train_smote)

    # ─── Fase 4: Evaluación ───
    print("\n📊 FASE 4: Evaluación en Test")
    print("-" * 40)
    eval_result = evaluate_on_test(model, X_test_scaled, y_test, BEST_THRESHOLD)

    # ─── Fase 5: Feature Importance ───
    print("\n🏆 FASE 5: Feature Importance")
    print("-" * 40)
    get_feature_importance(model, top_n=15)

    # ─── Fase 5.5: Guardar muestra real para export_edge ───
    print("\n📦 FASE 5.5: Guardando muestra escalada para parity")
    print("-" * 40)

    import pandas as pd
    from src.config import ARTIFACTS_DIR, PARITY_SAMPLE_SIZE

    n_sample = min(PARITY_SAMPLE_SIZE, len(X_train_scaled))

    if hasattr(X_train_scaled, "iloc"):
        X_sample = X_train_scaled.iloc[:n_sample]
    else:
        X_sample = pd.DataFrame(
            X_train_scaled[:n_sample],
            columns=model.feature_names
        )

    X_sample.to_csv(
        ARTIFACTS_DIR / "X_train_scaled_sample.csv",
        index=False
    )

    print(f"[train] ✅ Muestra guardada: {ARTIFACTS_DIR / 'X_train_scaled_sample.csv'}")
    print(f"[train]    Shape: {X_sample.shape}")

    # ─── Fase 6: Guardar artefactos ───
    print("\n💾 FASE 6: Guardado de artefactos")
    print("-" * 40)
    save_artifacts(
        model=model,
        scaler=pipeline_result["scaler"],
        encoder=pipeline_result["encoder"],
        num_imputer=pipeline_result["num_imputer"],
        cat_imputer=pipeline_result["cat_imputer"],
        ordinal_imputer=pipeline_result["ordinal_imputer"],
        threshold=BEST_THRESHOLD,
    )

    # ─── Resumen ───
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO")
    print(f"   Best round: {best_round}")
    print(f"   CV AUC:     {best_auc_cv:.6f}")
    print(f"   Test AUC:   {eval_result['test_auc']:.6f}")
    print(f"   Threshold:  {BEST_THRESHOLD}")
    print("=" * 60)


if __name__ == "__main__":
    main()