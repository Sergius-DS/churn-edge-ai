#!/usr/bin/env python
"""
scripts/train_edge_model.py
──────────────────────────────
Entrena el modelo edge optimizado para WASM:
  - Menos features (top N por importance)
  - Menos profundidad (max_depth=3)
  - Menos árboles (≤200 rondas)
  - Regularización más fuerte
Compara contra el modelo full.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.data_loading import load_and_prepare
from src.preprocessing import run_full_pipeline
from src.config import (
    ARTIFACTS_DIR,
    EDGE_DIR,
    XGB_EDGE_PARAMS,
    EDGE_CV_NUM_BOOST_ROUND,
    EDGE_EARLY_STOPPING_ROUNDS,
    EDGE_TOP_FEATURES,
    BEST_THRESHOLD,
    SEED,
    TARGET,
)


def main():
    print("=" * 60)
    print("🪶 CHURN EDGE AI — Modelo Edge Optimizado")
    print("=" * 60)

    # ─── Verificar que existe el modelo full ───
    full_model_path = ARTIFACTS_DIR / "xgb_model.joblib"
    assert full_model_path.exists(), (
        "Primero ejecuta: python scripts/train_model.py"
    )

    # ─── Fase 1: Carga y preprocessing ───
    print("\n📂 FASE 1: Carga y preparación de datos")
    print("-" * 40)
    X_train, X_test, y_train, y_test = load_and_prepare()

    pipeline_result = run_full_pipeline(X_train, X_test, y_train, y_test)
    X_train_scaled = pipeline_result["X_train_scaled"]
    X_test_scaled = pipeline_result["X_test_scaled"]
    y_train_smote = pipeline_result["y_train_smote"]
    y_test = pipeline_result["y_test"]

    # ─── Fase 2: Selección de features edge ───
    print("\n🎯 FASE 2: Selección de features edge")
    print("-" * 40)

    available = set(X_train_scaled.columns)
    selected = [f for f in EDGE_TOP_FEATURES if f in available]
    missing = [f for f in EDGE_TOP_FEATURES if f not in available]

    if missing:
        print(f"[edge] ⚠️ Features no encontradas (se omiten): {missing}")

    print(f"[edge] Features seleccionadas: {len(selected)}/{len(EDGE_TOP_FEATURES)}")
    for i, f in enumerate(selected, 1):
        print(f"[edge]   {i:>2}. {f}")

    X_train_edge = X_train_scaled[selected]
    X_test_edge = X_test_scaled[selected]

    # ─── Fase 3: Entrenamiento edge con CV ───
    print("\n🤖 FASE 3: Entrenamiento Edge con CV")
    print("-" * 40)

    dtrain_edge = xgb.DMatrix(X_train_edge, label=y_train_smote)

    cv_results = xgb.cv(
        dtrain=dtrain_edge,
        params=XGB_EDGE_PARAMS,
        num_boost_round=EDGE_CV_NUM_BOOST_ROUND,
        nfold=5,
        early_stopping_rounds=EDGE_EARLY_STOPPING_ROUNDS,
        seed=SEED,
        verbose_eval=25,
    )

    auc_cols = [
        c for c in cv_results.columns
        if "test" in c and "auc" in c and "mean" in c
    ]
    auc_col = auc_cols[0]
    best_round_edge = int(cv_results[auc_col].idxmax()) + 1
    best_auc_edge = cv_results[auc_col].max()

    print(f"\n[edge] Mejor ronda: {best_round_edge}")
    print(f"[edge] Mejor AUC CV: {best_auc_edge:.6f}")

    # Entrenar modelo final edge
    edge_model = xgb.train(
        params=XGB_EDGE_PARAMS,
        dtrain=dtrain_edge,
        num_boost_round=best_round_edge,
    )

    # ─── Fase 4: Evaluación en test ───
    print("\n📊 FASE 4: Evaluación en Test (Edge)")
    print("-" * 40)

    dtest_edge = xgb.DMatrix(X_test_edge, label=y_test)
    y_pred_prob_edge = edge_model.predict(dtest_edge)
    test_auc_edge = roc_auc_score(y_test, y_pred_prob_edge)

    y_pred_edge = (y_pred_prob_edge >= BEST_THRESHOLD).astype(int)
    cm_edge = confusion_matrix(y_test, y_pred_edge)
    report_edge = classification_report(y_test, y_pred_edge)
    tn, fp, fn, tp = cm_edge.ravel()

    print(f"[edge] Test AUC: {test_auc_edge:.6f}")
    print(f"[edge] Threshold: {BEST_THRESHOLD}")
    print(f"[edge] Matriz de confusión:\n{cm_edge}")
    print(f"[edge]\n{report_edge}")
    print(f"[edge] ✅ TP={tp} | FN={fn} | FP={fp} | TN={tn}")

    # ─── Fase 5: Comparación Full vs Edge ───
    print("\n⚖️ FASE 5: Comparación Full vs Edge")
    print("-" * 40)

    full_model = joblib.load(full_model_path)

    dtest_full = xgb.DMatrix(X_test_scaled, label=y_test)
    y_pred_prob_full = full_model.predict(dtest_full)
    test_auc_full = roc_auc_score(y_test, y_pred_prob_full)

    # Tamaños de archivo JSON
    full_json_path = EDGE_DIR / "xgb_model.json"
    full_size = full_json_path.stat().st_size if full_json_path.exists() else 0

    edge_json_path = EDGE_DIR / "xgb_model_edge.json"
    edge_model.save_model(edge_json_path)
    edge_size = edge_json_path.stat().st_size

    full_trees = len(full_model.get_dump())
    edge_trees = len(edge_model.get_dump())

    reduction = (1 - edge_size / full_size) * 100 if full_size > 0 else 0

    print(f"\n{'Métrica':<28} {'Full':>14} {'Edge':>14} {'Δ':>14}")
    print("─" * 74)
    print(f"{'Features':<28} {X_train_scaled.shape[1]:>14} {len(selected):>14} {len(selected) - X_train_scaled.shape[1]:>+14}")
    print(f"{'Árboles':<28} {full_trees:>14} {edge_trees:>14} {edge_trees - full_trees:>+14}")
    print(f"{'max_depth':<28} {4:>14} {3:>14} {-1:>+14}")
    print(f"{'Test AUC':<28} {test_auc_full:>14.6f} {test_auc_edge:>14.6f} {test_auc_edge - test_auc_full:>+14.6f}")
    print(f"{'JSON size (bytes)':<28} {full_size:>14,} {edge_size:>14,} {edge_size - full_size:>+14,}")
    print(f"{'Reducción tamaño':<28} {'':>14} {reduction:>13.1f}%")

    # ─── Fase 6: Guardar artefactos edge ───
    print("\n💾 FASE 6: Guardado de artefactos edge")
    print("-" * 40)

    joblib.dump(edge_model, ARTIFACTS_DIR / "xgb_model_edge.joblib")

    edge_config = {
        "best_threshold": BEST_THRESHOLD,
        "seed": SEED,
        "target_col": TARGET,
        "best_round": best_round_edge,
        "best_auc_cv": float(best_auc_edge),
        "test_auc": float(test_auc_edge),
        "n_features": len(selected),
        "feature_names": selected,
        "model_type": "edge",
    }
    joblib.dump(edge_config, ARTIFACTS_DIR / "model_config_edge.joblib")

    print(f"[edge] ✅ Modelo: {ARTIFACTS_DIR / 'xgb_model_edge.joblib'}")
    print(f"[edge] ✅ Config: {ARTIFACTS_DIR / 'model_config_edge.joblib'}")
    print(f"[edge] ✅ JSON:   {edge_json_path}")

    # ─── Fase 7: Guardar parity sample edge ───
    print("\n📦 FASE 7: Guardando parity sample edge")
    print("-" * 40)

    n_sample = min(100, len(X_train_edge))
    X_sample_edge = X_train_edge.iloc[:n_sample]
    sample_path = ARTIFACTS_DIR / "X_train_edge_sample.csv"
    X_sample_edge.to_csv(sample_path, index=False)
    print(f"[edge] ✅ Parity sample: {sample_path}")
    print(f"[edge]    Shape: ({n_sample}, {len(selected)})")

    # ─── Resumen ───
    print("\n" + "=" * 60)
    print("🪶 MODELO EDGE — RESUMEN FINAL")
    print("=" * 60)
    print(f"   Features:     {len(selected)} (de {X_train_scaled.shape[1]})")
    print(f"   Best round:   {best_round_edge}")
    print(f"   CV AUC:       {best_auc_edge:.6f}")
    print(f"   Test AUC:     {test_auc_edge:.6f}")
    print(f"   JSON size:    {edge_size:,} bytes ({reduction:.1f}% menos)")
    print(f"   Threshold:    {BEST_THRESHOLD}")
    print("=" * 60)

    # ─── Veredicto ───
    auc_drop = test_auc_full - test_auc_edge
    print()
    if auc_drop < 0.005:
        print(f"✅ AUC drop: {auc_drop:.6f} → Edge VIABLE para WASM 🚀")
    elif auc_drop < 0.01:
        print(f"⚠️ AUC drop: {auc_drop:.6f} → Edge ACEPTABLE, monitorear")
    else:
        print(f"❌ AUC drop: {auc_drop:.6f} → Edge pierde demasiada precisión")
        print("   Considera aumentar EDGE_TOP_FEATURES o ajustar XGB_EDGE_PARAMS")


if __name__ == "__main__":
    main()