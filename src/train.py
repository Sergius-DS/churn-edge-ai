"""
src/train.py
────────────
Entrenamiento XGBoost con CV, evaluación en test,
optimización de threshold y guardado de artefactos.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    SEED,
    TARGET,
    XGB_PARAMS,
    CV_NUM_BOOST_ROUND,
    CV_NFOLD,
    CV_EARLY_STOPPING_ROUNDS,
    BEST_THRESHOLD,
    ARTIFACTS_DIR,
)


def train_with_cv(
    X_train_scaled: pd.DataFrame,
    y_train_smote: pd.Series,
) -> tuple[xgb.Booster, int, float]:
    """
    Entrena XGBoost con cross-validation.
    Retorna el modelo final, la mejor ronda y el mejor AUC en CV.
    """
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_smote)

    cv_results = xgb.cv(
        dtrain=dtrain,
        params=XGB_PARAMS,
        num_boost_round=CV_NUM_BOOST_ROUND,
        nfold=CV_NFOLD,
        early_stopping_rounds=CV_EARLY_STOPPING_ROUNDS,
        seed=SEED,
        verbose_eval=50,
    )

    # Encontrar la mejor ronda
    auc_mean_cols = [
        col for col in cv_results.columns
        if col.startswith("test") and "auc" in col and col.endswith("mean")
    ]
    auc_col = auc_mean_cols[0]

    best_round = int(cv_results[auc_col].idxmax()) + 1
    best_auc_cv = cv_results[auc_col].max()

    print(f"[train] Mejor ronda: {best_round}")
    print(f"[train] Mejor AUC en CV: {best_auc_cv:.6f}")

    # Entrenar modelo final con la mejor ronda
    final_model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=best_round,
    )

    return final_model, best_round, best_auc_cv


def evaluate_on_test(
    model: xgb.Booster,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = BEST_THRESHOLD,
) -> dict:
    """Evalúa el modelo en test con el threshold de negocio."""
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)
    y_pred_prob = model.predict(dtest)

    test_auc = roc_auc_score(y_test, y_pred_prob)
    y_pred = (y_pred_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"[train] Test AUC: {test_auc:.6f}")
    print(f"[train] Threshold: {threshold}")
    print(f"[train] Matriz de confusión:\n{cm}")
    print(f"[train]\n{report}")
    print(f"[train] ✅ TP={tp} | FN={fn} | FP={fp} | TN={tn}")

    return {
        "test_auc": test_auc,
        "confusion_matrix": cm,
        "y_pred_prob": y_pred_prob,
        "threshold": threshold,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def optimize_threshold(
    X_train_ready: pd.DataFrame,
    y_train: pd.Series,
    best_round: int,
    seed: int = SEED,
) -> float:
    """
    Optimiza el threshold usando un split interno de train
    (antes de SMOTE) para no hacer leakage sobre test.
    """
    # Split interno
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_train_ready, y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=seed,
    )

    # SMOTE solo al sub-training
    smote_inner = SMOTE(random_state=seed)
    X_tr_inner_sm, y_tr_inner_sm = smote_inner.fit_resample(X_tr_inner, y_tr_inner)

    # Escalar
    scaler_inner = StandardScaler()
    X_tr_inner_sc = scaler_inner.fit_transform(X_tr_inner_sm)
    X_val_inner_sc = scaler_inner.transform(X_val_inner)

    # Entrenar modelo temporal
    dtr_inner = xgb.DMatrix(X_tr_inner_sc, label=y_tr_inner_sm)
    model_inner = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtr_inner,
        num_boost_round=best_round,
    )

    # Predecir en validación interna
    dval_inner = xgb.DMatrix(X_val_inner_sc)
    val_prob = model_inner.predict(dval_inner)

    # Buscar mejor threshold con F2 (prioriza recall)
    thresholds = np.linspace(0.10, 0.90, 81)
    best_f2 = 0
    best_thr = 0.5

    for thr in thresholds:
        pred = (val_prob >= thr).astype(int)
        f2 = fbeta_score(y_val_inner, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = thr

    print(f"[train] Threshold óptimo (F2): {best_thr:.2f} con F2={best_f2:.4f}")
    return best_thr


def get_feature_importance(model: xgb.Booster, top_n: int = 15) -> pd.DataFrame:
    """Retorna feature importance por gain."""
    importance_gain = model.get_score(importance_type="gain")
    importance_df = (
        pd.DataFrame(importance_gain.items(), columns=["feature", "gain"])
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[train] Top {top_n} features por gain:")
    print(importance_df.head(top_n).to_string(index=False))
    return importance_df.head(top_n)


def save_artifacts(
    model: xgb.Booster,
    scaler: StandardScaler,
    encoder,
    num_imputer,
    cat_imputer,
    ordinal_imputer,
    threshold: float,
    seed: int = SEED,
) -> None:
    """Guarda todos los artefactos del modelo."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACTS_DIR / "xgb_model.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(encoder, ARTIFACTS_DIR / "onehot_encoder.joblib")
    joblib.dump(num_imputer, ARTIFACTS_DIR / "num_imputer.joblib")
    joblib.dump(cat_imputer, ARTIFACTS_DIR / "cat_imputer.joblib")
    joblib.dump(ordinal_imputer, ARTIFACTS_DIR / "ordinal_imputer.joblib")

    from src.config import INCOME_ORDER, EDUCATION_ORDER
    mappings = {
        "income_order": INCOME_ORDER,
        "education_order": EDUCATION_ORDER,
    }
    joblib.dump(mappings, ARTIFACTS_DIR / "ordinal_mappings.joblib")

    config = {
        "best_threshold": threshold,
        "seed": seed,
        "target_col": TARGET,
    }
    joblib.dump(config, ARTIFACTS_DIR / "model_config.joblib")

    print(f"[train] ✅ Artefactos guardados en {ARTIFACTS_DIR}")