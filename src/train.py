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
    fbeta_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from src.config import (
    SEED,
    TARGET,
    XGB_PARAMS,
    XGB_EDGE_PARAMS,
    CV_NUM_BOOST_ROUND,
    CV_NFOLD,
    CV_EARLY_STOPPING_ROUNDS,
    EDGE_CV_NUM_BOOST_ROUND,
    EDGE_EARLY_STOPPING_ROUNDS,
    BEST_THRESHOLD,
    ARTIFACTS_DIR,
    INCOME_ORDER,
    EDUCATION_ORDER,
)


def train_with_cv(
    X_train_scaled: pd.DataFrame,
    y_train_smote: pd.Series,
    params: dict = None,
    num_boost_round: int = None,
    early_stopping_rounds: int = None,
) -> tuple[xgb.Booster, int, float]:
    """
    Entrena XGBoost con cross-validation.
    Permite usar XGB_PARAMS (baseline) o XGB_EDGE_PARAMS (edge/iot).
    """
    params = params or XGB_PARAMS
    num_boost_round = num_boost_round or CV_NUM_BOOST_ROUND
    early_stopping_rounds = early_stopping_rounds or CV_EARLY_STOPPING_ROUNDS

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train_smote)

    cv_results = xgb.cv(
        dtrain=dtrain,
        params=params,
        num_boost_round=num_boost_round,
        nfold=CV_NFOLD,
        early_stopping_rounds=early_stopping_rounds,
        seed=SEED,
        verbose_eval=50,
    )

    auc_mean_cols = [
        col for col in cv_results.columns
        if col.startswith("test") and "auc" in col and col.endswith("mean")
    ]
    auc_col = auc_mean_cols[0]

    best_round = int(cv_results[auc_col].idxmax()) + 1
    best_auc_cv = cv_results[auc_col].max()

    print(f"[train] Mejor ronda: {best_round}")
    print(f"[train] Mejor AUC en CV: {best_auc_cv:.6f}")

    final_model = xgb.train(
        params=params,
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
    params: dict = None,
) -> float:
    """
    Optimiza threshold usando split interno de train.
    Usa X_train_ready antes de SMOTE para evitar leakage.
    """
    params = params or XGB_PARAMS

    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_train_ready,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=seed,
    )

    smote_inner = SMOTE(random_state=seed)
    X_tr_inner_sm, y_tr_inner_sm = smote_inner.fit_resample(
        X_tr_inner,
        y_tr_inner,
    )

    scaler_inner = StandardScaler()
    X_tr_inner_sc = scaler_inner.fit_transform(X_tr_inner_sm)
    X_val_inner_sc = scaler_inner.transform(X_val_inner)

    dtr_inner = xgb.DMatrix(X_tr_inner_sc, label=y_tr_inner_sm)

    model_inner = xgb.train(
        params=params,
        dtrain=dtr_inner,
        num_boost_round=best_round,
    )

    dval_inner = xgb.DMatrix(X_val_inner_sc)
    val_prob = model_inner.predict(dval_inner)

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
    feature_order: list = None,
    artifact_dir: Path = ARTIFACTS_DIR,
) -> None:
    """
    Guarda todos los artefactos del modelo.
    artifact_dir permite guardar en carpetas separadas (baseline, edge, edge_iot).
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "xgb_model.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(encoder, artifact_dir / "onehot_encoder.joblib")
    joblib.dump(num_imputer, artifact_dir / "num_imputer.joblib")
    joblib.dump(cat_imputer, artifact_dir / "cat_imputer.joblib")
    joblib.dump(ordinal_imputer, artifact_dir / "ordinal_imputer.joblib")

    mappings = {
        "income_order": INCOME_ORDER,
        "education_order": EDUCATION_ORDER,
    }
    joblib.dump(mappings, artifact_dir / "ordinal_mappings.joblib")

    config = {
        "best_threshold": threshold,
        "seed": seed,
        "target_col": TARGET,
        "feature_order": feature_order or [],
    }
    joblib.dump(config, artifact_dir / "model_config.joblib")

    print(f"[train] ✅ Artefactos guardados en {artifact_dir}")