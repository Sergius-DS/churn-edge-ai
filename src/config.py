"""
src/config.py
─────────────
Constantes globales del proyecto.
Cualquier cambio de hiperparámetros, rutas o mappings se hace AQUÍ.

SSOT de datos: data/curated/churn_curated.csv (no el raw).
SSOT de contrato: src/data_contract.py (una sola fuente por hecho).
"""

from pathlib import Path

# ← CAMBIO: una sola fuente por hecho
from src.data_contract import (
    FORBIDDEN_FOR_TRAINING as LEAKAGE_COLUMNS,
    TARGET_MAP,
    IOT_FEATURES,
    IOT_PHYSICAL_BOUNDS as IOT_FEATURE_BOUNDS,
)


# ============================================================
# RUTAS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ← CAMBIO: el origen queda para forense; la SSOT operativa es el curated
RAW_ORIGIN_PATH = PROJECT_ROOT / "data" / "raw" / "Credit_card_churn.csv"
DATA_PATH       = PROJECT_ROOT / "data" / "curated" / "churn_curated.csv"

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic"

SYNTHETIC_IOT_CSV      = PROCESSED_PATH / "credit_card_churn_iot_synthetic.csv"
SYNTHETIC_IOT_METADATA = SYNTHETIC_PATH / "iot_features_metadata.json"

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
EDGE_DIR      = PROJECT_ROOT / "edge"

EDGE_ARTIFACTS_DIR     = ARTIFACTS_DIR / "edge"
EDGE_IOT_ARTIFACTS_DIR = ARTIFACTS_DIR / "edge_iot"

EDGE_IOT_MODEL_JSON    = EDGE_DIR / "xgb_model_edge_iot.json"
EDGE_IOT_CONTRACT_JSON = EDGE_DIR / "inference_contract_edge_iot.json"


# ============================================================
# REPRODUCIBILIDAD
# ============================================================
SEED   = 44
TARGET = "Attrition_Flag"


# ============================================================
# LIMPIEZA DE DATOS
# ============================================================
# ← CAMBIO: LEAKAGE_COLUMNS ahora viene del contrato (import arriba).
# ← CAMBIO: NUMERIC_LIKE_COLUMNS ELIMINADO.
#   Fósil del síntoma normalizado. Registrado en ADR-0005.
#   El contrato garantiza dtype; la coerción era un disfraz.


# ============================================================
# MAPEOS ORDINALES
# ============================================================
INCOME_ORDER = {
    "Less than $40K": 0,
    "$40K - $60K": 1,
    "$60K - $80K": 2,
    "$80K - $120K": 3,
    "$120K +": 4,
    "Unknown": None,
}

EDUCATION_ORDER = {
    "Uneducated": 0,
    "High School": 1,
    "College": 2,
    "Graduate": 3,
    "Post-Graduate": 4,
    "Doctorate": 5,
    "Unknown": None,
}

ORDINAL_SOURCE_COLUMNS = ["Income_Category", "Education_Level"]


# ============================================================
# SPLIT
# ============================================================
TEST_SIZE = 0.20


# ============================================================
# XGBOOST — Modelo FULL
# ============================================================
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "seed": SEED,
    "verbosity": 1,
}

CV_NUM_BOOST_ROUND       = 500
CV_NFOLD                 = 5
CV_EARLY_STOPPING_ROUNDS = 8


# ============================================================
# XGBOOST — Modelo EDGE
# ============================================================
XGB_EDGE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 3,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_lambda": 1.5,
    "seed": SEED,
    "verbosity": 1,
}

EDGE_CV_NUM_BOOST_ROUND       = 200
EDGE_EARLY_STOPPING_ROUNDS    = 10


# ============================================================
# FEATURES EDGE ACTUALES
# ============================================================
EDGE_TOP_FEATURES = [
    "Total_Trans_Ct",
    "Total_Revolving_Bal",
    "Total_Ct_Chng_Q4_Q1",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Total_Trans_Amt",
    "Total_Relationship_Count",
    "Avg_Utilization_Ratio",
    "Total_Amt_Chng_Q4_Q1",
    "Dependent_count",
    "Gender_F",
    "Marital_Status_Single",
]


# ============================================================
# FEATURES IoT
# ============================================================
# ← CAMBIO: IOT_FEATURES ahora viene del contrato (import arriba).
# ← CAMBIO: IOT_FEATURE_BOUNDS ahora viene del contrato (import arriba).

EDGE_IOT_TOP_FEATURES = EDGE_TOP_FEATURES + IOT_FEATURES


# ============================================================
# REFERENCIA FIJA PARA NORMALIZACIÓN IoT
# ============================================================
# ← CAMBIO: medida sobre TRAIN del curated, ddof=1.
# Procedencia: data/curated/iot_reference_provenance.json
# curated sha256 6bced2a3bf04db7b · n_train=7996 · seed=44
IOT_REFERENCE = {
    "Total_Trans_Ct":            (65.2112, 23.3703),
    "Total_Trans_Amt":           (4446.1595, 3412.3573),
    "Months_Inactive_12_mon":    (2.3488, 1.0054),
    "Contacts_Count_12_mon":     (2.4577, 1.1089),
    "Avg_Utilization_Ratio":     (0.2764, 0.2775),
    "Total_Ct_Chng_Q4_Q1":       (0.7058, 0.2231),
    "Total_Amt_Chng_Q4_Q1":      (0.7576, 0.2167),
    "Total_Relationship_Count":  (3.8034, 1.5517),
}


# ============================================================
# PARÁMETROS DE GENERACIÓN SINTÉTICA IoT
# ============================================================
SYNTHETIC_IOT_PARAMS = {
    "target_strength": 0.0,
    "noise_scale": 0.65,
}


# ============================================================
# THRESHOLD DE NEGOCIO
# ============================================================
# ← CAMBIO: STALE. Linaje corrupto. Se re-mide tras regeneración atómica.
BEST_THRESHOLD = 0.38


# ============================================================
# PARITY / EXPORT
# ============================================================
PARITY_SAMPLE_SIZE = 100
PARITY_TOLERANCE   = 1e-4


# ============================================================
# CONTRATO / PRIVACIDAD
# ============================================================
FORBIDDEN_FIELDS = [
    "pan",
    "cvv",
    "card_number",
    "account_number",
    "gps_exact",
    "latitude",
    "longitude",
    "device_id",
    "raw_merchant_name",
]