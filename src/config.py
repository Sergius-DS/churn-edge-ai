"""
src/config.py
─────────────
Constantes globales del proyecto.
Cualquier cambio de hiperparámetros, rutas o mappings se hace AQUÍ.
"""

from pathlib import Path


# ============================================================
# RUTAS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Credit_card_churn.csv"

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic"

SYNTHETIC_IOT_CSV = PROCESSED_PATH / "credit_card_churn_iot_synthetic.csv"
SYNTHETIC_IOT_METADATA = SYNTHETIC_PATH / "iot_features_metadata.json"

ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
EDGE_DIR = PROJECT_ROOT / "edge"

EDGE_ARTIFACTS_DIR = ARTIFACTS_DIR / "edge"
EDGE_IOT_ARTIFACTS_DIR = ARTIFACTS_DIR / "edge_iot"

EDGE_IOT_MODEL_JSON = EDGE_DIR / "xgb_model_edge_iot.json"
EDGE_IOT_CONTRACT_JSON = EDGE_DIR / "inference_contract_edge_iot.json"


# ============================================================
# REPRODUCIBILIDAD
# ============================================================
SEED = 44
TARGET = "Attrition_Flag"


# ============================================================
# LIMPIEZA DE DATOS
# ============================================================
LEAKAGE_COLUMNS = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

NUMERIC_LIKE_COLUMNS = [
    "Total_Amt_Chng_Q4_Q1",
    "Total_Ct_Chng_Q4_Q1",
]


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

CV_NUM_BOOST_ROUND = 500
CV_NFOLD = 5
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

EDGE_CV_NUM_BOOST_ROUND = 200
EDGE_EARLY_STOPPING_ROUNDS = 10


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
IOT_FEATURES = [
    "app_open_count_30d",
    "contactless_usage_count_30d",
    "payment_flow_abandon_count_30d",
    "nfc_failed_attempts_30d",
    "app_crash_count_30d",
    "pos_latency_p95_ms",
]

EDGE_IOT_TOP_FEATURES = EDGE_TOP_FEATURES + IOT_FEATURES


# ============================================================
# REFERENCIA FIJA PARA NORMALIZACIÓN IoT
# ============================================================
# Esto evita usar estadísticas del dataset completo.
# Es reproducible y sirve para inferencia sobre 1 sola fila.
#
# formato:
# "columna": (media_referencia, std_referencia)
# ============================================================
IOT_REFERENCE = {
    "Total_Trans_Ct": (140.0, 58.0),
    "Total_Trans_Amt": (4600.0, 2600.0),
    "Months_Inactive_12_mon": (2.3, 1.1),
    "Contacts_Count_12_mon": (2.4, 1.2),
    "Avg_Utilization_Ratio": (0.53, 0.26),
    "Total_Ct_Chng_Q4_Q1": (0.95, 0.25),
    "Total_Amt_Chng_Q4_Q1": (0.95, 0.25),
    "Total_Relationship_Count": (4.3, 1.1),
}


# ============================================================
# PARÁMETROS DE GENERACIÓN SINTÉTICA IoT
# ============================================================
SYNTHETIC_IOT_PARAMS = {
    "target_strength": 0.0,
    "noise_scale": 0.65,
}


# ============================================================
# RANGOS IoT PARA VALIDACIÓN
# ============================================================
IOT_FEATURE_BOUNDS = {
    "app_open_count_30d": (0, 365),
    "contactless_usage_count_30d": (0, 500),
    "payment_flow_abandon_count_30d": (0, 100),
    "nfc_failed_attempts_30d": (0, 100),
    "app_crash_count_30d": (0, 100),
    "pos_latency_p95_ms": (50, 20000),
}


# ============================================================
# THRESHOLD DE NEGOCIO
# ============================================================
BEST_THRESHOLD = 0.38


# ============================================================
# PARITY / EXPORT
# ============================================================
PARITY_SAMPLE_SIZE = 100
PARITY_TOLERANCE = 1e-4


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