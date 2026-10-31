// src/contract.rs
// Parser del contrato de inferencia exportado por Python.
// Soporta tanto edge_lite (12 features) como edge_iot (34 features + IoT).

use serde::Deserialize;
use std::collections::HashMap;

/// Contrato completo de inferencia.
/// Los campos IoT son opcionales para mantener backward-compatibility
/// con el contrato edge_lite.
#[derive(Debug, Deserialize)]
pub struct InferenceContract {
    // ── Campos base (presentes en edge_lite y edge_iot) ──
    pub version: u32,
    pub best_threshold: f64,
    pub feature_order: Vec<String>,
    pub numeric_imputer: NumericImputer,
    pub categorical_imputer: CategoricalImputer,
    pub ordinal: OrdinalConfig,
    pub onehot: OneHotConfig,
    pub scaler: ScalerConfig,

    // ── Campos IoT (opcionales, solo en edge_iot) ──
    /// Lista de features IoT que el modelo espera
    #[serde(default)]
    pub iot_features: Vec<String>,

    /// Rangos de validación para features IoT: feature -> [min, max]
    /// En JSON son arrays [lo, hi], parseados como Vec<f64> de 2 elementos
    #[serde(default)]
    pub iot_bounds: HashMap<String, Vec<f64>>,

    /// Campos prohibidos que NO deben aparecer en el input (PII, etc.)
    #[serde(default)]
    pub forbidden_fields: Vec<String>,

    /// Si true, el usuario debe dar consentimiento antes de la inferencia
    #[serde(default)]
    pub consent_required: bool,
}

#[derive(Debug, Deserialize)]
pub struct NumericImputer {
    pub columns: Vec<String>,
    pub values: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct CategoricalImputer {
    pub columns: Vec<String>,
    pub values: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct OrdinalConfig {
    pub income_column: String,
    pub education_column: String,
    pub income_mapping: HashMap<String, Option<i32>>,
    pub education_mapping: HashMap<String, Option<i32>>,
    pub income_ordinal_feature: String,
    pub education_ordinal_feature: String,
    pub income_unknown_feature: String,
    pub education_unknown_feature: String,
    pub income_median: f64,
    pub education_median: f64,
}

#[derive(Debug, Deserialize)]
pub struct OneHotConfig {
    pub columns: Vec<String>,
    pub categories: HashMap<String, Vec<String>>,
    pub encoded_feature_names: Vec<String>,
    pub category_feature_pairs: Vec<CategoryFeaturePair>,
}

#[derive(Debug, Deserialize)]
pub struct CategoryFeaturePair {
    pub column: String,
    pub category: String,
    pub feature: String,
}

#[derive(Debug, Deserialize)]
pub struct ScalerConfig {
    pub columns: Vec<String>,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

impl InferenceContract {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        serde_json::from_str(json_str)
            .map_err(|e| format!("Error parseando contrato: {}", e))
    }

    /// Retorna el índice de una feature en el scaler.
    pub fn scaler_index(&self, feature: &str) -> Option<usize> {
        self.scaler.columns.iter().position(|c| c == feature)
    }

    // ── Métodos helper para IoT ──

    /// Obtiene los bounds de una feature IoT como (min, max).
    /// Retorna None si la feature no tiene bounds o el formato es inválido.
    pub fn iot_bound(&self, feature: &str) -> Option<(f64, f64)> {
        self.iot_bounds.get(feature).and_then(|bounds| {
            if bounds.len() == 2 {
                Some((bounds[0], bounds[1]))
            } else {
                None
            }
        })
    }

    /// Verifica si una feature es IoT (está en iot_features).
    pub fn is_iot_feature(&self, feature: &str) -> bool {
        self.iot_features.iter().any(|f| f == feature)
    }

    /// Verifica si un campo está en la lista de forbidden_fields.
    pub fn is_forbidden(&self, field: &str) -> bool {
        self.forbidden_fields.iter().any(|f| f == field)
    }

    /// Retorna el número de features que el modelo espera.
    pub fn n_features(&self) -> usize {
        self.feature_order.len()
    }

    /// Verifica si el contrato tiene campos IoT (es edge_iot).
    pub fn has_iot(&self) -> bool {
        !self.iot_features.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iot_bound_parsing() {
        let mut bounds = HashMap::new();
        bounds.insert("app_open_count_30d".to_string(), vec![0.0, 365.0]);
        
        let contract = InferenceContract {
            version: 1,
            best_threshold: 0.33,
            feature_order: vec![],
            numeric_imputer: NumericImputer { columns: vec![], values: vec![] },
            categorical_imputer: CategoricalImputer { columns: vec![], values: vec![] },
            ordinal: OrdinalConfig {
                income_column: "".to_string(),
                education_column: "".to_string(),
                income_mapping: HashMap::new(),
                education_mapping: HashMap::new(),
                income_ordinal_feature: "".to_string(),
                education_ordinal_feature: "".to_string(),
                income_unknown_feature: "".to_string(),
                education_unknown_feature: "".to_string(),
                income_median: 0.0,
                education_median: 0.0,
            },
            onehot: OneHotConfig {
                columns: vec![],
                categories: HashMap::new(),
                encoded_feature_names: vec![],
                category_feature_pairs: vec![],
            },
            scaler: ScalerConfig { columns: vec![], mean: vec![], scale: vec![] },
            iot_features: vec!["app_open_count_30d".to_string()],
            iot_bounds: bounds,
            forbidden_fields: vec![],
            consent_required: true,
        };

        let bound = contract.iot_bound("app_open_count_30d");
        assert_eq!(bound, Some((0.0, 365.0)));
        
        let missing = contract.iot_bound("nonexistent");
        assert_eq!(missing, None);
    }

    #[test]
    fn test_is_iot_feature() {
        let contract = InferenceContract {
            version: 1,
            best_threshold: 0.33,
            feature_order: vec![],
            numeric_imputer: NumericImputer { columns: vec![], values: vec![] },
            categorical_imputer: CategoricalImputer { columns: vec![], values: vec![] },
            ordinal: OrdinalConfig {
                income_column: "".to_string(),
                education_column: "".to_string(),
                income_mapping: HashMap::new(),
                education_mapping: HashMap::new(),
                income_ordinal_feature: "".to_string(),
                education_ordinal_feature: "".to_string(),
                income_unknown_feature: "".to_string(),
                education_unknown_feature: "".to_string(),
                income_median: 0.0,
                education_median: 0.0,
            },
            onehot: OneHotConfig {
                columns: vec![],
                categories: HashMap::new(),
                encoded_feature_names: vec![],
                category_feature_pairs: vec![],
            },
            scaler: ScalerConfig { columns: vec![], mean: vec![], scale: vec![] },
            iot_features: vec!["app_open_count_30d".to_string(), "pos_latency_p95_ms".to_string()],
            iot_bounds: HashMap::new(),
            forbidden_fields: vec![],
            consent_required: false,
        };

        assert!(contract.is_iot_feature("app_open_count_30d"));
        assert!(contract.is_iot_feature("pos_latency_p95_ms"));
        assert!(!contract.is_iot_feature("Customer_Age"));
    }

    #[test]
    fn test_has_iot() {
        let contract_with_iot = InferenceContract {
            version: 1,
            best_threshold: 0.33,
            feature_order: vec![],
            numeric_imputer: NumericImputer { columns: vec![], values: vec![] },
            categorical_imputer: CategoricalImputer { columns: vec![], values: vec![] },
            ordinal: OrdinalConfig {
                income_column: "".to_string(),
                education_column: "".to_string(),
                income_mapping: HashMap::new(),
                education_mapping: HashMap::new(),
                income_ordinal_feature: "".to_string(),
                education_ordinal_feature: "".to_string(),
                income_unknown_feature: "".to_string(),
                education_unknown_feature: "".to_string(),
                income_median: 0.0,
                education_median: 0.0,
            },
            onehot: OneHotConfig {
                columns: vec![],
                categories: HashMap::new(),
                encoded_feature_names: vec![],
                category_feature_pairs: vec![],
            },
            scaler: ScalerConfig { columns: vec![], mean: vec![], scale: vec![] },
            iot_features: vec!["app_open_count_30d".to_string()],
            iot_bounds: HashMap::new(),
            forbidden_fields: vec![],
            consent_required: true,
        };

        assert!(contract_with_iot.has_iot());

        let contract_without_iot = InferenceContract {
            iot_features: vec![],
            ..contract_with_iot
        };

        assert!(!contract_without_iot.has_iot());
    }
}