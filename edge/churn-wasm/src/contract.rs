// src/contract.rs
// Parser del contrato de inferencia exportado por Python.

use serde::Deserialize;

/// Contrato completo de inferencia.
#[derive(Debug, Deserialize)]
pub struct InferenceContract {
    pub version: u32,
    pub best_threshold: f64,
    pub feature_order: Vec<String>,
    pub numeric_imputer: NumericImputer,
    pub categorical_imputer: CategoricalImputer,
    pub ordinal: OrdinalConfig,
    pub onehot: OneHotConfig,
    pub scaler: ScalerConfig,
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
    pub income_mapping: std::collections::HashMap<String, Option<i32>>,
    pub education_mapping: std::collections::HashMap<String, Option<i32>>,
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
    pub categories: std::collections::HashMap<String, Vec<String>>,
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
}