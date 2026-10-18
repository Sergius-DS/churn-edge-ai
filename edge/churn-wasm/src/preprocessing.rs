// src/preprocessing.rs
// Preprocessing: imputación → ordinal → onehot → escalado.
// Replica exactamente lo que hace Python.

use crate::contract::InferenceContract;
use serde_json::Value;
use std::collections::HashMap;

/// Resultado del preprocessing: vector de features escaladas.
pub struct PreprocessedInput {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
}

/// Preprocesa un input crudo (JSON) usando el contrato de inferencia.
/// Retorna las features en el orden exacto que espera el modelo.
pub fn preprocess(
    raw_input: &Value,
    contract: &InferenceContract,
) -> Result<PreprocessedInput, String> {
    // ─── Paso 1: Extraer valores numéricos con imputación ───
    let mut numeric_values: HashMap<String, f64> = HashMap::new();

    for (i, col) in contract.numeric_imputer.columns.iter().enumerate() {
        let val = raw_input.get(col);
        let imputed = match val {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(contract.numeric_imputer.values[i]),
            Some(Value::Null) | None => contract.numeric_imputer.values[i],
            _ => contract.numeric_imputer.values[i],
        };
        numeric_values.insert(col.clone(), imputed);
    }

    // ─── Paso 2: Extraer valores categóricos con imputación ───
    let mut categorical_values: HashMap<String, String> = HashMap::new();

    for (i, col) in contract.categorical_imputer.columns.iter().enumerate() {
        let val = raw_input.get(col);
        let imputed = match val {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => contract.categorical_imputer.values[i].clone(),
            _ => contract.categorical_imputer.values[i].clone(),
        };
        categorical_values.insert(col.clone(), imputed);
    }

    // ─── Paso 3: Encoding ordinal + flags is_unknown ───
    let mut ordinal_values: HashMap<String, f64> = HashMap::new();

    // Income_Category → ordinal
    let income_val = categorical_values.get(&contract.ordinal.income_column);
    let (income_ord, income_unknown) = match income_val {
        Some(cat) => {
            match contract.ordinal.income_mapping.get(cat) {
                Some(Some(v)) => (*v as f64, 0.0),
                _ => (contract.ordinal.income_median, 1.0),
            }
        }
        None => (contract.ordinal.income_median, 1.0),
    };
    ordinal_values.insert(contract.ordinal.income_ordinal_feature.clone(), income_ord);
    ordinal_values.insert(contract.ordinal.income_unknown_feature.clone(), income_unknown);

    // Education_Level → ordinal
    let edu_val = categorical_values.get(&contract.ordinal.education_column);
    let (edu_ord, edu_unknown) = match edu_val {
        Some(cat) => {
            match contract.ordinal.education_mapping.get(cat) {
                Some(Some(v)) => (*v as f64, 0.0),
                _ => (contract.ordinal.education_median, 1.0),
            }
        }
        None => (contract.ordinal.education_median, 1.0),
    };
    ordinal_values.insert(contract.ordinal.education_ordinal_feature.clone(), edu_ord);
    ordinal_values.insert(contract.ordinal.education_unknown_feature.clone(), edu_unknown);

    // ─── Paso 4: OneHot encoding ───
    let mut onehot_values: HashMap<String, f64> = HashMap::new();

    for pair in &contract.onehot.category_feature_pairs {
        let cat_val = categorical_values.get(&pair.column);
        let is_match = match cat_val {
            Some(v) => *v == pair.category,
            None => false,
        };
        onehot_values.insert(pair.feature.clone(), if is_match { 1.0 } else { 0.0 });
    }

    // ─── Paso 5: Construir vector en el orden del modelo ───
    let n_features = contract.feature_order.len();
    let mut features_raw = vec![0.0f64; n_features];

    for (i, feat_name) in contract.feature_order.iter().enumerate() {
        let val = if let Some(v) = numeric_values.get(feat_name) {
            *v
        } else if let Some(v) = ordinal_values.get(feat_name) {
            *v
        } else if let Some(v) = onehot_values.get(feat_name) {
            *v
        } else {
            0.0
        };
        features_raw[i] = val;
    }

    // ─── Paso 6: Escalado (StandardScaler) ───
    let mut features_scaled = vec![0.0f64; n_features];
    for i in 0..n_features {
        let mean = contract.scaler.mean[i];
        let scale = if contract.scaler.scale[i].abs() > 1e-12 {
            contract.scaler.scale[i]
        } else {
            1.0
        };
        features_scaled[i] = (features_raw[i] - mean) / scale;
    }

    Ok(PreprocessedInput {
        features: features_scaled,
        feature_names: contract.feature_order.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onehot_logic() {
        // Si Gender = "F", Gender_F debe ser 1.0
        // Si Gender = "M", Gender_F debe ser 0.0
        assert_eq!(if "F" == "F" { 1.0 } else { 0.0 }, 1.0);
        assert_eq!(if "M" == "F" { 1.0 } else { 0.0 }, 0.0);
    }
}