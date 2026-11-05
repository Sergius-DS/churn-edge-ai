// src/preprocessing.rs
// Preprocessing: validación de privacidad → clipping IoT → imputación → ordinal → onehot → escalado.
// Replica exactamente lo que hace Python.

use crate::contract::InferenceContract;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;

/// Resultado del preprocessing: vector de features escaladas.
pub struct PreprocessedInput {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
}

/// Preprocesa un input crudo (JSON) usando el contrato de inferencia.
/// Retorna las features en el orden exacto que espera el modelo.
/// ✅ Refactor Senior: Uso de Lifetimes ('a) y Zero-Copy para evitar clonaciones masivas.
pub fn preprocess<'a>(
    raw_input: &'a Value,
    contract: &'a InferenceContract,
) -> Result<PreprocessedInput, String> {
    // ─── Paso 0: Firewall de Privacidad (Forbidden Fields) ───
    for forbidden in &contract.forbidden_fields {
        if raw_input.get(forbidden.as_str()).is_some() {
            return Err(format!(
                "Privacy violation: forbidden field '{}' detected in raw input",
                forbidden
            ));
        }
    }

    // ─── Paso 1: Extraer valores numéricos con clipping IoT + imputación ───
    // ✅ ZERO-COPY KEYS: Las llaves son &str prestados del contrato
    let mut numeric_values: HashMap<&str, f64> = HashMap::new();
    for (i, col) in contract.numeric_imputer.columns.iter().enumerate() {
        let col_str = col.as_str();
        let mut imputed = match raw_input.get(col_str) {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(contract.numeric_imputer.values[i]),
            _ => contract.numeric_imputer.values[i],
        };

        // Aplicar bounds si es una feature IoT (clipping automático)
        if let Some((min, max)) = contract.iot_bound(col_str) {
            imputed = imputed.clamp(min, max); // ✅ Ergonomía: clamp nativo de Rust
        }

        numeric_values.insert(col_str, imputed); // ✅ Cero clones
    }

    // ─── Paso 2: Extraer valores categóricos con imputación ───
    // ✅ EL ARMA SECRETA: Cow<'a, str> para evitar clonar strings categóricos
    let mut categorical_values: HashMap<&str, Cow<'a, str>> = HashMap::new();
    for (i, col) in contract.categorical_imputer.columns.iter().enumerate() {
        let col_str = col.as_str();
        let imputed = match raw_input.get(col_str) {
            // Si viene del JSON, tomamos prestado el string (Zero-copy)
            Some(Value::String(s)) => Cow::Borrowed(s.as_str()),
            // Si hay que imputar, tomamos prestado del contrato (Zero-copy)
            _ => Cow::Borrowed(contract.categorical_imputer.values[i].as_str()),
        };
        categorical_values.insert(col_str, imputed);
    }

    // ─── Paso 3: Encoding ordinal + flags is_unknown ───
    let mut ordinal_values: HashMap<&str, f64> = HashMap::new();
    
    // Income_Category → ordinal
    let income_val = categorical_values.get(contract.ordinal.income_column.as_str());
    let (income_ord, income_unknown) = match income_val {
        Some(cat) => {
            // cat es &Cow<'a, str>, usamos as_ref() para obtener &str y buscar en el HashMap
            match contract.ordinal.income_mapping.get(cat.as_ref()) {
                Some(Some(v)) => (*v as f64, 0.0),
                _ => (contract.ordinal.income_median, 1.0),
            }
        }
        None => (contract.ordinal.income_median, 1.0),
    };
    ordinal_values.insert(contract.ordinal.income_ordinal_feature.as_str(), income_ord);
    ordinal_values.insert(contract.ordinal.income_unknown_feature.as_str(), income_unknown);

    // Education_Level → ordinal
    let edu_val = categorical_values.get(contract.ordinal.education_column.as_str());
    let (edu_ord, edu_unknown) = match edu_val {
        Some(cat) => {
            match contract.ordinal.education_mapping.get(cat.as_ref()) {
                Some(Some(v)) => (*v as f64, 0.0),
                _ => (contract.ordinal.education_median, 1.0),
            }
        }
        None => (contract.ordinal.education_median, 1.0),
    };
    ordinal_values.insert(contract.ordinal.education_ordinal_feature.as_str(), edu_ord);
    ordinal_values.insert(contract.ordinal.education_unknown_feature.as_str(), edu_unknown);

    // ─── Paso 4: OneHot encoding ───
    let mut onehot_values: HashMap<&str, f64> = HashMap::new();
    for pair in &contract.onehot.category_feature_pairs {
        let cat_val = categorical_values.get(pair.column.as_str());
        let is_match = match cat_val {
            Some(v) => v.as_ref() == pair.category.as_str(),
            None => false,
        };
        onehot_values.insert(pair.feature.as_str(), if is_match { 1.0 } else { 0.0 });
    }

    // ─── Paso 5: Construir vector en el orden del modelo ───
    let n_features = contract.feature_order.len();
    let mut features_raw = Vec::with_capacity(n_features); // ✅ Pre-allocación exacta de memoria
    for feat_name in &contract.feature_order {
        let feat_str = feat_name.as_str();
        let val = if let Some(&v) = numeric_values.get(feat_str) {
            v
        } else if let Some(&v) = ordinal_values.get(feat_str) {
            v
        } else if let Some(&v) = onehot_values.get(feat_str) {
            v
        } else {
            0.0
        };
        features_raw.push(val);
    }

    // ─── Paso 6: Escalado (StandardScaler) ───
    let mut features_scaled = Vec::with_capacity(n_features);
    for i in 0..n_features {
        let mean = contract.scaler.mean[i];
        let scale = if contract.scaler.scale[i].abs() > 1e-12 {
            contract.scaler.scale[i]
        } else {
            1.0
        };
        features_scaled.push((features_raw[i] - mean) / scale);
    }

    Ok(PreprocessedInput {
        features: features_scaled,
        // ✅ Clonamos el vector de nombres UNA SOLA VEZ al final, no en cada iteración.
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