// src/lib.rs
// Entry point del módulo WASM.
// Exporta funciones que JavaScript puede llamar.

mod contract;
mod inference;
mod model;
mod preprocessing;

use wasm_bindgen::prelude::*;
use serde_json::Value;

/// Resultado de la predicción como JSON string.
#[derive(serde::Serialize)]
struct PredictionResult {
    probability: f64,
    probability_pct: f64,
    margin: f64,
    is_churn: bool,
    threshold: f64,
    n_features: usize,
    n_trees: usize,
}

/// Función principal exportada a JavaScript.
///
/// # Argumentos
/// - `input_json`: JSON string con los datos crudos del cliente
/// - `model_json`: JSON string del modelo XGBoost (xgb_model_edge.json)
/// - `contract_json`: JSON string del contrato (inference_contract_edge.json)
///
/// # Retorna
/// JSON string con la predicción.
#[wasm_bindgen]
pub fn predict_churn(
    input_json: &str,
    model_json: &str,
    contract_json: &str,
) -> String {
    // ─── Parsear inputs ───
    let raw_input: Value = match serde_json::from_str(input_json) {
        Ok(v) => v,
        Err(e) => return error_response(&format!("Error parseando input: {}", e)),
    };

    let model = match model::XgbModel::from_json(model_json) {
        Ok(m) => m,
        Err(e) => return error_response(&format!("Error parseando modelo: {}", e)),
    };

    let contract = match contract::InferenceContract::from_json(contract_json) {
        Ok(c) => c,
        Err(e) => return error_response(&format!("Error parseando contrato: {}", e)),
    };

    // ─── Preprocessing ───
    let preprocessed = match preprocessing::preprocess(&raw_input, &contract) {
        Ok(p) => p,
        Err(e) => return error_response(&format!("Error en preprocessing: {}", e)),
    };

    // ─── Inferencia ───
    let margin = inference::predict_margin(&model, &preprocessed.features);
    let probability = inference::sigmoid(margin);
    let is_churn = probability >= contract.best_threshold;

    // ─── Resultado ───
    let result = PredictionResult {
        probability,
        probability_pct: probability * 100.0,
        margin,
        is_churn,
        threshold: contract.best_threshold,
        n_features: preprocessed.features.len(),
        n_trees: model.num_trees(),
    };

    match serde_json::to_string(&result) {
        Ok(json) => json,
        Err(e) => error_response(&format!("Error serializando resultado: {}", e)),
    }
}

/// Función auxiliar para validar paridad con Python.
/// Toma features ya escaladas y retorna probabilidad + margin.
#[wasm_bindgen]
pub fn predict_from_scaled_features(
    features_json: &str,
    model_json: &str,
) -> String {
    let features: Vec<f64> = match serde_json::from_str(features_json) {
        Ok(f) => f,
        Err(e) => return error_response(&format!("Error parseando features: {}", e)),
    };

    let model = match model::XgbModel::from_json(model_json) {
        Ok(m) => m,
        Err(e) => return error_response(&format!("Error parseando modelo: {}", e)),
    };

    let margin = inference::predict_margin(&model, &features);
    let probability = inference::sigmoid(margin);

    let result = serde_json::json!({
        "probability": probability,
        "margin": margin,
    });

    result.to_string()
}

/// Función de verificación de paridad.
/// Compara la predicción WASM con el expected de Python.
#[wasm_bindgen]
pub fn verify_parity(
    features_json: &str,
    expected_proba: f64,
    model_json: &str,
    tolerance: f64,
) -> String {
    let features: Vec<f64> = match serde_json::from_str(features_json) {
        Ok(f) => f,
        Err(e) => return error_response(&format!("Error parseando features: {}", e)),
    };

    let model = match model::XgbModel::from_json(model_json) {
        Ok(m) => m,
        Err(e) => return error_response(&format!("Error parseando modelo: {}", e)),
    };

    let margin = inference::predict_margin(&model, &features);
    let probability = inference::sigmoid(margin);
    let diff = (probability - expected_proba).abs();
    let passed = diff <= tolerance;

    let result = serde_json::json!({
        "python_proba": expected_proba,
        "rust_proba": probability,
        "diff": diff,
        "tolerance": tolerance,
        "passed": passed,
    });

    result.to_string()
}

/// Respuesta de error estándar.
fn error_response(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

// ═══════════════════════════════════════════════════════════
// TESTS NATIVOS (cargo test)
// ═══════════════════════════════════════════════════════════
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_basic() {
        assert!((inference::sigmoid(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_error_response() {
        let resp = error_response("test error");
        assert!(resp.contains("test error"));
    }
}