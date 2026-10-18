// tests/parity_raw.rs
// Paridad END-TO-END: input RAW → preprocessing Rust → árboles → proba.
// El valor esperado lo genera Python (scripts/validate_parity_raw.py).

use std::fs;

#[test]
fn raw_input_parity_with_python() {
    let case_json = fs::read_to_string("../../edge/raw_parity_case.json")
        .expect("Falta edge/raw_parity_case.json — ejecuta scripts/validate_parity_raw.py");
    let case: serde_json::Value = serde_json::from_str(&case_json).unwrap();

    let raw = case["raw"].to_string();
    let expected = case["expected_proba"].as_f64().unwrap();

    let model = fs::read_to_string("../../edge/xgb_model_edge.json").unwrap();
    let contract = fs::read_to_string("../../edge/inference_contract_edge.json").unwrap();

    let out = churn_wasm::predict_churn(&raw, &model, &contract);
    let res: serde_json::Value = serde_json::from_str(&out)
        .expect("predict_churn devolvió JSON inválido");

    assert!(
        !res.as_object().unwrap().contains_key("error"),
        "predict_churn falló: {}",
        out
    );

    let got = res["probability"].as_f64().unwrap();
    let diff = (got - expected).abs();

    println!("🐍 Python: {expected:.8} | 🦀 Rust: {got:.8} | diff: {diff:.2e}");
    assert!(
        diff < 1e-6,
        "PARIDAD RAW FALLÓ: python={expected} rust={got} diff={diff}"
    );
}