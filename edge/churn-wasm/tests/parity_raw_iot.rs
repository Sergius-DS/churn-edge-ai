// tests/parity_raw_iot.rs
// ═══════════════════════════════════════════════════════════
// PARIDAD END-TO-END EDGE IOT: input RAW (25 campos) → preprocessing Rust
// → árboles XGBoost → proba.
// ═══════════════════════════════════════════════════════════

use std::fs;

/// Tolerancia de paridad: Rust (f32 en árboles) vs Python (f64).
const TOLERANCE: f64 = 1e-6;

#[test]
fn iot_raw_input_parity_with_python() {
    // ── Cargar el golden file generado por Python ──
    let case_path = "../../edge/raw_parity_case_iot.json";
    let case_json = fs::read_to_string(case_path).unwrap_or_else(|_| {
        panic!(
            "Falta {} — ejecuta `uv run py scripts/export_edge_iot.py` primero",
            case_path
        )
    });

    let case: serde_json::Value = serde_json::from_str(&case_json)
        .expect("raw_parity_case_iot.json tiene JSON inválido");

    let raw = case["raw"].to_string();
    let expected = case["expected_proba"]
        .as_f64()
        .expect("Falta expected_proba en raw_parity_case_iot.json");

    // ── Cargar modelo y contrato IoT ──
    let model_json = fs::read_to_string("../../edge/xgb_model_edge_iot.json")
        .expect("Falta edge/xgb_model_edge_iot.json");
    let contract_json = fs::read_to_string("../../edge/inference_contract_edge_iot.json")
        .expect("Falta edge/inference_contract_edge_iot.json");

    // ✅ REFACTOR SENIOR: Inicializar el motor UNA SOLA VEZ (Cacheo en OnceLock)
    churn_wasm::init_engine(&model_json, &contract_json)
        .expect("Failed to initialize WASM engine");

    // ✅ REFACTOR SENIOR: Inferencia ultrarrápida (sin parsear JSONs pesados)
    let out = churn_wasm::predict_churn(&raw, true);
    let res: serde_json::Value = serde_json::from_str(&out)
        .expect("predict_churn devolvió JSON inválido");

    // ── Verificar que no hubo errores ──
    assert!(
        !res.as_object().unwrap().contains_key("error"),
        "predict_churn falló: {}",
        out
    );

    let got = res["probability"].as_f64().unwrap();
    let diff = (got - expected).abs();

    // ── Output legible ──
    println!("═══════════════════════════════════════════════════════════");
    println!("🔬 PARIDAD END-TO-END EDGE IOT");
    println!("═══════════════════════════════════════════════════════════");
    println!("🐍 Python (expected): {:.8}", expected);
    println!("🦀 Rust   (got):      {:.8}", got);
    println!("📉 Diff:              {:.2e}", diff);
    println!("🎯 Tolerancia:        {:.2e}", TOLERANCE);
    println!("🏁 Resultado:         {}", if diff < TOLERANCE { "✅ PASS" } else { "❌ FAIL" });
    println!("═══════════════════════════════════════════════════════════");

    // ── Assertion de paridad ──
    assert!(
        diff < TOLERANCE,
        "❌ PARIDAD IOT FALLÓ: python={:.8} rust={:.8} diff={:.2e} (tol={:.2e})",
        expected, got, diff, TOLERANCE
    );

    // ── Verificaciones adicionales ──
    assert_eq!(
        res["n_features"].as_u64().unwrap(),
        34,
        "El modelo IoT debe tener 34 features"
    );
    assert_eq!(
        res["n_trees"].as_u64().unwrap(),
        200,
        "El modelo IoT debe tener 200 árboles"
    );
    assert_eq!(
        res["threshold"].as_f64().unwrap(),
        0.33,
        "El threshold IoT debe ser 0.33"
    );
}

/// Test adicional: verificar que el firewall de privacidad rechaza campos prohibidos.
#[test]
fn iot_privacy_firewall_blocks_forbidden_fields() {
    let model_json = fs::read_to_string("../../edge/xgb_model_edge_iot.json").unwrap();
    let contract_json = fs::read_to_string("../../edge/inference_contract_edge_iot.json").unwrap();

    // ✅ Inicializar motor
    churn_wasm::init_engine(&model_json, &contract_json).expect("Failed to init engine");

    // Payload con un campo prohibido (pan)
    let malicious_input = serde_json::json!({
        "Customer_Age": 45,
        "pan": "4532123456789012",  // ← CAMPO PROHIBIDO
        "Gender": "F"
    });

    // ✅ Nueva firma: solo input y consentimiento
    let out = churn_wasm::predict_churn(&malicious_input.to_string(), true);
    let res: serde_json::Value = serde_json::from_str(&out).unwrap();

    assert!(
        res.as_object().unwrap().contains_key("error"),
        "El firewall debe rechazar el campo 'pan'"
    );

    let error_msg = res["error"].as_str().unwrap();
    assert!(
        error_msg.contains("Privacy violation") || error_msg.contains("forbidden"),
        "El error debe mencionar violación de privacidad: {}",
        error_msg
    );

    println!("✅ Firewall de privacidad funcionó: {}", error_msg);
}

/// Test adicional: verificar que el clipping IoT funciona.
#[test]
fn iot_bounds_clipping_works() {
    let model_json = fs::read_to_string("../../edge/xgb_model_edge_iot.json").unwrap();
    let contract_json = fs::read_to_string("../../edge/inference_contract_edge_iot.json").unwrap();

    // ✅ Inicializar motor
    churn_wasm::init_engine(&model_json, &contract_json).expect("Failed to init engine");

    // app_open_count_30d tiene bound [0, 365]. Enviamos 9999.
    let out_of_bounds_input = serde_json::json!({
        "Customer_Age": 45,
        "Gender": "F",
        "Dependent_count": 3,
        "Education_Level": "Graduate",
        "Marital_Status": "Married",
        "Income_Category": "$60K - $80K",
        "Card_Category": "Blue",
        "Months_on_book": 39,
        "Total_Relationship_Count": 5,
        "Months_Inactive_12_mon": 2,
        "Contacts_Count_12_mon": 3,
        "Credit_Limit": 12691.0,
        "Total_Revolving_Bal": 777,
        "Avg_Open_To_Buy": 11914.0,
        "Total_Amt_Chng_Q4_Q1": 0.8,
        "Total_Trans_Amt": 3500,
        "Total_Trans_Ct": 45,
        "Total_Ct_Chng_Q4_Q1": 0.69,
        "Avg_Utilization_Ratio": 0.22,
        "app_open_count_30d": 9999,  // ← Fuera de bounds (debe cliparse a 365)
        "contactless_usage_count_30d": 58,
        "payment_flow_abandon_count_30d": 5,
        "nfc_failed_attempts_30d": 3,
        "app_crash_count_30d": 2,
        "pos_latency_p95_ms": 1500
    });

    // ✅ Nueva firma: solo input y consentimiento
    let out = churn_wasm::predict_churn(&out_of_bounds_input.to_string(), true);
    let res: serde_json::Value = serde_json::from_str(&out).unwrap();

    // Debe producir una predicción válida (no error)
    assert!(
        !res.as_object().unwrap().contains_key("error"),
        "El clipping debe permitir procesar valores fuera de bounds. Error: {}",
        out
    );

    let proba = res["probability"].as_f64().unwrap();
    assert!(
        (0.0..=1.0).contains(&proba),
        "La probabilidad debe estar en [0, 1]: {}",
        proba
    );

    println!("✅ Clipping IoT funcionó. Proba con out-of-bounds input: {:.4}%", proba * 100.0);
}