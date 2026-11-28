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

    // ── Verificaciones adicionales (leídas de los artefactos, no hardcodeadas) ──
    let contract_v: serde_json::Value = serde_json::from_str(&contract_json)
        .expect("contrato IoT con JSON inválido");
    let model_v: serde_json::Value = serde_json::from_str(&model_json)
        .expect("modelo IoT con JSON inválido");

    let exp_threshold = contract_v["best_threshold"]
        .as_f64()
        .expect("contrato sin best_threshold");
    let exp_features = contract_v["n_features"]
        .as_u64()
        .expect("contrato sin n_features");

    // El largo del array ES el conteo. Leer, no re-derivar por substring.
    // El JSON de XGBoost usa arrays paralelos (left_children, right_children,
    // split_indices, split_conditions, base_weights, sum_hessian), no objetos-nodo.
    // "nodeid" no existe en este formato — era un falso amigo del text dump.
    let exp_trees = model_v["learner"]["gradient_booster"]["model"]["trees"]
        .as_array()
        .expect("modelo sin learner.gradient_booster.model.trees")
        .len() as u64;

    assert!(
        (res["threshold"].as_f64().unwrap() - exp_threshold).abs() < 1e-12,
        "Rust debe reportar el threshold del contrato: got={} exp={}",
        res["threshold"], exp_threshold
    );
    assert_eq!(
        res["n_features"].as_u64().unwrap(),
        exp_features,
        "Rust debe reportar n_features del contrato"
    );
    assert_eq!(
        res["n_trees"].as_u64().unwrap(),
        exp_trees,
        "Rust debe reportar los árboles del modelo exportado"
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

/// CHARACTERIZATION TEST — fija el comportamiento ACTUAL, no el deseado.
///
/// GAP CONOCIDO (hueco 1): el preprocessing clipea fuera de rango SIN flag.
/// 9999 aperturas en 30 días (333/día) produce una predicción plausible,
/// sin error y sin marcador de degradación. En CI dio 45.54% — zona de
/// máxima incertidumbre, exactamente donde un input imposible debería
/// rechazarse en lugar de responderse con una decisión de negocio.
///
/// AL CERRAR EL HUECO 1: invertir el assert. Exigir `clipped`/`degraded`,
/// no la ausencia de error. Este test debe fallar en ese commit.
#[test]
fn iot_out_of_bounds_is_silently_clipped_characterization() {
    let model_json = fs::read_to_string("../../edge/xgb_model_edge_iot.json").unwrap();
    let contract_json = fs::read_to_string("../../edge/inference_contract_edge_iot.json").unwrap();

    // ✅ Inicializar motor
    churn_wasm::init_engine(&model_json, &contract_json).expect("Failed to init engine");

    // app_open_count_30d tiene bound [0, 365]. Enviamos 9999.
    // Avg_Utilization_Ratio corregido a 0.061 (= 777/12691) para cumplir INV-2.
    // Antes tenía 0.22, que violaba la identidad algebraica del contrato.
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
        "Avg_Utilization_Ratio": 0.061,  // ← corregido: 777/12691 = 0.061 (antes 0.22, violaba INV-2)
        "app_open_count_30d": 9999,      // ← Fuera de bounds (debe cliparse a 365)
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

    println!("✅ CHARACTERIZATION: clipping silencioso funcionó. Proba con out-of-bounds input: {:.4}%", proba * 100.0);
    println!("   ⚠️  Este comportamiento es el DEFECTO. Ver docstring del test.");
}
/// Paridad sobre el golden de máxima sensibilidad. El típico (p~0.0015)
/// tolera un error de logit 166× mayor; este fixture cierra esa holgura.
#[test]
fn iot_sensitive_golden_parity() {
    let case_json = fs::read_to_string("../../edge/raw_parity_case_iot_sensitive.json")
        .expect("Falta raw_parity_case_iot_sensitive.json — re-ejecutar export_edge_iot.py");
    let case: serde_json::Value = serde_json::from_str(&case_json).unwrap();
    
    let sens = case["sensitivity_p_1mp"].as_f64()
        .expect("golden sensible sin sensitivity_p_1mp");
    let floor = case["sensitivity_floor"].as_f64()
        .expect("golden sensible sin sensitivity_floor");
    assert!(sens >= floor, "sensibilidad {sens} < piso declarado {floor}");
    
    let tol = case["parity_tolerance"]["proba_abs"].as_f64()
        .expect("golden sensible sin parity_tolerance.proba_abs");
    
    // Los dos goldens deben ser del mismo run (ancla embebida, no manifest).
    let typical: serde_json::Value = serde_json::from_str(
        &fs::read_to_string("../../edge/raw_parity_case_iot.json").unwrap()).unwrap();
    let fp_s = case["run_fingerprint"].as_str().expect("sensible sin run_fingerprint");
    let fp_t = typical["run_fingerprint"].as_str().expect("tipico sin run_fingerprint");
    assert_eq!(fp_s, fp_t, "los dos goldens deben pertenecer al mismo run");
    
    let expected = case["expected_proba"].as_f64().unwrap();
    let raw = case["raw"].to_string();
    let model_json = fs::read_to_string("../../edge/xgb_model_edge_iot.json").unwrap();
    let contract_json = fs::read_to_string("../../edge/inference_contract_edge_iot.json").unwrap();
    churn_wasm::init_engine(&model_json, &contract_json).expect("init");
    
    let out = churn_wasm::predict_churn(&raw, true);
    let res: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(!res.as_object().unwrap().contains_key("error"), "predict_churn falló: {out}");
    
    let got = res["probability"].as_f64().unwrap();
    let diff = (got - expected).abs();
    println!("sensible: p={expected:.8} rust={got:.8} diff={diff:.2e} sens={sens:.4}");
    assert!(diff < tol, "PARIDAD SENSIBLE FALLÓ: diff={diff:.2e} tol={tol:.2e}");
}