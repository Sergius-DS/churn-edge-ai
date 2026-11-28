// edge/churn-wasm/tests/census_base_score.rs
//! CENSO B4 — base_score.
//!
//! C1 ATERRIZADO: XGBoost 2.x+ emite base_score como array-string ("[5E-1]").
//! `parse_base_score_as_margin` desenvuelve los corchetes y devuelve el logit.
//! El fallback 0.5 histórico coincidía con logit(0.5)=0, por eso la paridad
//! nunca detectó el bug. Ahora:
//!   "[5E-1]" → logit(0.5) =  0.0
//!   "[3E-1]" → logit(0.3) = -0.847298
//!
//! El primer assert verifica el FORMATO del JSON (no el parseo): si xgboost
//! deja de emitir array-string, este test avisa para revisar el supuesto.

use churn_wasm::model::XgbModel;

const LIVE_MODEL: &str = include_str!("../../xgb_model_edge_iot.json");

#[test]
fn census_base_score_bracket_format_is_parsed_correctly() {
    let v: serde_json::Value = serde_json::from_str(LIVE_MODEL).unwrap();
    let raw = v["learner"]["learner_model_param"]["base_score"]
        .as_str().expect("base_score no es string").to_string();

    // Guard de formato: el JSON vivo usa array-string. `raw.parse::<f64>()`
    // falla sobre "[5E-1]" — el parser necesita desenvolver corchetes.
    assert!(raw.parse::<f64>().is_err(),
        "CENSO: el formato dejó de ser array-string (raw={raw:?}); revisar el supuesto");

    // Caso vivo: 0.5 → 0.0. Ahora por logit(0.5), no por fallback.
    let m = XgbModel::from_json(LIVE_MODEL).expect("modelo vivo debe cargar");
    assert_eq!(m.base_score_raw(), 0.0);

    // Caso no trivial: 0.3 → -0.847298 (antes del fix: 0.0 silencioso).
    let mut v2 = v.clone();
    v2["learner"]["learner_model_param"]["base_score"] = serde_json::json!("[3E-1]");
    let m2 = XgbModel::from_json(&v2.to_string())
        .expect("C1 aterrizado: '[3E-1]' debe cargar");
    let expected = (0.3_f64 / (1.0 - 0.3_f64)).ln();
    assert!((m2.base_score_raw() - expected).abs() < 1e-6,
        "C1: '[3E-1]' debe producir {expected}, obtuvo {} (0.0 = bug vivo)",
        m2.base_score_raw());

    println!("C1: '{raw}' → {:.6}, '[3E-1]' → {:.6}",
        m.base_score_raw(), m2.base_score_raw());
}

#[test]
fn base_score_outside_unit_interval_is_rejected() {
    let base: serde_json::Value = serde_json::from_str(LIVE_MODEL).unwrap();
    for bad in ["[0]", "[1]", "[1.5]", "[-0.1]", "[NaN]"] {
        let mut v = base.clone();
        v["learner"]["learner_model_param"]["base_score"] = serde_json::json!(bad);
        assert!(XgbModel::from_json(&v.to_string()).is_err(),
            "C1: base_score {bad} debe rechazarse (logit diverge o fuera de rango)");
    }
}

#[test]
fn legacy_unbracketed_base_score_is_accepted() {
    let mut v: serde_json::Value = serde_json::from_str(LIVE_MODEL).unwrap();
    v["learner"]["learner_model_param"]["base_score"] = serde_json::json!("5E-1");
    let m = XgbModel::from_json(&v.to_string())
        .expect("formato legado sin corchetes debe seguir cargando");
    assert_eq!(m.base_score_raw(), 0.0);
}