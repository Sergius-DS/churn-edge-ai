// edge/churn-wasm/tests/census_bounds_schema.rs
//! CENSO B3 — forma de los bounds. Fija el comportamiento ACTUAL.
//!
//! GAP DECLARADO: iot_bounds es HashMap<String, Vec<f64>>. El comentario de
//! contract.rs dice "arrays [lo, hi] de 2 elementos"; nada lo enforced.
//! from_json es serde puro, sin validación post-parse: acepta [], [100,0]
//! y [1,2,3] por igual.
//!
//! AL CERRAR C: HashMap<String, [f64; 2]> rechaza longitud ≠ 2 en el parse,
//! y una invariante lo <= hi finitos rechaza el invertido. Los tres tests
//! se invierten a is_err() en el mismo diff.

use churn_wasm::contract::InferenceContract;

const LIVE: &str = include_str!("../../inference_contract_edge_iot.json");

fn with_bound(bound: serde_json::Value) -> Result<InferenceContract, String> {
    let mut v: serde_json::Value = serde_json::from_str(LIVE).unwrap();
    v["iot_bounds"]["app_open_count_30d"] = bound;
    InferenceContract::from_json(&v.to_string())
}

#[test]
fn census_empty_bound_is_accepted() {
    let c = with_bound(serde_json::json!([]))
        .expect("CENSO: hoy [] deserializa; si falla, C ya aterrizó");
    assert!(c.iot_bounds["app_open_count_30d"].is_empty());
    println!("CENSO: bound [] ACEPTADO. Vec<f64> no exige longitud 2.");
}

#[test]
fn census_inverted_bound_is_accepted() {
    let c = with_bound(serde_json::json!([100.0, 0.0]))
        .expect("CENSO: hoy [100,0] deserializa; si falla, C ya aterrizó");
    let b = &c.iot_bounds["app_open_count_30d"];
    assert!(b[0] > b[1], "min > max aceptado sin invariante");
    println!("CENSO: bound [100,0] ACEPTADO. min > max llega a f64::clamp.");
}

#[test]
fn census_oversized_bound_is_accepted() {
    let c = with_bound(serde_json::json!([1.0, 2.0, 3.0]))
        .expect("CENSO: hoy [1,2,3] deserializa; si falla, C ya aterrizó");
    assert_eq!(c.iot_bounds["app_open_count_30d"].len(), 3);
    println!("CENSO: bound de 3 elementos ACEPTADO. El tercero se ignora.");
}