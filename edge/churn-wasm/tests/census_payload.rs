//! CENSO B2 — payload. Fija el comportamiento ACTUAL, no el deseado.
//!
//! GAP DECLARADO: preprocessing.rs:40 `_ => contract.numeric_imputer.values[i]`
//! colapsa tres estados en uno. Ausente, null, string, bool, array y objeto
//! reciben todos la media del imputer. `invalid ≠ missing` no se cumple.
//! La rama de la línea 39 (`Number::as_f64().unwrap_or`) es muerta:
//! sin `arbitrary_precision`, as_f64 es total. La red está donde no hay caída.
//!
//! AL CERRAR C: el segundo test se invierte — string donde se espera número
//! debe producir Err, no la media.

use churn_wasm::contract::InferenceContract;
use churn_wasm::preprocessing::preprocess;
use serde_json::json;

const LIVE: &str = include_str!("../../inference_contract_edge_iot.json");

fn live() -> InferenceContract {
    InferenceContract::from_json(LIVE).expect("contrato vivo debe parsear")
}

/// EL DEFECTO: payload vacío → cliente completo con las medias del imputer.
#[test]
fn census_empty_payload_produces_a_plausible_client() {
    let c = live();
    let r = preprocess(&json!({}), &c);
    assert!(r.is_ok(), "CENSO: hoy {{}} se imputa completo; si falla, C ya aterrizó");
    println!("CENSO: payload {{}} → Ok. 25 campos inventados desde el imputer.");
}

/// EL DEFECTO, más filoso: tipo inválido tratado como ausencia.
#[test]
fn census_string_where_number_expected_is_imputed() {
    let c = live();
    let r = preprocess(&json!({ "Total_Trans_Ct": "9999" }), &c);
    assert!(r.is_ok(), "CENSO: hoy un string se imputa en silencio; si falla, C ya aterrizó");
    println!("CENSO: \"9999\" (string) → Ok. Confusión de tipos = ausencia.");
}