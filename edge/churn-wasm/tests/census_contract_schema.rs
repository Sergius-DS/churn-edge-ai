//! CENSO B1 — esquema del contrato. Fija el comportamiento ACTUAL, no el deseado.
//!
//! GAP DECLARADO: los cuatro #[serde(default)] de contract.rs (25/30/34/38)
//! tienen default APAGADO. Los campos base NO tienen default. Esa asimetría
//! es el hallazgo: el contrato exige lo que describe el modelo y permite
//! faltar lo que lo protege.
//!
//! check_contract_posture.py detecta esto del lado Python. Del lado Rust no
//! había gate equivalente: contract_live.rs hornea el contrato desplegado y
//! verifica que deserializa — pasaría igual con las cuatro secciones removidas.
//!
//! AL CERRAR C: el segundo test se invierte a is_err(). Debe fallar en ese
//! commit, y su diff es la evidencia del endurecimiento.

use churn_wasm::contract::InferenceContract;

const LIVE: &str = include_str!("../../inference_contract_edge_iot.json");

/// Control positivo del censo. Sin él, "el contrato mutilado carga" no
/// distingue un parser permisivo de uno que solo perdona los cuatro controles.
#[test]
fn census_empty_contract_is_rejected() {
    let r = InferenceContract::from_json("{}");
    assert!(r.is_err(), "CENSO: {{}} debe rechazarse por campos base ausentes");
    let err = r.err().unwrap();
    assert!(
        err.contains("missing field"),
        "el rechazo debe ser por ausencia, no por JSON inválido: {err}"
    );
    println!("CENSO: {{}} → Err ({err}). Los campos base son obligatorios.");
}

/// EL DEFECTO: el artefacto desplegado, sin sus cuatro controles, carga.
#[test]
fn census_live_contract_minus_four_controls_still_loads() {
    let mut v: serde_json::Value = serde_json::from_str(LIVE).unwrap();
    let obj = v.as_object_mut().unwrap();
    for k in ["iot_features", "iot_bounds", "forbidden_fields", "consent_required"] {
        assert!(obj.remove(k).is_some(), "el contrato vivo debe traer {k}");
    }

    let c = InferenceContract::from_json(&v.to_string())
        .expect("CENSO: hoy el contrato mutilado deserializa; si falla, C ya aterrizó");

    assert!(c.forbidden_fields.is_empty(), "firewall vacío por omisión");
    assert!(c.iot_bounds.is_empty(),       "sin clipping por omisión");
    assert!(c.iot_features.is_empty(),     "features IoT no declaradas");
    assert!(!c.consent_required,           "consentimiento apagado por omisión");

    println!("CENSO: contrato sin 4 controles CARGA. firewall=[] bounds={{}} consent=false");
}