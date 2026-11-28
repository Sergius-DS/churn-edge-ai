// edge/churn-wasm/tests/census_contract_schema.rs
//! CENSO B1 — esquema del contrato.
//!
//! HISTORIAL:
//! - fe8e996 (censo B): los cuatro #[serde(default)] tenían default APAGADO.
//!   El contrato vivo menos sus cuatro controles DESERIALIZABA. Firewall=[],
//!   bounds={}, consent=false. El defecto era que el parser perdonaba lo
//!   que protege y exigía lo que describe el modelo.
//!
//! - C3 (este commit): los cuatro #[serde(default)] se eliminaron de
//!   contract.rs. Un contrato sin controles se rechaza en el parse.
//!   El test `census_live_contract_minus_four_controls_is_rejected`
//!   verifica la inversión: lo que antes cargaba, ahora falla.
//!
//! El control positivo (`census_empty_contract_is_rejected`) se conserva:
//! sin él, un test de rechazo no distingue un parser estricto de uno roto.

use churn_wasm::contract::InferenceContract;

const LIVE: &str = include_str!("../../inference_contract_edge_iot.json");

/// Control positivo del censo. Sin él, "el contrato mutilado se rechaza"
/// no distingue un parser estricto de uno que falla por JSON inválido.
#[test]
fn census_empty_contract_is_rejected() {
    let r = InferenceContract::from_json("{}");
    assert!(r.is_err(), "{{}} debe rechazarse por campos base ausentes");
    let err = r.err().unwrap();
    assert!(
        err.contains("missing field"),
        "el rechazo debe ser por ausencia, no por JSON inválido: {err}"
    );
}

/// C3 INVIERTE EL CENSO B: el contrato vivo sin sus cuatro controles
/// ahora se RECHAZA. Los #[serde(default)] fueron eliminados de
/// contract.rs — firewall, bounds, features IoT y consentimiento son
/// obligatorios. Un contrato mutilado muere en el parse, no en runtime.
///
/// Commit de referencia del censo B: fe8e996.
#[test]
fn census_live_contract_minus_four_controls_is_rejected() {
    let mut v: serde_json::Value = serde_json::from_str(LIVE).unwrap();
    let obj = v.as_object_mut().unwrap();
    for k in ["iot_features", "iot_bounds", "forbidden_fields", "consent_required"] {
        assert!(obj.remove(k).is_some(), "el contrato vivo debe traer {k}");
    }

    let r = InferenceContract::from_json(&v.to_string());
    assert!(
        r.is_err(),
        "C3: contrato sin los 4 controles debe rechazarse en el parse"
    );
    let err = r.err().unwrap();
    assert!(
        err.contains("missing field"),
        "el rechazo debe ser por ausencia de campo obligatorio: {err}"
    );
}