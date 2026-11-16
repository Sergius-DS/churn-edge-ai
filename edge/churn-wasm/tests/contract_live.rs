// edge/churn-wasm/tests/contract_live.rs
// Hornea el contrato DESPLEGADO en el binario de test.
// Si Python cambia el schema y Rust no lo tolera, esto falla en CI.

const LIVE: &str = include_str!("../../inference_contract_edge_iot.json");

#[test]
fn live_contract_deserializes() {
    let c: churn_wasm::contract::InferenceContract =
        serde_json::from_str(LIVE).expect("el contrato desplegado debe parsear en Rust");
    
    // Verificaciones básicas de integridad tras el parseo
    assert_eq!(c.version, 2, "el export escribe v2");
}