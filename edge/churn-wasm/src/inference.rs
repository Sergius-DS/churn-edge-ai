// src/inference.rs
// Motor de inferencia: recorre los árboles XGBoost y produce la predicción.
use crate::model::XgbModel;

/// Recorre un solo árbol y retorna la predicción (valor de la hoja).
/// ✅ Refactor Senior: Acepta &[f64] para hacer el cast al vuelo (Zero-Copy).
fn predict_single_tree(tree: &crate::model::XgbTree, features: &[f64]) -> f32 {
    let mut node: usize = 0;
    loop {
        let left_child = tree.left_children[node];
        // Si left_child == -1, es una hoja
        if left_child == -1 {
            return tree.base_weights[node] as f32;
        }
        // Nodo interno: comparar feature con split_condition
        let feat_idx = tree.split_indices[node] as usize;
        let condition = tree.split_conditions[node] as f32;
        
        // ✅ Zero-Copy: Evaluamos y hacemos el cast (as f32) en el momento exacto.
        // Mantenemos la lógica exacta de seguridad de índices de tu código original.
        if feat_idx < features.len() && (features[feat_idx] as f32) < condition {
            node = left_child as usize;
        } else {
            node = tree.right_children[node] as usize;
        }
    }
}

/// Calcula el margin (raw score) sumando todos los árboles + base_score.
/// ✅ Refactor Senior: Cero allocaciones en el Heap.
pub fn predict_margin(model: &XgbModel, features: &[f64]) -> f64 {
    // 1. Iniciar margin en f32 (XGBoost acumula en f32 internamente)
    let mut margin: f32 = model.base_score_raw() as f32;
    
    // 2. Recorrer árboles pasando el slice &[f64] original (sin clonar)
    for tree in &model.learner.gradient_booster.model.trees {
        margin += predict_single_tree(tree, features);
    }
    
    // 3. Devolver a f64 para el sigmoide final
    margin as f64
}

/// Función sigmoide: convierte margin a probabilidad.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Predicción completa: probabilidad de churn.
pub fn predict_proba(model: &XgbModel, features: &[f64]) -> f64 {
    let margin = predict_margin(model, features);
    sigmoid(margin)
}

/// Predicción con threshold: retorna clase (0 o 1).
pub fn predict_class(model: &XgbModel, features: &[f64], threshold: f64) -> i32 {
    let proba = predict_proba(model, features);
    if proba >= threshold {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let x = 2.5;
        assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < 1e-10);
    }
}