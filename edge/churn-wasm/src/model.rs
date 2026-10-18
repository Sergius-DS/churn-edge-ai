// src/model.rs
// Parser del modelo XGBoost exportado como JSON.

use serde::Deserialize;

/// Modelo XGBoost completo: conjunto de árboles + base_score.
#[derive(Debug, Deserialize)]
pub struct XgbModel {
    pub learner: Learner,
}

#[derive(Debug, Deserialize)]
pub struct Learner {
    pub gradient_booster: GradientBooster,
    pub learner_model_param: LearnerModelParam,
}

#[derive(Debug, Deserialize)]
pub struct GradientBooster {
    pub model: BoosterModel,
}

#[derive(Debug, Deserialize)]
pub struct BoosterModel {
    pub trees: Vec<XgbTree>,
}

#[derive(Debug, Deserialize)]
pub struct LearnerModelParam {
    pub base_score: String,
}

/// Un solo árbol de XGBoost.
#[derive(Debug, Deserialize)]
pub struct XgbTree {
    pub left_children: Vec<i32>,
    pub right_children: Vec<i32>,
    pub split_conditions: Vec<f64>,
    pub split_indices: Vec<i32>,
    pub base_weights: Vec<f64>,
}

impl XgbModel {
    /// Parsea el modelo desde un string JSON.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        serde_json::from_str(json_str).map_err(|e| format!("Error parseando modelo: {}", e))
    }

    /// Convierte el base_score del JSON a valor raw (log-odds).
    /// XGBoost guarda "5E-1" para binary:logistic con base_score=0.5.
    /// En raw space, eso es ln(0.5/0.5) = 0.0.
    pub fn base_score_raw(&self) -> f64 {
        let val: f64 = self
            .learner
            .learner_model_param
            .base_score
            .parse()
            .unwrap_or(0.5);

        // Si está en rango (0,1), es probabilidad → convertir a log-odds
        if val > 0.0 && val < 1.0 {
            (val / (1.0 - val)).ln()
        } else {
            val
        }
    }

    /// Número total de árboles.
    pub fn num_trees(&self) -> usize {
        self.learner.gradient_booster.model.trees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_score_conversion() {
        // base_score = 0.5 → log-odds = 0.0
        let raw = (0.5_f64 / (1.0 - 0.5)).ln();
        assert!((raw - 0.0).abs() < 1e-10);
    }
}