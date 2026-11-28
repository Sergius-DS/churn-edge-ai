// src/model.rs
// Parser del modelo XGBoost exportado como JSON.

use serde::Deserialize;

/// Modelo XGBoost completo: conjunto de árboles + base_score.
#[derive(Debug, Deserialize)]
pub struct XgbModel {
    pub learner: Learner,
    /// Cache del base_score calculado. No viene en el JSON, lo calculamos en from_json.
    #[serde(skip)]
    pub cached_base_score: f64,
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

/// Parsea base_score y lo devuelve YA como margen (logit).
///
/// XGBoost 2.x+ emite `"[5E-1]"`. `str::parse::<f64>()` falla sobre ese
/// formato; el fallback `0.5` histórico coincide con `logit(0.5)=0` y
/// por eso la paridad nunca lo detectó. Acepta:
///   "5E-1"       (formato antiguo, por si reaparece)
///   "[5E-1]"     (formato 2.x, presente en el JSON vivo)
///   "[5E-1,]"    (con coma trailing defensiva)
///
/// Fail-closed: un base_score fuera de (0,1) es export roto, no default.
/// Rechaza también 0.0 y 1.0 porque logit diverge.
fn parse_base_score_as_margin(raw: &str) -> Result<f64, String> {
    let cleaned = raw
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches(',');
    
    let prob: f64 = cleaned
        .parse()
        .map_err(|e| format!("base_score invalido {raw:?}: {e}"))?;
    
    if !prob.is_finite() || prob <= 0.0 || prob >= 1.0 {
        return Err(format!("base_score fuera de (0,1): {prob}"));
    }
    
    Ok((prob / (1.0 - prob)).ln())
}

impl XgbModel {
    /// Parsea el modelo desde un string JSON.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let mut model: Self = serde_json::from_str(json_str)
            .map_err(|e| format!("Error parseando modelo: {}", e))?;
        
        // C1 ATERRIZADO: parse_base_score_as_margin desenvuelve corchetes,
        // valida (0,1) y devuelve logit. Sin fallback: un base_score invalido
        // es export roto, no default.
        model.cached_base_score = parse_base_score_as_margin(&model.learner.learner_model_param.base_score)
            .map_err(|e| format!("modelo invalido: {e}"))?;
        
        Ok(model)
    }

    /// Convierte el base_score del JSON a valor raw (log-odds).
    pub fn base_score_raw(&self) -> f64 {
        self.cached_base_score // O(1) Lookup, cero parsing
    }

    /// Número total de árboles.
    pub fn num_trees(&self) -> usize {
        self.learner.gradient_booster.model.trees.len()
    }
}