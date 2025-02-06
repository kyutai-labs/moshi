use crate::nn::{
    linear, MaybeQuantizedEmbedding as Embedding, MaybeQuantizedLinear as Linear,
    MaybeQuantizedVarBuilder as VarBuilder,
};
use candle::{Result, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub possible_values: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ConditionerConfig {
    Lut(LutConfig),
}

pub type Config = HashMap<String, ConditionerConfig>;

#[derive(Debug, Clone)]
pub struct LutConditioner {
    embed: Embedding,
    output_proj: Linear,
    #[allow(unused)]
    learnt_padding: Tensor,
    possible_values: HashMap<String, usize>,
}

impl LutConditioner {
    pub fn new(output_dim: usize, cfg: &LutConfig, vb: VarBuilder) -> Result<Self> {
        let embed = Embedding::new(cfg.n_bins + 1, cfg.dim, vb.pp("embed"))?;
        let output_proj = linear(cfg.dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        let possible_values: HashMap<String, usize> =
            cfg.possible_values.iter().enumerate().map(|(i, v)| (v.to_string(), i)).collect();
        Ok(Self { embed, output_proj, learnt_padding, possible_values })
    }

    pub fn condition(&self, value: &str) -> Result<Condition> {
        let idx = match self.possible_values.get(value) {
            None => candle::bail!("unknown value for lut conditioner '{value}'"),
            Some(idx) => *idx,
        };
        let cond = Tensor::from_vec(vec![idx as u32], (1, 1), self.embed.embeddings().device())?
            .apply(&self.embed)?
            .apply(&self.output_proj)?;
        Ok(Condition::AddToInput(cond))
    }
}

#[derive(Debug, Clone)]
pub enum Conditioner {
    Lut(LutConditioner),
}

#[derive(Debug, Clone)]
pub struct ConditionProvider {
    conditioners: HashMap<String, Conditioner>,
}

#[derive(Debug, Clone)]
pub enum Condition {
    AddToInput(Tensor),
}

impl ConditionProvider {
    pub fn new(output_dim: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("conditioners");
        let mut conditioners = HashMap::new();
        for (conditioner_name, conditioner_cfg) in cfg.iter() {
            let vb = vb.pp(conditioner_name);
            let conditioner = match conditioner_cfg {
                ConditionerConfig::Lut(cfg) => {
                    Conditioner::Lut(LutConditioner::new(output_dim, cfg, vb)?)
                }
            };
            conditioners.insert(conditioner_name.to_string(), conditioner);
        }
        Ok(Self { conditioners })
    }

    pub fn condition_lut(&self, name: &str, value: &str) -> Result<Condition> {
        let lut = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::Lut(l)) => l,
        };
        let cond = lut.condition(value)?;
        Ok(cond)
    }
}
