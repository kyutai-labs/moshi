use crate::nn::{
    linear, MaybeQuantizedEmbedding as Embedding, MaybeQuantizedLinear as Linear,
    MaybeQuantizedVarBuilder as VarBuilder,
};
use candle::{DType, Result, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LutConfig {
    pub n_bins: usize,
    pub dim: usize,
    pub possible_values: Vec<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ContinuousAttributeConfig {
    pub dim: usize,
    pub scale_factor: f32,
    pub max_period: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ConditionerConfig {
    Lut(LutConfig),
    ContinuousAttribute(ContinuousAttributeConfig),
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
pub struct ContinuousAttributeConditioner {
    scale_factor: f32,
    max_period: f32,
    dim: usize,
    output_proj: Linear,
    #[allow(unused)]
    learnt_padding: Tensor,
    device: candle::Device,
}

impl ContinuousAttributeConditioner {
    pub fn new(output_dim: usize, cfg: &ContinuousAttributeConfig, vb: VarBuilder) -> Result<Self> {
        let output_proj = linear(cfg.dim, output_dim, false, vb.pp("output_proj"))?;
        let learnt_padding = vb.get_as_tensor((1, 1, output_dim), "learnt_padding")?;
        Ok(Self {
            scale_factor: cfg.scale_factor,
            max_period: cfg.max_period,
            dim: cfg.dim,
            output_proj,
            learnt_padding,
            device: vb.device().clone(),
        })
    }

    // `positions` should have shape (b, t, 1), the output will be (b, t, dim)
    pub fn create_sin_embeddings(&self, positions: &Tensor, dtype: DType) -> Result<Tensor> {
        let dev = positions.device();
        let half_dim = self.dim / 2;
        let positions = positions.to_dtype(dtype)?;
        let adim: Vec<_> = (0..half_dim)
            .map(|i| 1f32 / self.max_period.powf(i as f32 / (half_dim - 1) as f32))
            .collect();
        let adim = Tensor::from_vec(adim, (1, 1, ()), dev)?;
        let freqs = positions.broadcast_mul(&adim)?;
        let pos_emb = Tensor::cat(&[freqs.cos()?, freqs.sin()?], candle::D::Minus1)?;
        Ok(pos_emb)
    }

    // TODO(laurent): should we support different values per batch element?
    pub fn condition(&self, value: f32) -> Result<Condition> {
        let value = value * self.scale_factor;
        let positions = Tensor::full(value, (1, 1, 1), &self.device)?;
        let cond = self
            .create_sin_embeddings(&positions, DType::F32)?
            .to_dtype(self.output_proj.dtype())?
            .apply(&self.output_proj)?;
        Ok(Condition::AddToInput(cond))
    }
}

#[derive(Debug, Clone)]
pub enum Conditioner {
    Lut(LutConditioner),
    ContinuousAttribute(ContinuousAttributeConditioner),
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
                ConditionerConfig::ContinuousAttribute(cfg) => Conditioner::ContinuousAttribute(
                    ContinuousAttributeConditioner::new(output_dim, cfg, vb)?,
                ),
            };
            conditioners.insert(conditioner_name.to_string(), conditioner);
        }
        Ok(Self { conditioners })
    }

    pub fn condition_lut(&self, name: &str, value: &str) -> Result<Condition> {
        let lut = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::Lut(l)) => l,
            Some(_) => candle::bail!("cannot use conditioner with a str value {name}"),
        };
        let cond = lut.condition(value)?;
        Ok(cond)
    }

    pub fn condition_cont(&self, name: &str, value: f32) -> Result<Condition> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::ContinuousAttribute(c)) => c,
            Some(_) => candle::bail!("cannot use conditioner with a str value {name}"),
        };
        let cond = c.condition(value)?;
        Ok(cond)
    }

    pub fn learnt_padding(&self, name: &str) -> Result<Condition> {
        let c = match self.conditioners.get(name) {
            None => candle::bail!("unknown conditioner {name}"),
            Some(Conditioner::ContinuousAttribute(c)) => c.learnt_padding.clone(),
            Some(Conditioner::Lut(c)) => c.learnt_padding.clone(),
        };
        Ok(Condition::AddToInput(c))
    }
}
