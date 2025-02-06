use candle::quantized::QTensor;
use candle::{DType, Device, Module, Result, Shape, Tensor};
use candle_transformers::quantized_nn as candle_qnn;
use candle_transformers::quantized_var_builder::VarBuilder as QuantizedVarBuilder;

use std::sync::Arc;

#[derive(Clone)]
pub enum MaybeQuantizedWeight {
    // Enum types around real and quantized model weights
    Real(Tensor),
    Quantized(Arc<QTensor>),
}

impl MaybeQuantizedWeight {
    fn to_tensor(&self, dev: &Device) -> Result<Tensor> {
        match self {
            Self::Real(t) => Ok(t.clone()),
            Self::Quantized(t) => t.dequantize(dev),
        }
    }
}

pub fn matmul_dtype(device: &candle::Device) -> DType {
    // Dtype used for intermediate matmul in attention during quantized execution
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

#[derive(Clone)]
pub enum MaybeQuantizedVarBuilder<'a> {
    // Enum types around real and quantized var builders
    Real(candle_nn::VarBuilder<'a>),
    Quantized(QuantizedVarBuilder),
}

impl MaybeQuantizedVarBuilder<'_> {
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        match self {
            Self::Real(weights) => MaybeQuantizedVarBuilder::Real(weights.pp(s)),
            Self::Quantized(weights) => MaybeQuantizedVarBuilder::Quantized(weights.pp(s)),
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, path: &str) -> Result<MaybeQuantizedWeight> {
        let w = match self {
            Self::Real(weights) => MaybeQuantizedWeight::Real(weights.get(s, path)?),
            Self::Quantized(weights) => MaybeQuantizedWeight::Quantized(weights.get(s, path)?),
        };
        Ok(w)
    }

    pub fn get_as_tensor<S: Into<Shape>>(&self, s: S, path: &str) -> Result<Tensor> {
        let w = match self {
            Self::Real(weights) => MaybeQuantizedWeight::Real(weights.get(s, path)?),
            Self::Quantized(weights) => MaybeQuantizedWeight::Quantized(weights.get(s, path)?),
        };
        w.to_tensor(self.device())
    }

    pub fn get_unquantized<S: Into<Shape>>(&self, s: S, path: &str) -> Result<Tensor> {
        match self {
            Self::Real(weights) => weights.get(s, path),
            Self::Quantized(weights) => weights.get(s, path)?.dequantize(weights.device()),
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        match self {
            Self::Real(weights) => weights.contains_tensor(name),
            Self::Quantized(weights) => weights.contains_key(name),
        }
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Real(weights) => weights.device(),
            Self::Quantized(weights) => weights.device(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Real(weights) => weights.dtype(),
            Self::Quantized(_) => DType::F32,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MaybeQuantizedLinear {
    Real(candle_nn::Linear),
    Quantized(candle_qnn::Linear),
}

impl Module for MaybeQuantizedLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Real(module) => module.forward(xs),
            Self::Quantized(module) => module.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MaybeQuantizedEmbedding {
    Real(candle_nn::Embedding),
    Quantized(candle_qnn::Embedding),
}

impl MaybeQuantizedEmbedding {
    pub fn new(in_vocab_size: usize, dim: usize, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let emb = match vb {
            MaybeQuantizedVarBuilder::Real(weights) => {
                MaybeQuantizedEmbedding::Real(candle_nn::embedding(in_vocab_size, dim, weights)?)
            }
            MaybeQuantizedVarBuilder::Quantized(weights) => MaybeQuantizedEmbedding::Quantized(
                candle_transformers::quantized_nn::Embedding::new(in_vocab_size, dim, weights)?,
            ),
        };
        Ok(emb)
    }

    pub fn embeddings(&self) -> &Tensor {
        match self {
            MaybeQuantizedEmbedding::Real(weights) => weights.embeddings(),
            MaybeQuantizedEmbedding::Quantized(weights) => weights.embeddings(),
        }
    }

    pub fn hidden_size(&self) -> Result<usize> {
        let size = match self {
            MaybeQuantizedEmbedding::Real(weights) => weights.hidden_size(),
            MaybeQuantizedEmbedding::Quantized(weights) => weights.embeddings().dim(1)?,
        };
        Ok(size)
    }
}

impl Module for MaybeQuantizedEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Real(module) => module.forward(xs),
            Self::Quantized(module) => module.forward(xs),
        }
    }
}

pub fn linear(
    in_d: usize,
    out_d: usize,
    bias: bool,
    vb: MaybeQuantizedVarBuilder,
) -> Result<MaybeQuantizedLinear> {
    let output_linear = match vb {
        MaybeQuantizedVarBuilder::Real(weights) => {
            if bias {
                MaybeQuantizedLinear::Real(candle_nn::linear(in_d, out_d, weights)?)
            } else {
                MaybeQuantizedLinear::Real(candle_nn::linear_no_bias(in_d, out_d, weights)?)
            }
        }
        MaybeQuantizedVarBuilder::Quantized(weights) => {
            MaybeQuantizedLinear::Quantized(candle_qnn::linear_b(in_d, out_d, bias, weights)?)
        }
    };
    Ok(output_linear)
}

pub fn linear_from(
    weight: MaybeQuantizedWeight,
    bias: Option<Tensor>,
) -> Result<MaybeQuantizedLinear> {
    let layer = match weight {
        MaybeQuantizedWeight::Real(w) => {
            MaybeQuantizedLinear::Real(candle_nn::Linear::new(w, bias))
        }
        MaybeQuantizedWeight::Quantized(w) => {
            MaybeQuantizedLinear::Quantized(candle_qnn::Linear::from_arc(w, bias)?)
        }
    };
    Ok(layer)
}
