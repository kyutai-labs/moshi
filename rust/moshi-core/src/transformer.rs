// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// Implements various modules for transformers with support for both quantized and unquantized forwards
// Main differences between quantized and unquantized execution:
// 1. For quantized models' attention `matmul_dtype`` converts intermediate activations to BF16 for
// more efficient matmuls
// 2. Quantized tensors cannot be easily split (regarding cross attention and QKV proj weights)
// 3. Linear and Quantized linear layers are two different types
use crate::nn::{
    linear, linear_from, matmul_dtype, MaybeQuantizedLinear, MaybeQuantizedVarBuilder,
};
use crate::streaming::{StreamMask, StreamTensor, StreamingModule};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};

use crate::kv_cache::KvCache;
use candle::Context;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub causal: bool,
    pub norm_first: bool,
    pub bias_ff: bool,
    pub bias_attn: bool,
    pub layer_scale: Option<f64>,
    pub positional_embedding: PositionalEmbedding,
    pub use_conv_block: bool,
    pub cross_attention: Option<(CrossAttentionGating, crate::NormType, Option<usize>)>,
    pub conv_kernel_size: usize,
    pub use_conv_bias: bool,
    pub gating: Option<candle_nn::Activation>,
    pub norm: crate::NormType,
    pub context: usize,
    pub max_period: usize,
    pub max_seq_len: usize,

    pub kv_repeat: usize,
    pub dim_feedforward: usize,
    pub conv_layout: bool,

    #[serde(default)]
    pub shared_cross_attn: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub enum PositionalEmbedding {
    Rope,
    Sin,
    None,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CrossAttentionGating {
    // Configure Type of gating used at the output of vision cross-attention layers
    Normal,
    ConstantGatedTanh,
    ConstantGatedSigmoid,
    ConditionalGatedTanh,
    ConditionalGatedSigmoid,
    ConditionalGatedSigmoidLearnableBias,
    ConditionalGatedTanhLearnableBias,
}

#[derive(Debug, Clone)]
pub enum CaSrc {
    // Input to cross-attention to handle cases where the
    // cross-attention source can be shared across timesteps and/or layers
    // either a single tensor (has yet to be projected)
    // or pre-computed K,V projections;
    Tokens(Tensor),
    KeysValues((Tensor, Tensor)),
}

#[derive(Debug, Clone)]
pub struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    pub fn new(d_model: usize, _init: f64, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let scale = vb.get_unquantized(d_model, "scale")?;
        Ok(Self { scale })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale)
    }
}

#[derive(Debug, Clone)]
pub enum XaGate {
    // Optional gating at the output of a cross-attention layer
    // Normal: No gating | Identity
    Normal,
    // ConstantGated: Multiply by a scalar
    ConstantGated {
        alpha: Tensor,
    },
    // ConditionalGated: Pass the input x through a small MLP;
    // The output yields a vector of scales (one for each channel)
    // that x is then multiplied by
    ConditionalGated {
        in_proj: MaybeQuantizedLinear,
        out_proj: MaybeQuantizedLinear,
        activation: candle_nn::init::NonLinearity,
        learnable_bias: bool,
    },
}

impl XaGate {
    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let gating_cfg =
            cfg.cross_attention.map(|v| v.0).context("no cross-attention specified")?;
        match gating_cfg {
            // no gating
            CrossAttentionGating::Normal => Ok(Self::Normal),
            // constant (per-layer parameter) with tanh activation
            CrossAttentionGating::ConstantGatedTanh => {
                let alpha = vb.get_unquantized((1, 1, 1), "alpha")?.tanh()?;
                Ok(Self::ConstantGated { alpha })
            }
            // constant (per-layer parameter) with sigmoid activation
            CrossAttentionGating::ConstantGatedSigmoid => {
                let alpha =
                    candle_nn::ops::sigmoid(&(vb.get_unquantized((1, 1, 1), "alpha")? - 4.0)?)?;
                Ok(Self::ConstantGated { alpha })
            }
            // input conditional (small MLP) with tanh or sigmoid act
            CrossAttentionGating::ConditionalGatedTanh
            | CrossAttentionGating::ConditionalGatedSigmoid
            | CrossAttentionGating::ConditionalGatedSigmoidLearnableBias
            | CrossAttentionGating::ConditionalGatedTanhLearnableBias => {
                let dim = cfg.d_model;
                let hidden_dims = (0.125 * dim as f32).floor() as usize;
                let learnable_bias = matches!(
                    gating_cfg,
                    CrossAttentionGating::ConditionalGatedSigmoidLearnableBias
                        | CrossAttentionGating::ConditionalGatedTanhLearnableBias
                );
                let in_proj = linear(dim, hidden_dims, false, vb.pp("alpha.0"))?;
                let out_proj = linear(hidden_dims, dim, learnable_bias, vb.pp("alpha.2"))?;
                let activation = match gating_cfg {
                    CrossAttentionGating::ConditionalGatedTanh
                    | CrossAttentionGating::ConditionalGatedTanhLearnableBias => {
                        candle_nn::init::NonLinearity::Tanh
                    }
                    CrossAttentionGating::ConditionalGatedSigmoid
                    | CrossAttentionGating::ConditionalGatedSigmoidLearnableBias => {
                        candle_nn::init::NonLinearity::Sigmoid
                    }
                    _ => candle::bail!("Invalid cross-attention config specified."),
                };
                Ok(Self::ConditionalGated { in_proj, out_proj, activation, learnable_bias })
            }
        }
    }
}

impl Module for XaGate {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Normal => Ok(xs.clone()),
            Self::ConstantGated { alpha } => xs.broadcast_mul(alpha),
            Self::ConditionalGated { in_proj, out_proj, activation, learnable_bias } => {
                let alpha = xs.apply(in_proj)?.relu()?.apply(out_proj)?;
                let alpha = match (activation, learnable_bias) {
                    (candle_nn::init::NonLinearity::Tanh, _) => alpha.tanh(),
                    (candle_nn::init::NonLinearity::Sigmoid, true) => {
                        candle_nn::ops::sigmoid(&alpha)
                    }
                    (candle_nn::init::NonLinearity::Sigmoid, false) => {
                        candle_nn::ops::sigmoid(&(alpha - 4.0)?)
                    }
                    _ => candle::bail!("Invalid non-linearity specified in cross-attention gating"),
                };
                xs * alpha?
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMultiheadCrossAttention {
    //Cross-attention modules. Q and KV projections are separate
    // because x (speech tokens) and ca_src (cross-attention source) can have
    // different dimensions
    in_proj_q: MaybeQuantizedLinear,
    in_proj_kv: MaybeQuantizedLinear,
    out_proj: MaybeQuantizedLinear,
    kv_repeat: usize,
    num_heads: usize,
    gate: XaGate,
    span: tracing::Span,
}

impl StreamingMultiheadCrossAttention {
    pub fn new(
        cfg: &Config,
        vb: MaybeQuantizedVarBuilder,
        gate_vb: Option<MaybeQuantizedVarBuilder>,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_kv = cfg.num_heads / cfg.kv_repeat;
        let out_kv_dim = num_kv * (embed_dim / cfg.num_heads);
        let out_dim = embed_dim + 2 * out_kv_dim;
        // Case 1 (legacy): A  single in_proj; i.e., both x and ca_src *must* have
        // the same number of dims this is only possible for non-quantized tensors though
        // as we will need to split Q/KV weights down the line even when they have the same
        // shape since they take different inputs
        let (in_proj_q, in_proj_kv) = if vb.contains_key("in_proj_weight") {
            match &vb {
                MaybeQuantizedVarBuilder::Quantized(_) => candle::bail!("Quantized cross-attention layers require a separate in_proj_weight_q and in_proj_weight_kv"),
                MaybeQuantizedVarBuilder::Real(weights) => {
                    let in_proj_weight = weights.get((out_dim, embed_dim), "in_proj_weight")?;
                    let in_proj_weight_q = in_proj_weight.narrow(0, 0, embed_dim)?;
                    let in_proj_weight_kv = in_proj_weight.narrow(0, embed_dim, 2 * out_kv_dim)?;
                    let (in_proj_bias_q, in_proj_bias_kv) = if cfg.bias_attn {
                        let b = weights.get(out_dim, "in_proj_bias")?;
                        let in_proj_bias_q = b.narrow(0, 0, embed_dim)?;
                        let in_proj_bias_kv = b.narrow(0, embed_dim, 2 * out_kv_dim)?;
                        (Some(in_proj_bias_q), Some(in_proj_bias_kv))
                    } else {
                        (None, None)
                    };
                    (MaybeQuantizedLinear::Real(candle_nn::Linear::new(in_proj_weight_q, in_proj_bias_q)),
                    MaybeQuantizedLinear::Real(candle_nn::Linear::new(in_proj_weight_kv, in_proj_bias_kv)))

            }
        }
        } else {
            // Case 2: Separate projections for query (x) and kv (ca_src)
            let kv_in_dim = match cfg.cross_attention.map(|v| v.2) {
                None => candle::bail!("cfg.cross_attention is None in cross_attention module"),
                Some(d) => match d {
                    None | Some(0) => embed_dim,
                    Some(dd) => dd,
                },
            };
            let in_proj_weight_q = vb.get((embed_dim, embed_dim), "in_proj_weight_q")?;
            let in_proj_weight_kv = vb.get((2 * out_kv_dim, kv_in_dim), "in_proj_weight_kv")?;

            // Biases are always unquantized
            let (in_proj_bias_q, in_proj_bias_kv) = if cfg.bias_attn {
                (
                    Some(vb.get_unquantized(embed_dim, "in_proj_bias_q")?),
                    Some(vb.get_unquantized(2 * out_kv_dim, "in_proj_bias_kv")?),
                )
            } else {
                (None, None)
            };

            // Finally, we can build the actual linear layers
            let in_proj_q = linear_from(in_proj_weight_q, in_proj_bias_q)?;
            let in_proj_kv = linear_from(in_proj_weight_kv, in_proj_bias_kv)?;
            (in_proj_q, in_proj_kv)
        };

        let out_proj = linear(embed_dim, embed_dim, cfg.bias_attn, vb.pp("out_proj"))?;
        let gate = match gate_vb {
            None => XaGate::new(cfg, vb.pp("gate"))?,
            Some(layer_gate_vb) => XaGate::new(cfg, layer_gate_vb)?,
        };
        Ok(Self {
            in_proj_q,
            in_proj_kv,
            out_proj,
            kv_repeat: cfg.kv_repeat,
            num_heads: cfg.num_heads,
            gate,
            span: tracing::span!(tracing::Level::TRACE, "mhca"),
        })
    }

    pub fn is_quantized(&self) -> bool {
        match self.in_proj_q {
            MaybeQuantizedLinear::Quantized(_) => true,
            MaybeQuantizedLinear::Real(_) => false,
        }
    }

    pub fn compute_kv(&self, ca_src: &CaSrc) -> Result<(Tensor, Tensor)> {
        // this is used twice:
        // in the standard forward pass of the cross-attention
        // for vision models, after loading an image we can precompute its KV projections
        // as the image is constant across multiple timesteps
        match ca_src {
            CaSrc::KeysValues(cakv) => Ok(cakv.clone()),
            CaSrc::Tokens(xs) => {
                let kv = xs.apply(&self.in_proj_kv)?;
                let (ca_b, ca_t, ca_dim) = kv.dims3()?;
                let head_dim = ca_dim / (2 * self.num_heads);
                let kv = kv.reshape((ca_b, ca_t, 2, (), head_dim))?;
                // convert to correct float point type for quantized models
                let kv =
                    if self.is_quantized() { kv.to_dtype(matmul_dtype(xs.device()))? } else { kv };
                let k = kv.i((.., .., 0))?;
                let v = kv.i((.., .., 1))?;
                let k = k.transpose(1, 2)?.contiguous()?; // b,h,k,d
                let v = v.transpose(1, 2)?.contiguous()?; // b,h,k,d
                Ok((k, v))
            }
        }
    }

    pub fn forward(&self, xs: &Tensor, ca_src: &CaSrc, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        if self.kv_repeat != 1 {
            candle::bail!("only kv-repeat = 1 is supported")
        }
        let (b, t, hd) = xs.dims3()?;
        let head_dim = hd / self.num_heads;
        // time_dim = 1, layout: b,t,h,d
        let q = xs.apply(&self.in_proj_q)?;
        let original_dtype = q.dtype();
        let q = q.reshape((b, t, self.num_heads, head_dim))?;
        let q = if self.is_quantized() { q.to_dtype(matmul_dtype(xs.device()))? } else { q };
        let (k, v) = self.compute_kv(ca_src)?;
        // qk_layer_norm = None
        // kv_repeat = 1, otherwise we would need repeat_kv
        let q = q.transpose(1, 2)?.contiguous()?; // b,h,t,d

        let pre_ws = q.matmul(&k.t()?)?; // b,h,t,k
        let pre_ws = (pre_ws * (head_dim as f64).powf(-0.5))?;

        let pre_ws = match mask {
            None => pre_ws,
            Some(mask) => pre_ws.broadcast_add(mask)?,
        };

        let ws = candle_nn::ops::softmax_last_dim(&pre_ws)?; // b,h,t,k
        let xs = ws.matmul(&v)?; // b,h,t,d
        let xs = xs
            .transpose(1, 2)? // b,t,h,d
            .reshape((b, t, hd))?
            .to_dtype(original_dtype)?
            .apply(&self.out_proj)?
            .apply(&self.gate)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct Rope {
    sin: Tensor,
    cos: Tensor,
}

impl Rope {
    pub fn apply_rotary_emb(&self, qk: &Tensor) -> Result<Tensor> {
        let qk_dtype = qk.dtype();
        candle_nn::rotary_emb::rope_i(&qk.to_dtype(DType::F32)?, &self.cos, &self.sin)?
            .to_dtype(qk_dtype)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, theta: f32, dev: &Device) -> Result<Self> {
        let inv_freq: Vec<_> =
            (0..dim).step_by(2).map(|i| 1f32 / theta.powf(i as f32 / dim as f32)).collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        Ok(Self { inv_freq })
    }

    pub fn rope(&self, pos: &Tensor) -> Result<Rope> {
        let t = pos.to_dtype(DType::F32)?;
        let freqs = match *t.dims() {
            [d] => t.reshape((d, 1))?.matmul(&self.inv_freq)?,
            [b, d] => t.reshape((b * d, 1))?.matmul(&self.inv_freq)?.reshape((b, d, ()))?,
            _ => candle::bail!("Invalid shape for rotary embedding {pos:?}"),
        };
        Ok(Rope { sin: freqs.sin()?, cos: freqs.cos()? })
    }
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

#[derive(Debug, Clone)]
pub struct StreamingMultiheadAttention {
    // Self-attention with KV Cache
    in_proj: MaybeQuantizedLinear,
    out_proj: MaybeQuantizedLinear,
    kv_repeat: usize,
    num_heads: usize,
    context: usize,
    kv_cache: KvCache,
    use_flash_attn: bool,
    span: tracing::Span,
}

impl StreamingMultiheadAttention {
    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_kv = cfg.num_heads / cfg.kv_repeat;
        let out_dim = embed_dim + 2 * num_kv * (embed_dim / cfg.num_heads);
        let in_proj_weight = vb.get((out_dim, embed_dim), "in_proj_weight")?;
        let in_proj_bias =
            if cfg.bias_attn { Some(vb.get_unquantized(out_dim, "in_proj_bias")?) } else { None };
        let in_proj = linear_from(in_proj_weight, in_proj_bias)?;
        let out_proj = linear(embed_dim, embed_dim, cfg.bias_attn, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            out_proj,
            kv_repeat: cfg.kv_repeat,
            num_heads: cfg.num_heads,
            context: cfg.context,
            kv_cache: KvCache::new(2, cfg.context),
            use_flash_attn: false,
            span: tracing::span!(tracing::Level::TRACE, "mha"),
        })
    }

    pub fn is_quantized(&self) -> bool {
        match self.in_proj {
            MaybeQuantizedLinear::Quantized(_) => true,
            MaybeQuantizedLinear::Real(_) => false,
        }
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        rope: Option<&Rope>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        if self.kv_repeat != 1 {
            candle::bail!("only kv-repeat = 1 is supported")
        }
        let (b, t, hd) = xs.dims3()?;
        let head_dim = hd / self.num_heads;
        // time_dim = 1, layout: b,t,h,d
        let qkv = xs.apply(&self.in_proj)?.reshape((b, t, 3, self.num_heads, head_dim))?;
        let original_dtype = qkv.dtype();
        let qkv = if self.is_quantized() { qkv.to_dtype(matmul_dtype(xs.device()))? } else { qkv };
        let q = qkv.i((.., .., 0))?;
        let k = qkv.i((.., .., 1))?;
        let v = qkv.i((.., .., 2))?;
        // qk_layer_norm = None
        // kv_repeat = 1, otherwise we would need repeat_kv
        let mut q = q.transpose(1, 2)?.contiguous()?; // b,h,t,d
        let mut k = k.transpose(1, 2)?.contiguous()?; // b,h,k,d
        let v = v.transpose(1, 2)?.contiguous()?; // b,h,k,d
        if let Some(rope) = rope.as_ref() {
            q = rope.apply_rotary_emb(&q)?;
            k = rope.apply_rotary_emb(&k)?;
        }

        let (k, v) = { self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)? };
        // The KV cache keeps all the data at the moment, we want to trim
        // down the part that comes from the cache to at most context to
        // be coherent with the mask shape we provide.
        let k_len = k.dim(2)?;
        let k_target_len = t + usize::min(self.context, k_len - t);
        let (k, v) = if k_target_len < k_len {
            let k = k.narrow(2, k_len - k_target_len, k_target_len)?;
            let v = v.narrow(2, k_len - k_target_len, k_target_len)?;
            (k, v)
        } else {
            (k.clone(), v.clone())
        };

        let xs = if q.dtype() == DType::BF16 && self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, mask.is_some())?.transpose(1, 2)?
        } else {
            let pre_ws = q.matmul(&k.t()?)?; // b,h,t,k
            let pre_ws = (pre_ws * (head_dim as f64).powf(-0.5))?;

            let pre_ws = match mask {
                None => pre_ws,
                Some(mask) => pre_ws.broadcast_add(mask)?,
            };

            let ws = candle_nn::ops::softmax_last_dim(&pre_ws)?; // b,h,t,k
            ws.matmul(&v)? // b,h,t,d
        };

        let xs = xs
            .transpose(1, 2)? // b,t,h,d
            .reshape((b, t, hd))?
            .to_dtype(original_dtype)?
            .apply(&self.out_proj)?;
        Ok(xs)
    }

    pub fn reset_kv_cache(&mut self) {
        self.kv_cache.reset()
    }

    pub fn set_kv_cache(&mut self, kv_cache: KvCache) {
        self.kv_cache = kv_cache
    }
}

#[derive(Debug, Clone)]
pub enum Mlp {
    //Feed Forward layers
    NoGating {
        linear1: MaybeQuantizedLinear,
        linear2: MaybeQuantizedLinear,
    },
    Gating {
        linear_in: MaybeQuantizedLinear,
        linear_out: MaybeQuantizedLinear,
        activation: candle_nn::Activation,
    },
}

impl Mlp {
    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;
        match cfg.gating {
            None => {
                let linear1 = linear(d_model, cfg.dim_feedforward, cfg.bias_ff, vb.pp("linear1"))?;
                let linear2 = linear(cfg.dim_feedforward, d_model, cfg.bias_ff, vb.pp("linear2"))?;
                Ok(Self::NoGating { linear1, linear2 })
            }
            Some(activation) => {
                let vb = vb.pp("gating");
                let hidden = if cfg.dim_feedforward == 4 * d_model {
                    11 * d_model / 4
                } else {
                    2 * cfg.dim_feedforward / 3
                };
                let linear_in = linear(d_model, 2 * hidden, cfg.bias_ff, vb.pp("linear_in"))?;
                let linear_out = linear(hidden, d_model, cfg.bias_ff, vb.pp("linear_out"))?;
                Ok(Self::Gating { linear_in, linear_out, activation })
            }
        }
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::NoGating { linear1, linear2 } => xs.apply(linear1)?.gelu_erf()?.apply(linear2),
            Self::Gating { linear_in, linear_out, activation } => {
                let xs = xs.apply(linear_in)?;
                let (b, t, _) = xs.dims3()?;
                let xs = xs.reshape((b, t, 2, ()))?;
                let xs = (xs.i((.., .., 0))?.apply(activation)? * xs.i((.., .., 1))?)?;
                xs.apply(linear_out)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub(crate) alpha: Tensor,
    pub(crate) eps: f32,
}

impl RmsNorm {
    pub fn new(d_model: usize, eps: f32, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let alpha = vb.get_unquantized((1, 1, d_model), "alpha")?.reshape(d_model)?;
        Ok(Self { alpha, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(xs, &self.alpha, self.eps)
    }
}

#[derive(Debug, Clone)]
pub struct LayerNorm {
    inner: candle_nn::LayerNorm,
}

impl LayerNorm {
    pub fn new(d_model: usize, eps: f32, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let bias = vb.get_unquantized(d_model, "bias")?;
        let alpha = if vb.contains_key("alpha") {
            vb.get_unquantized((1, 1, d_model), "alpha")?.reshape(d_model)?
        } else {
            vb.get_unquantized(d_model, "weight")?.reshape(d_model)?
        };
        let inner = candle_nn::LayerNorm::new(alpha, bias, eps as f64);
        Ok(Self { inner })
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub enum Norm {
    LayerNorm(LayerNorm),
    RmsNorm(RmsNorm),
}

impl Norm {
    pub fn new(d_model: usize, cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let norm = Self::new_shortcut(d_model, cfg.norm, vb)?;
        Ok(norm)
    }

    pub fn new_shortcut(
        d_model: usize,
        typ: crate::NormType,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let norm = match typ {
            crate::NormType::LayerNorm => {
                let norm = LayerNorm::new(d_model, 1e-5, vb)?;
                Self::LayerNorm(norm)
            }
            crate::NormType::RmsNorm => {
                let norm = RmsNorm::new(d_model, 1e-8, vb)?;
                Self::RmsNorm(norm)
            }
        };
        Ok(norm)
    }
}

impl Module for Norm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(m) => m.forward(xs),
            Self::RmsNorm(m) => m.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTransformerLayer {
    self_attn: StreamingMultiheadAttention,
    mlp: Mlp,
    norm1: Norm,
    norm2: Norm,
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
    cross_attn: Option<(Norm, StreamingMultiheadCrossAttention)>,
    norm_first: bool,
    span: tracing::Span,
}

impl StreamingTransformerLayer {
    pub fn new(
        cfg: &Config,
        vb: MaybeQuantizedVarBuilder,
        shared_ca_vb: Option<MaybeQuantizedVarBuilder>,
    ) -> Result<Self> {
        if cfg.use_conv_block {
            candle::bail!("conv-block is not supported")
        }
        let d_model = cfg.d_model;
        let mlp = Mlp::new(cfg, vb.clone())?;
        let norm1 = Norm::new(d_model, cfg, vb.pp("norm1"))?;
        let norm2 = Norm::new(d_model, cfg, vb.pp("norm2"))?;
        let layer_scale_1 = match cfg.layer_scale {
            None => None,
            Some(ls) => {
                let ls = LayerScale::new(d_model, ls, vb.pp("layer_scale_1"))?;
                Some(ls)
            }
        };
        let layer_scale_2 = match cfg.layer_scale {
            None => None,
            Some(ls) => {
                let ls = LayerScale::new(d_model, ls, vb.pp("layer_scale_2"))?;
                Some(ls)
            }
        };
        let self_attn = StreamingMultiheadAttention::new(cfg, vb.pp("self_attn"))?;
        let cross_attn = match cfg.cross_attention.map(|v| v.1) {
            Some(norm_type) => {
                let norm_cross = Norm::new_shortcut(d_model, norm_type, vb.pp("norm_cross"))?;
                let cross_attn = match shared_ca_vb {
                    None => {
                        StreamingMultiheadCrossAttention::new(cfg, vb.pp("cross_attention"), None)?
                    }
                    Some(shared_vb) => StreamingMultiheadCrossAttention::new(
                        cfg,
                        shared_vb.pp("cross_attention"),
                        Some(vb.pp("cross_attention.gate")),
                    )?,
                };
                Some((norm_cross, cross_attn))
            }
            None => None,
        };
        Ok(Self {
            self_attn,
            mlp,
            norm1,
            norm2,
            layer_scale_1,
            layer_scale_2,
            cross_attn,
            norm_first: cfg.norm_first,
            span: tracing::span!(tracing::Level::TRACE, "transformer-layer"),
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        rope: Option<&Rope>,
        ca_src: Option<&CaSrc>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        if !self.norm_first {
            candle::bail!("only norm_first = true is supported")
        }
        let norm1 = xs.apply(&self.norm1)?;
        let xs = (xs
            + self.self_attn.forward(&norm1, rope, mask)?.apply(&self.layer_scale_1.as_ref())?)?;

        let xs = match (self.cross_attn.as_mut(), ca_src) {
            (Some((norm_cross, cross_attn)), Some(ca_src)) => {
                let residual = &xs;
                let xs = xs.apply(norm_cross)?;
                (residual + cross_attn.forward(&xs, ca_src, None)?)?
            }
            _ => xs,
        };

        let xs =
            (&xs + xs.apply(&self.norm2)?.apply(&self.mlp)?.apply(&self.layer_scale_2.as_ref()))?;
        Ok(xs)
    }

    pub fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
    }

    pub fn set_kv_cache(&mut self, kv_cache: KvCache) {
        self.self_attn.set_kv_cache(kv_cache);
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTransformer {
    // Main transformer
    layers: Vec<StreamingTransformerLayer>,
    positional_embedding: PositionalEmbedding,
    max_period: usize,
    causal: bool,
    num_heads: usize,
    context: usize,
    last_reset_pos: Vec<usize>,
    rope: Option<RotaryEmbedding>,
}

impl StreamingTransformer {
    pub fn new(cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let rope = match cfg.positional_embedding {
            PositionalEmbedding::Rope => {
                let rope = RotaryEmbedding::new(
                    cfg.d_model / cfg.num_heads,
                    cfg.max_period as f32,
                    vb.device(),
                )?;
                Some(rope)
            }
            PositionalEmbedding::None | PositionalEmbedding::Sin => None,
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            // Also send weights of first layer as only it contains the KQV proj weights
            // for shared cross-attention layers
            let shared_vb = if cfg.shared_cross_attn { Some(vb_l.pp(0)) } else { None };
            let layer = StreamingTransformerLayer::new(cfg, vb_l.pp(layer_idx), shared_vb)?;
            layers.push(layer)
        }
        Ok(Self {
            layers,
            positional_embedding: cfg.positional_embedding,
            max_period: cfg.max_period,
            causal: cfg.causal,
            num_heads: cfg.num_heads,
            context: cfg.context,
            last_reset_pos: vec![],
            rope,
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        self.forward_ca(xs, None)
    }

    fn current_seq_len(&self) -> usize {
        self.layers[0].self_attn.kv_cache.current_seq_len()
    }

    pub fn forward_ca(&mut self, xs: &Tensor, ca_src: Option<&CaSrc>) -> Result<Tensor> {
        let (b, t, c) = xs.dims3()?;
        if !self.causal {
            candle::bail!("only causal mode is supported")
        }
        if self.last_reset_pos.is_empty() {
            self.last_reset_pos.resize(b, 0);
        }
        let current_seq_len = self.current_seq_len();
        // We will extract at most "context" from the kv_cache.
        // Note that the mask still discards the values that are before context as this can happen
        // when t > context.
        let mask = {
            // mask shape should be b, h, t, k
            // self.layers[0].self_attn.kv_cache.attn_mask(t, xs.device())?;
            // let mask = mask.broadcast_left((b, self.num_heads))?;
            let ks = self.layers[0].self_attn.kv_cache.positions(t);
            let min_ks = ks.iter().min().context("no positions, is t == 0?")?;
            if t == 1 && self.last_reset_pos.iter().all(|v| v <= min_ks) {
                // No need for a mask here.
                None
            } else {
                let mut mask = Vec::with_capacity(b * self.num_heads * t * ks.len());
                for &last_reset_pos in self.last_reset_pos.iter() {
                    for t_pos in 0..t {
                        let t_pos = t_pos + current_seq_len;
                        for &k_pos in ks.iter() {
                            let m = if last_reset_pos <= k_pos
                                && k_pos <= t_pos
                                && t_pos <= k_pos + self.context
                            {
                                0f32
                            } else {
                                f32::NEG_INFINITY
                            };
                            mask.push(m);
                        }
                    }
                }
                let mask = Tensor::from_vec(mask, (b, 1, t, ks.len()), xs.device())?
                    .to_dtype(xs.dtype())?
                    .expand((b, self.num_heads, t, ks.len()))?;
                Some(mask)
            }
        };
        // pos is used for the rotary embeddings, as these are relative embeddings there is no need
        // to adjust them for the actual position using last_reset_pos.
        let pos =
            Tensor::arange(current_seq_len as u32, (current_seq_len + t) as u32, xs.device())?;
        let rope = match self.rope {
            Some(ref rope) => Some(rope.rope(&pos)?),
            None => None,
        };
        let mut xs = match self.positional_embedding {
            PositionalEmbedding::Rope | PositionalEmbedding::None => xs.clone(),
            PositionalEmbedding::Sin => {
                let dev = xs.device();
                let theta = self.max_period as f32;
                let half_dim = c / 2;
                let positions = pos.unsqueeze(1)?.to_dtype(DType::F32)?;
                let inv_freq: Vec<_> = (0..half_dim)
                    .map(|i| 1f32 / theta.powf(i as f32 / (half_dim - 1) as f32))
                    .collect();
                let inv_freq_len = inv_freq.len();
                let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
                let freqs = positions.broadcast_mul(&inv_freq)?;
                let pos_emb = Tensor::cat(&[freqs.cos()?, freqs.sin()?], D::Minus1)?;
                xs.broadcast_add(&pos_emb)?
            }
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, rope.as_ref(), ca_src, mask.as_ref())?
        }
        Ok(xs)
    }

    pub fn maybe_precompute_ca_kv(&self, ca_src: Option<CaSrc>) -> Result<Option<CaSrc>> {
        let ca_src = match ca_src {
            None => None,
            Some(CaSrc::KeysValues(_)) => ca_src,
            Some(tokens) => {
                if self.layers.is_empty() {
                    Some(tokens)
                } else {
                    match &self.layers[0].cross_attn {
                        None => Some(tokens),
                        Some((_, ca_module)) => {
                            let (k, v) = ca_module.compute_kv(&tokens)?;
                            Some(CaSrc::KeysValues((k, v)))
                        }
                    }
                }
            }
        };
        Ok(ca_src)
    }

    pub fn copy_state(&mut self, from: &Self) -> Result<()> {
        if self.layers.len() != from.layers.len() {
            candle::bail!("cannot copy kv-caches as the transformers have different depths")
        }
        self.last_reset_pos = from.last_reset_pos.clone();
        self.layers
            .iter_mut()
            .zip(from.layers.iter())
            .for_each(|(v, w)| v.set_kv_cache(w.self_attn.kv_cache.clone()));
        Ok(())
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize, batch_size: usize) -> Result<()> {
        if self.last_reset_pos.is_empty() {
            self.last_reset_pos.resize(batch_size, 0);
        }
        if batch_idx >= self.last_reset_pos.len() {
            candle::bail!("batch_idx {} is out of bounds for last_reset_pos", batch_idx)
        }
        self.last_reset_pos[batch_idx] = self.current_seq_len();
        Ok(())
    }
}

impl StreamingModule for StreamingTransformer {
    fn reset_state(&mut self) {
        self.last_reset_pos.clear();
        self.layers.iter_mut().for_each(|v| v.reset_kv_cache())
    }

    fn step(&mut self, xs: &StreamTensor, _: &StreamMask) -> Result<StreamTensor> {
        // TODO: Use the StreamMask
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => Ok(StreamTensor::from_tensor(self.forward(xs)?)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectedTransformer {
    // Projected transformer with unquantized projection
    transformer: StreamingTransformer,
    input_proj: Option<MaybeQuantizedLinear>,
    output_projs: Vec<Option<MaybeQuantizedLinear>>,
    conv_layout: bool,
    span: tracing::Span,
}

impl ProjectedTransformer {
    pub fn new(
        input_dim: usize,
        output_dims: &[usize],
        cfg: &Config,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::new(cfg, vb.pp("transformer"))?;
        let input_proj = if input_dim == cfg.d_model {
            None
        } else {
            let l = linear(input_dim, cfg.d_model, false, vb.pp("input_proj"))?;
            Some(l)
        };
        let mut output_projs = Vec::with_capacity(output_dims.len());
        let vb_o = vb.pp("output_projs");
        for (i, &output_dim) in output_dims.iter().enumerate() {
            let output_proj = if output_dim == cfg.d_model {
                None
            } else {
                let l = linear(cfg.d_model, output_dim, false, vb_o.pp(i))?;
                Some(l)
            };
            output_projs.push(output_proj)
        }
        Ok(Self {
            transformer,
            input_proj,
            output_projs,
            conv_layout: cfg.conv_layout,
            span: tracing::span!(tracing::Level::TRACE, "proj-transformer"),
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let _enter = self.span.enter();
        let xs = if self.conv_layout { xs.transpose(1, 2)? } else { xs.clone() };
        let xs = xs.apply(&self.input_proj.as_ref())?;
        let xs = self.transformer.forward(&xs)?;
        let mut ys = Vec::with_capacity(self.output_projs.len());
        for output_proj in self.output_projs.iter() {
            let ys_ = xs.apply(&output_proj.as_ref())?;
            let ys_ = if self.conv_layout { ys_.transpose(1, 2)? } else { ys_ };
            ys.push(ys_)
        }
        Ok(ys)
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize, batch_size: usize) -> Result<()> {
        self.transformer.reset_batch_idx(batch_idx, batch_size)
    }
}

impl StreamingModule for ProjectedTransformer {
    fn reset_state(&mut self) {
        self.transformer.reset_state()
    }

    fn step(&mut self, xs: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
        let xs = xs.apply(&|x: &Tensor| {
            if self.conv_layout {
                x.transpose(1, 2)
            } else {
                Ok(x.clone())
            }
        })?;
        let xs = xs.apply(&self.input_proj.as_ref())?;
        let xs = self.transformer.step(&xs, m)?;
        let ys = xs.apply(&self.output_projs[0].as_ref())?;
        ys.apply(&|y: &Tensor| {
            if self.conv_layout {
                y.transpose(1, 2)
            } else {
                Ok(y.clone())
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum Transformer {
    Standard(ProjectedTransformer),
    Batched(crate::batched_transformer::ProjectedTransformer),
}

impl StreamingModule for Transformer {
    fn reset_state(&mut self) {
        match self {
            Transformer::Standard(t) => t.reset_state(),
            Transformer::Batched(t) => t.reset_state(),
        }
    }

    fn step(&mut self, xs: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
        match self {
            Transformer::Standard(t) => t.step(xs, m),
            Transformer::Batched(t) => t.step(xs, m),
        }
    }
}

impl Transformer {
    pub fn new(
        batch_size: Option<usize>,
        dim: usize,
        cfg: &Config,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let transformer = match batch_size {
            Some(batch_size) => {
                let transformer = crate::batched_transformer::ProjectedTransformer::new(
                    dim,
                    &[dim],
                    batch_size,
                    cfg,
                    MaybeQuantizedVarBuilder::Real(vb),
                )?;
                Transformer::Batched(transformer)
            }
            None => {
                let transformer = ProjectedTransformer::new(
                    dim,
                    &[dim],
                    cfg,
                    MaybeQuantizedVarBuilder::Real(vb),
                )?;
                Transformer::Standard(transformer)
            }
        };
        Ok(transformer)
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Vec<Tensor>> {
        match self {
            Transformer::Standard(t) => t.forward(xs),
            Transformer::Batched(t) => t.forward(xs, &().into()),
        }
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize, batch_size: usize) -> Result<()> {
        match self {
            Transformer::Standard(t) => t.reset_batch_idx(batch_idx, batch_size),
            Transformer::Batched(t) => t.reset_batch_idx(batch_idx),
        }
    }
}
