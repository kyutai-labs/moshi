// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::streaming::{StreamTensor, StreamingModule};
use crate::transformer::{get_mask, CrossAttention, PositionalEmbedding, RotaryEmbedding};

use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_transformers::quantized_nn::{layer_norm, linear_b, Linear};
use candle_transformers::quantized_var_builder::VarBuilder;
use std::sync::Arc;

pub use crate::transformer::Config;

#[derive(Debug, Clone)]
pub struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    pub fn new(d_model: usize, _init: f64, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(d_model, "scale")?.dequantize(vb.device())?;
        Ok(Self { scale })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    kv_repeat: usize,
    num_heads: usize,
    context: usize,
    neg_inf: Tensor,
    rope: Option<Arc<RotaryEmbedding>>,
    kv_cache: candle_nn::kv_cache::KvCache,
    use_kv_cache: bool,
    pos: usize,
    span: tracing::Span,
}

fn matmul_dtype(device: &candle::Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

impl StreamingMultiheadAttention {
    pub fn new(rope: &Option<Arc<RotaryEmbedding>>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_kv = cfg.num_heads / cfg.kv_repeat;
        let out_dim = embed_dim + 2 * num_kv * (embed_dim / cfg.num_heads);
        let in_proj_weight = vb.get((out_dim, embed_dim), "in_proj_weight")?;
        let in_proj_bias = if cfg.bias_attn {
            Some(vb.get(out_dim, "in_proj_bias")?.dequantize(vb.device())?)
        } else {
            None
        };
        let in_proj = Linear::from_arc(in_proj_weight, in_proj_bias)?;
        let out_proj = linear_b(embed_dim, embed_dim, cfg.bias_attn, vb.pp("out_proj"))?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, vb.device())?;
        Ok(Self {
            in_proj,
            out_proj,
            rope: rope.clone(),
            kv_repeat: cfg.kv_repeat,
            num_heads: cfg.num_heads,
            context: cfg.context,
            neg_inf,
            kv_cache: candle_nn::kv_cache::KvCache::new(2, cfg.max_seq_len),
            use_kv_cache: true,
            pos: 0,
            span: tracing::span!(tracing::Level::TRACE, "mha"),
        })
    }

    pub fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        if self.kv_repeat != 1 {
            candle::bail!("only kv-repeat = 1 is supported")
        }
        let (b, t, hd) = xs.dims3()?;
        let head_dim = hd / self.num_heads;
        // time_dim = 1, layout: b,t,h,d
        let qkv = xs.apply(&self.in_proj)?.reshape((b, t, 3, self.num_heads, head_dim))?;
        let original_dtype = qkv.dtype();
        let qkv = qkv.to_dtype(matmul_dtype(xs.device()))?;
        let q = qkv.i((.., .., 0))?;
        let k = qkv.i((.., .., 1))?;
        let v = qkv.i((.., .., 2))?;
        // qk_layer_norm = None
        // kv_repeat = 1, otherwise we would need repeat_kv
        let mut q = q.transpose(1, 2)?.contiguous()?; // b,h,t,d
        let mut k = k.transpose(1, 2)?.contiguous()?; // b,h,k,d
        let v = v.transpose(1, 2)?.contiguous()?; // b,h,k,d
        if let Some(rope) = &self.rope {
            q = rope.apply_rotary_emb(&q, self.pos)?;
            k = rope.apply_rotary_emb(&k, self.pos)?;
        }

        let (k, v) = if self.use_kv_cache {
            self.pos += k.dim(2)?;
            self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?
        } else {
            (k, v)
        };
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

        let pre_ws = q.matmul(&k.t()?)?; // b,h,t,k
        let pre_ws = (pre_ws * (head_dim as f64).powf(-0.5))?;

        let pre_ws = match mask {
            None => pre_ws,
            Some(mask) => {
                let mask = mask.broadcast_left((b, self.num_heads))?;
                let neg_inf = self.neg_inf.broadcast_as(pre_ws.shape())?;
                mask.where_cond(&neg_inf, &pre_ws)?
            }
        };

        let ws = candle_nn::ops::softmax_last_dim(&pre_ws)?; // b,h,t,k
        let xs = ws.matmul(&v)?; // b,h,t,d
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

    pub fn set_kv_cache(&mut self, kv_cache: candle_nn::kv_cache::KvCache) {
        self.kv_cache = kv_cache
    }
}

#[derive(Debug, Clone)]
pub struct StreamingMultiheadCrossAttention {
    in_proj_q: Linear,
    in_proj_k: Linear,
    in_proj_v: Linear,
    out_proj: Linear,
    kv_repeat: usize,
    num_heads: usize,
    neg_inf: Tensor,
    span: tracing::Span,
}

impl StreamingMultiheadCrossAttention {
    pub fn new(_ca: CrossAttention, _cfg: &Config, _vb: VarBuilder) -> Result<Self> {
        candle::bail!("cross-attn is not supported at the moment")
    }

    pub fn forward(&self, xs: &Tensor, ca_src: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        if self.kv_repeat != 1 {
            candle::bail!("only kv-repeat = 1 is supported")
        }
        let (b, t, hd) = xs.dims3()?;
        let head_dim = hd / self.num_heads;
        // time_dim = 1, layout: b,t,h,d
        let q = xs.apply(&self.in_proj_q)?;
        let k = ca_src.apply(&self.in_proj_k)?;
        let v = ca_src.apply(&self.in_proj_v)?;
        let (ca_b, ca_t, ca_dim) = k.dims3()?;
        let q = q.reshape((b, t, self.num_heads, head_dim))?;
        let k = k.reshape((ca_b, ca_t, ca_dim / head_dim, head_dim))?;
        let v = v.reshape((ca_b, ca_t, ca_dim / head_dim, head_dim))?;
        // qk_layer_norm = None
        // kv_repeat = 1, otherwise we would need repeat_kv
        let q = q.transpose(1, 2)?.contiguous()?; // b,h,t,d
        let k = k.transpose(1, 2)?.contiguous()?; // b,h,k,d
        let v = v.transpose(1, 2)?.contiguous()?; // b,h,k,d

        let pre_ws = q.matmul(&k.t()?)?; // b,h,t,k
        let pre_ws = (pre_ws * (head_dim as f64).powf(-0.5))?;

        let pre_ws = match mask {
            None => pre_ws,
            Some(mask) => {
                let mask = mask.broadcast_left((b, self.num_heads))?;
                let neg_inf = self.neg_inf.broadcast_as(pre_ws.shape())?;
                mask.where_cond(&neg_inf, &pre_ws)?
            }
        };

        let ws = candle_nn::ops::softmax_last_dim(&pre_ws)?; // b,h,t,k
        let xs = ws.matmul(&v)?; // b,h,t,d
        let xs = xs
            .transpose(1, 2)? // b,t,h,d
            .reshape((b, t, hd))?
            .apply(&self.out_proj)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub enum Mlp {
    NoGating { linear1: Linear, linear2: Linear },
    Gating { linear_in: Linear, linear_out: Linear, activation: candle_nn::Activation },
}

impl Mlp {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;
        match cfg.gating {
            None => {
                let linear1 =
                    linear_b(d_model, cfg.dim_feedforward, cfg.bias_ff, vb.pp("linear1"))?;
                let linear2 =
                    linear_b(cfg.dim_feedforward, d_model, cfg.bias_ff, vb.pp("linear2"))?;
                Ok(Self::NoGating { linear1, linear2 })
            }
            Some(activation) => {
                let vb = vb.pp("gating");
                let hidden = if cfg.dim_feedforward == 4 * d_model {
                    11 * d_model / 4
                } else {
                    2 * cfg.dim_feedforward / 3
                };
                // TODO: Maybe use bias_ff here?
                let linear_in = linear_b(d_model, 2 * hidden, false, vb.pp("linear_in"))?;
                let linear_out = linear_b(hidden, d_model, false, vb.pp("linear_out"))?;
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
pub enum Norm {
    LayerNorm(candle_nn::LayerNorm),
    RmsNorm(crate::transformer::RmsNorm),
}

pub fn rms_norm(d_model: usize, eps: f32, vb: VarBuilder) -> Result<crate::transformer::RmsNorm> {
    let alpha = vb.get((1, 1, d_model), "alpha")?.dequantize(vb.device())?.reshape(d_model)?;
    Ok(crate::transformer::RmsNorm { alpha, eps })
}

impl Norm {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = match cfg.norm {
            crate::NormType::LayerNorm => {
                let norm = layer_norm(cfg.d_model, 1e-5, vb)?;
                Self::LayerNorm(norm)
            }
            crate::NormType::RmsNorm => {
                let norm = rms_norm(cfg.d_model, 1e-8, vb)?;
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
    cross_attn: Option<(candle_nn::LayerNorm, StreamingMultiheadCrossAttention)>,
    norm_first: bool,
    span: tracing::Span,
}

impl StreamingTransformerLayer {
    pub fn new(rope: &Option<Arc<RotaryEmbedding>>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        if cfg.use_conv_block {
            candle::bail!("conv-block is not supported")
        }
        let d_model = cfg.d_model;
        let mlp = Mlp::new(cfg, vb.clone())?;
        let (norm1, norm2) = match cfg.norm {
            crate::NormType::LayerNorm => {
                let norm1 = layer_norm(d_model, 1e-5, vb.pp("norm1"))?;
                let norm2 = layer_norm(d_model, 1e-5, vb.pp("norm2"))?;
                (Norm::LayerNorm(norm1), Norm::LayerNorm(norm2))
            }
            crate::NormType::RmsNorm => {
                let norm1 = rms_norm(cfg.d_model, 1e-8, vb.pp("norm1"))?;
                let norm2 = rms_norm(cfg.d_model, 1e-8, vb.pp("norm2"))?;
                (Norm::RmsNorm(norm1), Norm::RmsNorm(norm2))
            }
        };
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
        let self_attn = StreamingMultiheadAttention::new(rope, cfg, vb.pp("self_attn"))?;
        let cross_attn = match cfg.cross_attention {
            Some(ca) => {
                let norm_cross = layer_norm(cfg.d_model, 1e-5, vb.pp("norm_cross"))?;
                let cross_attn =
                    StreamingMultiheadCrossAttention::new(ca, cfg, vb.pp("cross_attention"))?;
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
        ca_src: Option<&Tensor>,

        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        if !self.norm_first {
            candle::bail!("only norm_first = true is supported")
        }
        let norm1 = xs.apply(&self.norm1)?;
        let xs =
            (xs + self.self_attn.forward(&norm1, mask)?.apply(&self.layer_scale_1.as_ref())?)?;

        let xs = match (&self.cross_attn, ca_src) {
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
        self.self_attn.reset_kv_cache()
    }

    pub fn set_kv_cache(&mut self, kv_cache: candle_nn::kv_cache::KvCache) {
        self.self_attn.set_kv_cache(kv_cache)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTransformer {
    layers: Vec<StreamingTransformerLayer>,
    context: usize,
    positional_embedding: PositionalEmbedding,
    max_period: usize,
}

impl StreamingTransformer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_l = vb.pp("layers");
        let rope = match cfg.positional_embedding {
            PositionalEmbedding::Rope => {
                let rope = RotaryEmbedding::new(
                    cfg.d_model / cfg.num_heads,
                    cfg.max_seq_len,
                    cfg.max_period as f32,
                    vb.device(),
                )?;
                Some(Arc::new(rope))
            }
            PositionalEmbedding::None | PositionalEmbedding::Sin => None,
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let layer = StreamingTransformerLayer::new(&rope, cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        Ok(Self {
            layers,
            context: cfg.context,
            positional_embedding: cfg.positional_embedding,
            max_period: cfg.max_period,
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        self.forward_ca(xs, None)
    }

    pub fn forward_ca(&mut self, xs: &Tensor, ca_src: Option<&Tensor>) -> Result<Tensor> {
        let (_b, t, c) = xs.dims3()?;
        // We will extract at most "context" from the kv_cache.
        // Note that the mask will discard the values that are before context.
        let pos = self.layers[0].self_attn.kv_cache.k_cache().current_seq_len().min(self.context);
        let mask =
            if t == 1 { None } else { Some(get_mask(t, pos + t, self.context, xs.device())?) };
        let mut xs = match self.positional_embedding {
            PositionalEmbedding::Rope | PositionalEmbedding::None => xs.clone(),
            PositionalEmbedding::Sin => {
                let dev = xs.device();
                let theta = self.max_period as f32;
                let half_dim = c / 2;
                let positions = Tensor::arange(pos as u32, (pos + t) as u32, dev)?
                    .unsqueeze(1)?
                    .to_dtype(DType::F32)?;
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
            xs = layer.forward(&xs, ca_src, mask.as_ref())?
        }
        Ok(xs)
    }

    pub fn copy_state(&mut self, from: &Self) -> Result<()> {
        if self.layers.len() != from.layers.len() {
            candle::bail!("cannot copy kv-caches as the transformers have different depths")
        }
        self.layers
            .iter_mut()
            .zip(from.layers.iter())
            .for_each(|(v, w)| v.set_kv_cache(w.self_attn.kv_cache.clone()));
        Ok(())
    }
}

impl StreamingModule for StreamingTransformer {
    fn reset_state(&mut self) {
        self.layers.iter_mut().for_each(|v| v.reset_kv_cache())
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => Ok(StreamTensor::from_tensor(self.forward(xs)?)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectedTransformer {
    transformer: StreamingTransformer,
    input_proj: Option<Linear>,
    output_projs: Vec<Option<Linear>>,
    conv_layout: bool,
    span: tracing::Span,
}

impl ProjectedTransformer {
    pub fn new(
        input_dim: usize,
        output_dims: &[usize],
        cfg: &Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::new(cfg, vb.pp("transformer"))?;
        let input_proj = if input_dim == cfg.d_model {
            None
        } else {
            let l = linear_b(input_dim, cfg.d_model, false, vb.pp("input_proj"))?;
            Some(l)
        };
        let mut output_projs = Vec::with_capacity(output_dims.len());
        let vb_o = vb.pp("output_projs");
        for (i, &output_dim) in output_dims.iter().enumerate() {
            let output_proj = if output_dim == cfg.d_model {
                None
            } else {
                let l = linear_b(cfg.d_model, output_dim, false, vb_o.pp(i))?;
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
}

impl StreamingModule for ProjectedTransformer {
    fn reset_state(&mut self) {
        self.transformer.reset_state()
    }

    fn step(&mut self, xs: &StreamTensor) -> Result<StreamTensor> {
        let xs = xs.apply(&|x: &Tensor| {
            if self.conv_layout {
                x.transpose(1, 2)
            } else {
                Ok(x.clone())
            }
        })?;
        let xs = xs.apply(&self.input_proj.as_ref())?;
        let xs = self.transformer.step(&xs)?;
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
