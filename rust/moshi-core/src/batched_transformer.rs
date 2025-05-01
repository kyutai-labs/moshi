// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
use crate::nn::{
    linear, linear_from, matmul_dtype, MaybeQuantizedLinear, MaybeQuantizedVarBuilder,
};
use crate::streaming::{StreamMask, StreamTensor, StreamingModule};
use candle::{IndexOp, Module, Result, Tensor};

use crate::kv_cache::{
    IndicesAndMask, ScatteredCacheBuilder as KvCacheBuilder, ScatteredKvCache as KvCache,
};

use crate::transformer::{
    CaSrc, Config, LayerScale, PositionalEmbedding, Rope, RotaryEmbedding,
    StreamingMultiheadCrossAttention,
};

#[derive(Debug, Clone)]
pub struct StreamingMultiheadAttention {
    // Self-attention with KV Cache
    in_proj: MaybeQuantizedLinear,
    out_proj: MaybeQuantizedLinear,
    kv_repeat: usize,
    num_heads: usize,
    context: usize,
    kv_cache: KvCache,
    span: tracing::Span,
}

impl StreamingMultiheadAttention {
    pub fn new(
        cfg: &Config,
        builder: &KvCacheBuilder,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let head_dim = embed_dim / cfg.num_heads;
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
            kv_cache: builder.make_cache(num_kv, head_dim)?,
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
        iam: &IndicesAndMask,
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

        let (k, v) = { self.kv_cache.append(&k.contiguous()?, &v.contiguous()?, iam)? };
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

        let xs = {
            let pre_ws = q.matmul(&k.t()?)?; // b,h,t,k
            let pre_ws = (pre_ws * (head_dim as f64).powf(-0.5))?;
            let pre_ws = pre_ws.broadcast_add(iam.mask())?;
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
        builder: &KvCacheBuilder,
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
        let self_attn = StreamingMultiheadAttention::new(cfg, builder, vb.pp("self_attn"))?;
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
        iam: &IndicesAndMask,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        if !self.norm_first {
            candle::bail!("only norm_first = true is supported")
        }
        let norm1 = xs.apply(&self.norm1)?;
        let xs = (xs
            + self.self_attn.forward(&norm1, rope, iam)?.apply(&self.layer_scale_1.as_ref())?)?;

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

    pub fn set_kv_cache(&mut self, kv_cache: KvCache) {
        self.self_attn.set_kv_cache(kv_cache);
    }
}

#[derive(Debug, Clone)]
pub struct StreamingTransformer {
    // Main transformer
    layers: Vec<StreamingTransformerLayer>,
    positional_embedding: PositionalEmbedding,
    causal: bool,
    builder: KvCacheBuilder,
    rope: Option<RotaryEmbedding>,
}

impl StreamingTransformer {
    pub fn new(batch_size: usize, cfg: &Config, vb: MaybeQuantizedVarBuilder) -> Result<Self> {
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
        let builder = KvCacheBuilder::new(batch_size, cfg.context, vb.dtype(), vb.device())?;
        for layer_idx in 0..cfg.num_layers {
            // Also send weights of first layer as only it contains the KQV proj weights
            // for shared cross-attention layers
            let shared_vb = if cfg.shared_cross_attn { Some(vb_l.pp(0)) } else { None };
            let layer =
                StreamingTransformerLayer::new(cfg, &builder, vb_l.pp(layer_idx), shared_vb)?;
            layers.push(layer)
        }
        Ok(Self {
            layers,
            positional_embedding: cfg.positional_embedding,
            causal: cfg.causal,
            builder,
            rope,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, m: &StreamMask) -> Result<Tensor> {
        self.forward_ca(xs, None, m)
    }

    pub fn batch_size(&self) -> usize {
        self.builder.batch_size()
    }

    fn positions(&self) -> &[usize] {
        self.builder.positions()
    }

    pub fn forward_ca(
        &mut self,
        xs: &Tensor,
        ca_src: Option<&CaSrc>,
        m: &StreamMask,
    ) -> Result<Tensor> {
        let (b, t, _c) = xs.dims3()?;
        if b != self.batch_size() {
            candle::bail!("unexpected batch size {b} != {}", self.batch_size())
        }
        if !self.causal {
            candle::bail!("only causal mode is supported")
        }
        let iam = match m.cpu() {
            None => candle::bail!("batched-transformer expects a mask"),
            Some(m) => self.builder.indices_and_mask(t, m)?,
        };
        let rope = match self.rope {
            Some(ref rope) => {
                let pos = self
                    .positions()
                    .iter()
                    .map(|&v| (0..t).map(|i| (v + i) as u32).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                let pos = Tensor::new(pos, xs.device())?;
                Some(rope.rope(&pos)?)
            }
            None => None,
        };
        let mut xs = match self.positional_embedding {
            PositionalEmbedding::Rope | PositionalEmbedding::None => xs.clone(),
            PositionalEmbedding::Sin => candle::bail!("sin positional embedding is not supported"),
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, rope.as_ref(), ca_src, &iam)?
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
        self.layers
            .iter_mut()
            .zip(from.layers.iter())
            .for_each(|(v, w)| v.set_kv_cache(w.self_attn.kv_cache.clone()));
        Ok(())
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        if batch_idx >= self.batch_size() {
            candle::bail!("batch_idx {batch_idx} is out of bounds for last_reset_pos")
        }
        self.builder.reset_batch_index(batch_idx);
        Ok(())
    }
}

impl StreamingModule for StreamingTransformer {
    fn reset_state(&mut self) {
        self.builder.reset();
    }

    fn step(&mut self, xs: &StreamTensor, m: &StreamMask) -> Result<StreamTensor> {
        match xs.as_option() {
            None => Ok(StreamTensor::empty()),
            Some(xs) => Ok(StreamTensor::from_tensor(self.forward(xs, m)?)),
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
        batch_size: usize,
        cfg: &Config,
        vb: MaybeQuantizedVarBuilder,
    ) -> Result<Self> {
        let transformer = StreamingTransformer::new(batch_size, cfg, vb.pp("transformer"))?;
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

    pub fn forward(&mut self, xs: &Tensor, m: &StreamMask) -> Result<Vec<Tensor>> {
        let _enter = self.span.enter();
        let xs = if self.conv_layout { xs.transpose(1, 2)? } else { xs.clone() };
        let xs = xs.apply(&self.input_proj.as_ref())?;
        let xs = self.transformer.forward(&xs, m)?;
        let mut ys = Vec::with_capacity(self.output_projs.len());
        for output_proj in self.output_projs.iter() {
            let ys_ = xs.apply(&output_proj.as_ref())?;
            let ys_ = if self.conv_layout { ys_.transpose(1, 2)? } else { ys_ };
            ys.push(ys_)
        }
        Ok(ys)
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        self.transformer.reset_batch_idx(batch_idx)
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
