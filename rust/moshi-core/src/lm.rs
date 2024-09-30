// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::transformer;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;

thread_local! {
    pub static VERBOSE: bool = {
        match std::env::var("MIMI_VERBOSE") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DepFormerConfig {
    pub transformer: transformer::Config,
    pub num_slices: usize,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub transformer: transformer::Config,
    pub depformer: Option<DepFormerConfig>,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
}

impl Config {
    // /lustre/scwpod02/client/kyutai/alex/mimi_exp/xps/af78657c/outputs/hyperparams.json
    // Update 2024-03-19: Sin embeddings -> None, RmsNorm fix, scale factor 4.125
    // Update 2024-05-02: split text_vocab_size into text_in_vocab_size and text_out_vocab_size.
    // embeddings.
    pub fn v0_1() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 4096,
            num_heads: 32,
            num_layers: 32,
            dim_feedforward: 4096 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 3000,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: crate::NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = transformer::Config {
            d_model: 1024,
            num_heads: 16,
            num_layers: 6,
            dim_feedforward: 1024 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 8,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: Some(candle_nn::Activation::Silu),
            norm: crate::NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::None,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = DepFormerConfig { num_slices: 8, transformer: depformer_cfg };
        Self {
            transformer: lm_cfg,
            depformer: Some(depformer_cfg),
            audio_vocab_size: 2049,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32000,
            audio_codebooks: 8,
        }
    }

    pub fn v0_1_streaming(num_slices: usize) -> Self {
        let mut s = Self::v0_1();
        s.audio_codebooks = 16;
        if let Some(depformer) = s.depformer.as_mut() {
            depformer.num_slices = num_slices;
            depformer.transformer.context = num_slices;
        }
        s
    }

    // /lustre/scwpod02/client/kyutai/neilz/mimi_exp/xps/6bbe4692/outputs/hyperparams.json
    pub fn tts_v0_1() -> Self {
        let lm_cfg = transformer::Config {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 4096 * 2, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 4096,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: None,
            norm: crate::NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = transformer::Config {
            d_model: 1024,
            num_heads: 16,
            num_layers: 6,
            dim_feedforward: 1024 * 4, // dim * hidden_scale
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 4096,
            max_period: 10000,
            use_conv_block: false,
            use_conv_bias: true,
            cross_attention: None,
            gating: None,
            norm: crate::NormType::LayerNorm,
            positional_embedding: transformer::PositionalEmbedding::Sin,
            conv_layout: false,
            conv_kernel_size: 3,
            kv_repeat: 1,
            max_seq_len: 4096,
        };
        let depformer_cfg = DepFormerConfig { num_slices: 16, transformer: depformer_cfg };
        Self {
            transformer: lm_cfg,
            depformer: Some(depformer_cfg),
            audio_vocab_size: 2050,
            text_in_vocab_size: 32001,
            text_out_vocab_size: 32001,
            audio_codebooks: 16,
        }
    }
}

#[derive(Debug, Clone)]
struct DepFormerSlice {
    transformer: transformer::StreamingTransformer,
    // Note that the embedding for the first slice does not have the same dimension as the
    // embedding for the other slices as it takes a text token as input rather than an audio token.
    emb: candle_nn::Embedding,
    linear_in: candle_nn::Linear,  // depformer_in.{idx}
    linear_out: candle_nn::Linear, // linears.{idx}
}

impl DepFormerSlice {
    fn new(
        in_vocab_size: usize,
        out_vocab_size: usize,
        main_transformer_dim: usize,
        cfg: &transformer::Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim = cfg.d_model;
        let transformer = transformer::StreamingTransformer::new(cfg, vb.pp("transformer"))?;
        let emb = candle_nn::embedding(in_vocab_size, dim, vb.pp("emb"))?;
        let linear_in = candle_nn::linear_no_bias(main_transformer_dim, dim, vb.pp("linear_in"))?;
        let linear_out = candle_nn::linear_no_bias(dim, out_vocab_size, vb.pp("linear_out"))?;
        Ok(Self { transformer, emb, linear_in, linear_out })
    }
}

#[derive(Debug, Clone)]
pub struct DepFormer {
    first_eos_step_idx: Option<usize>,
    audio_eos_token: u32,
    audio_padding_token: u32,
    slices: Vec<DepFormerSlice>,
}

impl DepFormer {
    pub fn new(
        text_vocab_size: usize,
        audio_vocab_size: usize,
        main_transformer_dim: usize,
        cfg: &DepFormerConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut slices = Vec::with_capacity(cfg.num_slices);
        for slice_idx in 0..cfg.num_slices {
            let in_vs = if slice_idx == 0 { text_vocab_size } else { audio_vocab_size };
            // The depformer cannot predict the audio padding token.
            let slice = DepFormerSlice::new(
                in_vs,
                audio_vocab_size - 1, // The depformer cannot emit an audio padding token.
                main_transformer_dim,
                &cfg.transformer,
                vb.pp(slice_idx),
            )?;
            slices.push(slice)
        }
        Ok(Self {
            slices,
            audio_eos_token: audio_vocab_size as u32 - 2,
            audio_padding_token: audio_vocab_size as u32 - 1,
            first_eos_step_idx: None,
        })
    }

    /// Run a transformer sampling step, getting a token id per codebook.
    /// - `xs` is the previous layer hidden state.
    pub fn sample(
        &mut self,
        step_idx: usize,
        xs: &Tensor,
        text_token: Option<u32>,
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<u32>> {
        use crate::streaming::StreamingModule;
        let dev = xs.device();
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        for slice_idx in 0..self.slices.len() {
            // Token shifting by 2.
            if slice_idx == 0 {
                self.slices[slice_idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(slice_idx);
                rhs[0].transformer.copy_state(&lhs[slice_idx - 1].transformer)?
            }
            let slice = &mut self.slices[slice_idx];
            let xs = slice.linear_in.forward(xs)?;
            let xs = match last_token {
                Some(last_token) => {
                    // TODO(laurent): this seems a bit weird, does it mean that the acoustic delay
                    // is somewhat hardcoded to ([1] + [0] * 7) * 2?
                    // Maybe step_idx > 1 should be step_idx > 0 instead?
                    let last_token = if slice_idx < 2 || slice_idx == 9 || step_idx > 1 {
                        last_token
                    } else {
                        self.audio_padding_token
                    };
                    let token_id = Tensor::from_vec(vec![last_token], (1, 1), dev)?;
                    let token_emb = slice.emb.forward(&token_id)?;
                    xs.broadcast_add(&token_emb)?
                }
                None => xs,
            };
            let xs = slice.transformer.forward(&xs)?;
            let logits = xs.apply(&slice.linear_out)?;
            let logits = match logits.dim(0)? {
                1 => logits.i((0, 0))?,
                b_size => candle::bail!("unexpected batch size {b_size}"),
            };
            let token = self.sample_maybe_postpone_eos(step_idx, &logits, lp)?;
            if VERBOSE.with(|v| *v) {
                println!("sampled {token} logits {slice_idx}:\n{logits}");
            }
            last_token = Some(token);
            tokens.push(token)
        }
        Ok(tokens)
    }

    fn sample_maybe_postpone_eos(
        &mut self,
        step_idx: usize,
        logits: &Tensor,
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<u32> {
        let token = lp.sample(logits)?;
        let token = if token == self.audio_eos_token {
            let first_eos_step_idx = match self.first_eos_step_idx {
                Some(step_idx) => step_idx,
                None => {
                    self.first_eos_step_idx = Some(step_idx);
                    step_idx
                }
            };
            // We enforce that the generation continues for a couple steps so that we don't
            // stop in the middle of a word.
            if step_idx >= first_eos_step_idx + 5 {
                token
            } else {
                lp.sample_f(logits, |pr| pr[self.audio_eos_token as usize] = 1e-9)?
            }
        } else {
            token
        };
        Ok(token)
    }

    // Sampling with classifier free guidance.
    pub fn sample_cfg(
        &mut self,
        step_idx: usize,
        xs: &Tensor,
        cfg_alpha: f64,
        text_token: Option<u32>,
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<u32>> {
        use crate::streaming::StreamingModule;
        let dev = xs.device();
        let mut tokens = Vec::with_capacity(self.slices.len());
        let mut last_token = text_token;
        for slice_idx in 0..self.slices.len() {
            // Token shifting by 2.
            if slice_idx == 0 {
                self.slices[slice_idx].transformer.reset_state();
            } else {
                let (lhs, rhs) = self.slices.split_at_mut(slice_idx);
                rhs[0].transformer.copy_state(&lhs[slice_idx - 1].transformer)?
            }
            let slice = &mut self.slices[slice_idx];
            let xs = slice.linear_in.forward(xs)?;
            let xs = match last_token {
                Some(last_token) => {
                    // TODO(laurent): same as above, this seems to hardcode some delays and it's not
                    // obvious why it should be step_idx > 1 rather than step_idx > 0.
                    let last_token = if slice_idx < 2 || step_idx > 1 {
                        last_token
                    } else {
                        self.audio_padding_token
                    };
                    let token_id = Tensor::from_vec(vec![last_token], (1, 1), dev)?;
                    let token_emb = slice.emb.forward(&token_id)?;
                    xs.broadcast_add(&token_emb)?
                }
                None => xs,
            };
            let xs = slice.transformer.forward(&xs)?;
            let logits = xs.apply(&slice.linear_out)?;
            let logits = match logits.dim(0)? {
                2 => ((logits.i((0, 0))? * cfg_alpha)? - (logits.i((1, 0))? * (cfg_alpha - 1.))?)?,
                b_size => candle::bail!("unexpected batch size {b_size}"),
            };
            let token = if slice_idx == 0 {
                self.sample_maybe_postpone_eos(step_idx, &logits, lp)?
            } else {
                lp.sample(&logits)?
            };
            last_token = Some(token);
            tokens.push(token)
        }
        Ok(tokens)
    }
}

#[derive(Debug, Clone)]
pub struct Lm {
    pub transformer: transformer::StreamingTransformer,
    pub text_emb: candle_nn::Embedding,
    pub audio_embs: Vec<candle_nn::Embedding>,
    pub text_linear: candle_nn::Linear,
    pub out_norm: transformer::Norm,
    pub depformer: Option<DepFormer>,
    pub dtype: DType,
}

impl Lm {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.transformer.d_model;
        let depformer = match &cfg.depformer {
            None => None,
            Some(depformer_cfg) => {
                let depformer = DepFormer::new(
                    cfg.text_in_vocab_size,
                    cfg.audio_vocab_size,
                    d_model,
                    depformer_cfg,
                    vb.pp("depformer"),
                )?;
                Some(depformer)
            }
        };
        let text_emb = candle_nn::embedding(cfg.text_in_vocab_size, d_model, vb.pp("text_emb"))?;
        let out_norm = transformer::Norm::new(d_model, &cfg.transformer, vb.pp("out_norm"))?;
        let text_linear =
            candle_nn::linear_no_bias(d_model, cfg.text_out_vocab_size, vb.pp("text_linear"))?;
        let transformer =
            transformer::StreamingTransformer::new(&cfg.transformer, vb.pp("transformer"))?;
        let vb_e = vb.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.audio_codebooks);
        for i in 0..cfg.audio_codebooks {
            let emb = candle_nn::embedding(cfg.audio_vocab_size, d_model, vb_e.pp(i))?;
            audio_embs.push(emb)
        }
        Ok(Self {
            transformer,
            text_emb,
            text_linear,
            audio_embs,
            out_norm,
            depformer,
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        self.text_emb.embeddings().device()
    }

    pub fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
    ) -> candle::Result<(Tensor, Tensor)> {
        if VERBOSE.with(|v| *v) {
            print!("text_ids ");
            if let Some(text_ids) = text_ids.as_ref() {
                let text_ids = text_ids.flatten_all()?.to_vec1::<u32>()?;
                println!("{text_ids:?}");
            } else {
                println!("none")
            }
            print!("audio_ids ");
            for audio_id in audio_ids.iter() {
                if let Some(audio_id) = audio_id {
                    let audio_id = audio_id.flatten_all()?.to_vec1::<u32>()?;
                    print!(" {audio_id:?}");
                } else {
                    print!(" none")
                }
            }
            println!();
        }
        let mut emb = match text_ids.as_ref() {
            Some(text_ids) => text_ids.apply(&self.text_emb)?,
            None => {
                let device = self.text_emb.embeddings().device();
                Tensor::zeros((1, 1, self.text_emb.hidden_size()), self.dtype, device)?
            }
        };

        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(audio_ids) = audio_ids {
                let e = audio_ids.apply(audio_emb)?;
                emb = (emb + e)?
            }
        }
        let ys = self.transformer.forward(&emb)?;
        let ys = ys.apply(&self.out_norm)?;
        let logits = ys.apply(&self.text_linear)?;
        if VERBOSE.with(|v| *v) {
            println!("logits:\n{logits}");
        }
        Ok((logits, ys))
    }

    pub fn forward_ca(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
        ca_src: &Tensor,
    ) -> candle::Result<(Tensor, Tensor)> {
        let b_size = ca_src.dim(0)?;
        let mut emb = match text_ids {
            Some(text_ids) => text_ids.apply(&self.text_emb)?,
            None => {
                let device = self.text_emb.embeddings().device();
                Tensor::zeros((b_size, 1, self.text_emb.hidden_size()), self.dtype, device)?
            }
        };
        for (audio_emb, audio_ids) in self.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(audio_ids) = audio_ids {
                let e = audio_ids.apply(audio_emb)?;
                emb = emb.broadcast_add(&e)?
            }
        }
        let ys = self.transformer.forward_ca(&emb, Some(ca_src))?;
        let ys = ys.apply(&self.out_norm)?;
        let logits = ys.apply(&self.text_linear)?;
        Ok((logits, ys))
    }
}

#[derive(Debug, Clone)]
pub enum LmModel {
    Lm(Lm),
    QuantizedLm(crate::quantized_lm::Lm),
}

impl LmModel {
    pub fn forward(
        &mut self,
        text_ids: Option<Tensor>,
        audio_ids: Vec<Option<Tensor>>,
    ) -> candle::Result<(Tensor, Tensor)> {
        match self {
            Self::Lm(m) => m.forward(text_ids, audio_ids),
            Self::QuantizedLm(m) => m.forward(text_ids, audio_ids),
        }
    }

    pub fn depformer_sample(
        &mut self,
        step_idx: usize,
        xs: &Tensor,
        text_token: Option<u32>,
        lp: &mut candle_transformers::generation::LogitsProcessor,
    ) -> Result<Option<Vec<u32>>> {
        let sample = match self {
            Self::Lm(m) => match &mut m.depformer {
                None => None,
                Some(m) => {
                    let sample = m.sample(step_idx, xs, text_token, lp)?;
                    Some(sample)
                }
            },
            Self::QuantizedLm(m) => match &mut m.depformer {
                None => None,
                Some(m) => {
                    let sample = m.sample(step_idx, xs, text_token, lp)?;
                    Some(sample)
                }
            },
        };
        Ok(sample)
    }

    pub fn device(&self) -> &Device {
        match self {
            Self::Lm(m) => m.device(),
            Self::QuantizedLm(m) => m.device(),
        }
    }
}

pub fn load<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    quantized: bool,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1();
    let model = if quantized {
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_file, dev)?;
        let lm = crate::quantized_lm::Lm::new(&cfg, vb)?;
        LmModel::QuantizedLm(lm)
    } else {
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], dtype, dev)? };
        let lm = Lm::new(&cfg, vb)?;
        LmModel::Lm(lm)
    };
    Ok(model)
}

pub fn load_streaming<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming(8);
    let is_gguf = model_file.as_ref().extension().map_or(false, |v| v == "gguf");
    let lm = if is_gguf {
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_file, dev)?;
        let lm = crate::quantized_lm::Lm::new(&cfg, vb)?;
        LmModel::QuantizedLm(lm)
    } else {
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], dtype, dev)? };
        let lm = Lm::new(&cfg, vb)?;
        LmModel::Lm(lm)
    };
    Ok(lm)
}

pub fn load_streaming_both_ways<P: AsRef<std::path::Path>>(
    model_file: P,
    dtype: DType,
    dev: &Device,
) -> Result<LmModel> {
    let cfg = Config::v0_1_streaming(16);
    let is_gguf = model_file.as_ref().extension().map_or(false, |v| v == "gguf");
    let lm = if is_gguf {
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(model_file, dev)?;
        let lm = crate::quantized_lm::Lm::new(&cfg, vb)?;
        LmModel::QuantizedLm(lm)
    } else {
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], dtype, dev)? };
        let lm = Lm::new(&cfg, vb)?;
        LmModel::Lm(lm)
    };
    Ok(lm)
}
