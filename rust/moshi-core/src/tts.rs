// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::transformer::CaSrc;
use candle::{Context, DType, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use candle_transformers::models::t5;

pub struct Config {
    pub t5: t5::Config,
    pub lm: crate::lm::Config,
    pub mimi: crate::mimi::Config,
    pub max_duration_s: f64,
    pub speaker_cond_duration_s: f64,
    pub max_speakers: usize,
}

impl Config {
    pub fn v0_1(t5: t5::Config) -> Self {
        let lm = crate::lm::Config::tts_v0_1();
        let mimi = crate::mimi::Config::v0_1(None);
        Self { t5, lm, mimi, max_duration_s: 60., speaker_cond_duration_s: 4., max_speakers: 5 }
    }

    pub fn v0_2(t5: t5::Config) -> Self {
        let lm = crate::lm::Config::tts_v0_1();
        let mimi = crate::mimi::Config::v0_1(None);
        Self { t5, lm, mimi, max_duration_s: 60., speaker_cond_duration_s: 10., max_speakers: 2 }
    }
}

#[derive(Clone)]
pub struct Model {
    t5: t5::T5EncoderModel,
    pub lm: crate::lm::LmModel,
    speaker_cond: Option<(crate::mimi::Mimi, Linear)>,
    t5_proj: Linear,
    pub sample_rate: f64,
    frame_rate: f64,
    audio_vocab_size: u32,
    audio_codebooks: usize,
    pub max_duration_s: f64,
    max_speakers: usize,
    end_of_gen: Option<usize>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb_t5: VarBuilder,
        vb_lm: VarBuilder,
        vb_speaker_cond: Option<VarBuilder>,
    ) -> Result<Self> {
        let t5 = t5::T5EncoderModel::load(vb_t5, &cfg.t5)?;
        let speaker_cond = match vb_speaker_cond {
            None => None,
            Some(vb) => {
                let mimi = crate::mimi::Mimi::new(cfg.mimi.clone(), vb)?;
                let proj = linear_no_bias(
                    cfg.mimi.seanet.dimension,
                    cfg.lm.transformer.d_model,
                    vb_lm.pp("condition_provider.conditioners.speaker_wavs.output_proj"),
                )?;
                Some((mimi, proj))
            }
        };
        let t5_proj = {
            let name = if speaker_cond.is_some() {
                "condition_provider.conditioners.diarized_transcript_in_segment.output_proj"
            } else {
                "condition_provider.conditioners.transcript_in_segment.output_proj"
            };
            linear_no_bias(cfg.t5.d_model, cfg.lm.transformer.d_model, vb_lm.pp(name))?
        };
        let lm =
            crate::lm::LmModel::new(&cfg.lm, crate::nn::MaybeQuantizedVarBuilder::Real(vb_lm))?;
        Ok(Self {
            t5,
            lm,
            speaker_cond,
            t5_proj,
            sample_rate: cfg.mimi.sample_rate,
            frame_rate: cfg.mimi.frame_rate,
            audio_vocab_size: cfg.lm.audio_vocab_size as u32,
            audio_codebooks: cfg.lm.audio_codebooks,
            max_duration_s: cfg.max_duration_s,
            max_speakers: cfg.max_speakers,
            end_of_gen: None,
        })
    }
}

pub fn add_sin_embeddings(xs: &Tensor) -> Result<Tensor> {
    let target_dtype = xs.dtype();
    let (_b_size, seq_len, dim) = xs.dims3()?;
    let dev = xs.device();
    let half_dim = dim / 2;
    let positions =
        Tensor::arange(0u32, seq_len as u32, dev)?.unsqueeze(1)?.to_dtype(DType::F32)?;
    let inv_freq: Vec<_> =
        (0..half_dim).map(|i| 1f32 / 10000f32.powf(i as f32 / (half_dim - 1) as f32)).collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
    let freqs = positions.broadcast_mul(&inv_freq)?;
    let pos_emb = Tensor::cat(&[freqs.cos()?, freqs.sin()?], D::Minus1)?;
    let xs = xs.to_dtype(DType::F32)?.broadcast_add(&pos_emb)?;
    xs.to_dtype(target_dtype)
}

impl Model {
    pub fn conditions(
        &mut self,
        token_ids: &Tensor,
        speaker_pcm: Option<&Tensor>,
    ) -> Result<Tensor> {
        let t5_condition =
            self.t5.forward(token_ids)?.to_dtype(candle::DType::BF16)?.apply(&self.t5_proj)?;
        let conditions = match speaker_pcm {
            None => t5_condition,
            Some(speaker_pcm) => {
                let sc = match self.speaker_cond.as_mut() {
                    None => candle::bail!("speaker_pcm specified without a speaker-cond model"),
                    Some((mimi, proj)) => mimi
                        .encode_pre_quantize(speaker_pcm)?
                        .t()?
                        .to_dtype(candle::DType::BF16)?
                        .apply(proj)?,
                };
                let z = sc.zeros_like()?;
                let mut c1 = vec![&t5_condition, &sc];
                let mut c2 = vec![&t5_condition, &z];
                for _i in 0..self.max_speakers - 1 {
                    c1.push(&z);
                    c2.push(&z);
                }
                let c1 = Tensor::cat(&c1, 1)?;
                let c2 = Tensor::cat(&c2, 1)?;
                let xs = Tensor::cat(&[&c1, &c2], 0)?;
                add_sin_embeddings(&xs)?
            }
        };
        Ok(conditions)
    }

    pub fn sample(&mut self, conditions: &Tensor, cfg_alpha: f64) -> Result<Vec<Vec<u32>>> {
        let lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            299792458,
            candle_transformers::generation::Sampling::TopK { k: 100, temperature: 0.8 },
        );
        self.sample_lp(conditions, cfg_alpha, lp)
    }

    pub fn sample_lp(
        &mut self,
        conditions: &Tensor,
        cfg_alpha: f64,
        mut lp: candle_transformers::generation::LogitsProcessor,
    ) -> Result<Vec<Vec<u32>>> {
        let max_steps = (self.max_duration_s * self.frame_rate) as usize + 1;
        let audio_codebooks = self.audio_codebooks;
        let audio_vocab_size = self.audio_vocab_size;
        let mut audio_tokens: Vec<Vec<u32>> = vec![vec![u32::MAX; audio_codebooks]; max_steps + 2];
        let forced_audio_tokens = crate::lm::ForcedAudioTokens::new(
            /* acoustic_delay= */ 2,
            self.lm.audio_pad_token(),
            &[audio_codebooks],
        );
        let quantizer_bins = audio_vocab_size - 2; // 2048
        for step_idx in 0..(max_steps + 2) {
            let mut codes = Vec::with_capacity(audio_codebooks);
            for codebook in 0..audio_codebooks {
                let t = if codebook == 0 {
                    if step_idx == 0 {
                        audio_vocab_size - 1
                    } else {
                        audio_tokens[step_idx - 1][0]
                    }
                } else if step_idx <= 2 {
                    audio_vocab_size - 1
                } else {
                    audio_tokens[step_idx - 3][codebook]
                };
                let t = Tensor::new(&[t], conditions.device())?.unsqueeze(0)?;
                codes.push(Some(t))
            }
            let (_text_logits, ys) = self.lm.forward_ca(
                None,
                codes,
                &CaSrc::Tokens(conditions.clone()),
                None,
                &().into(),
            )?;
            let last_audio_tokens = if self.speaker_cond.is_some() {
                self.lm.depformer_sample_cfg(
                    &ys,
                    cfg_alpha,
                    None,
                    forced_audio_tokens.forced_tokens(step_idx),
                    &mut lp,
                )?
            } else {
                self.lm.depformer_sample(
                    &ys,
                    None,
                    forced_audio_tokens.forced_tokens(step_idx),
                    &mut lp,
                )?
            };
            let last_audio_tokens = last_audio_tokens.context("no depformer")?;
            for (c_idx, token) in last_audio_tokens.into_iter().enumerate() {
                if step_idx > 0 && token >= quantizer_bins && self.end_of_gen.is_none() {
                    // Continue generating for two steps to get the final acoustic tokens.
                    self.end_of_gen = Some(step_idx + 2)
                }
                let delay = if c_idx == 0 { 0 } else { 2 };
                audio_tokens[step_idx.saturating_sub(delay)][c_idx] = token
            }
            if Some(step_idx) == self.end_of_gen {
                break;
            }
        }
        Ok(audio_tokens)
    }
}
