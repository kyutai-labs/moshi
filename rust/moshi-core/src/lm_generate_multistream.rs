// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use candle::{IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::transformer::CaSrc;

pub const UNGENERATED: u32 = u32::MAX;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub generated_audio_codebooks: usize,
    pub input_audio_codebooks: usize,
    pub audio_vocab_size: usize,
    pub acoustic_delay: usize,
    pub text_pad_token: u32,
    pub text_eop_token: u32,
    pub text_start_token: u32,
}

impl Config {
    pub fn v0_1() -> Self {
        Self {
            generated_audio_codebooks: 8,
            input_audio_codebooks: 8,
            audio_vocab_size: 2049,
            acoustic_delay: 2,
            text_eop_token: 0,
            text_pad_token: 3,
            text_start_token: 32000,
        }
    }

    pub fn v0_1_two_ways() -> Self {
        Self {
            generated_audio_codebooks: 16,
            input_audio_codebooks: 0,
            audio_vocab_size: 2049,
            acoustic_delay: 2,
            text_eop_token: 0,
            text_pad_token: 3,
            text_start_token: 32000,
        }
    }

    pub fn v0_1_one_way() -> Self {
        Self {
            generated_audio_codebooks: 8,
            input_audio_codebooks: 0,
            audio_vocab_size: 2049,
            acoustic_delay: 2,
            text_eop_token: 0,
            text_pad_token: 3,
            text_start_token: 32000,
        }
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn total_audio_codebooks(&self) -> usize {
        self.generated_audio_codebooks + self.input_audio_codebooks
    }
}

pub struct State {
    model: crate::lm::LmModel,
    audio_tokens: Vec<Vec<u32>>,
    text_tokens: Vec<u32>,
    audio_lp: LogitsProcessor,
    text_lp: LogitsProcessor,
    step_idx: usize,
    pad_mult: Option<f32>,
    // For repetition penalty, we provide the context len (in text tokens) and the penalty.
    repetition_penalty: Option<(usize, f32)>,
    forced_audio_tokens: crate::lm::ForcedAudioTokens,
    user_rating: u32,
    cfg_alpha: Option<f64>,
    config: Config,
}

impl State {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: crate::lm::LmModel,
        max_step_idx: usize,
        audio_lp: LogitsProcessor,
        text_lp: LogitsProcessor,
        pad_mult: Option<f32>,
        repetition_penalty: Option<(usize, f32)>,
        cfg_alpha: Option<f64>,
        config: Config,
    ) -> Self {
        let audio_tokens: Vec<Vec<u32>> = vec![
            vec![UNGENERATED; config.total_audio_codebooks()];
            max_step_idx + config.acoustic_delay
        ];
        let text_tokens = vec![UNGENERATED; max_step_idx + config.acoustic_delay];
        let forced_audio_tokens = crate::lm::ForcedAudioTokens::new(
            config.acoustic_delay,
            config.audio_pad_token(),
            &[8, 8],
        );
        Self {
            model,
            audio_tokens,
            text_tokens,
            audio_lp,
            text_lp,
            step_idx: 0,
            pad_mult,
            repetition_penalty,
            forced_audio_tokens,
            user_rating: 0, // 0 indicates no ratings have been submitted from the front
            cfg_alpha,
            config,
        }
    }

    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    fn audio_pad_token(&self) -> u32 {
        self.config.audio_pad_token()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn user_rating(&self) -> u32 {
        self.user_rating
    }
    pub fn set_user_rating(&mut self, grade: u32) {
        self.user_rating = grade
    }

    fn apply_repetition_penalty(&self, logits: Tensor) -> candle::Result<Tensor> {
        let logits = match self.repetition_penalty {
            None => logits,
            Some((_, 1.)) => logits,
            Some((context_size, penalty)) => {
                let device = logits.device();
                let mut logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
                let mut already_seen = std::collections::HashSet::new();
                let mut non_pad_tokens = 0;
                for &token_id in self.text_tokens(false).iter().rev() {
                    if token_id == self.config.text_pad_token
                        || token_id == self.config.text_eop_token
                        || token_id == self.config.text_start_token
                    {
                        continue;
                    }
                    // Look at the last [context_size] tokens at most, count all tokens there even
                    // if we already saw them.
                    if non_pad_tokens >= context_size {
                        break;
                    }
                    non_pad_tokens += 1;

                    if already_seen.contains(&token_id) {
                        continue;
                    }

                    already_seen.insert(token_id);
                    if let Some(logit) = logits.get_mut(token_id as usize) {
                        if *logit >= 0. {
                            *logit /= penalty
                        } else {
                            *logit *= penalty
                        }
                    }
                }
                let logits_len = logits.len();
                Tensor::from_vec(logits, logits_len, device)?
            }
        };
        Ok(logits)
    }

    // The acoustic tokens are written with a delay, so this can create "gaps" of UNGENERATED
    // tokens in the case where we call `step_audio_prompt` *after* `step`.
    pub fn step_(
        &mut self,
        text_token: Option<u32>,
        input_audio_tokens: &[u32],
        force_text_token: Option<u32>,
        ca_src: Option<&CaSrc>,
        conditions: Option<&crate::conditioner::Condition>,
    ) -> candle::Result<u32> {
        let mut codes = Vec::with_capacity(self.config.total_audio_codebooks());
        let dev = self.model.device();
        for (c_idx, &t) in input_audio_tokens.iter().enumerate() {
            self.audio_tokens[self.step_idx][c_idx + self.config.generated_audio_codebooks] = t
        }
        let batch_size = if self.cfg_alpha.is_some() { 2 } else { 1 };
        for codebook in 0..self.config.total_audio_codebooks() {
            let t = if codebook == 0 || codebook == self.config.generated_audio_codebooks {
                if self.step_idx == 0 {
                    self.audio_pad_token()
                } else {
                    self.audio_tokens[self.step_idx - 1][codebook]
                }
            } else if self.step_idx <= self.config.acoustic_delay {
                self.audio_pad_token()
            } else {
                self.audio_tokens[self.step_idx - self.config.acoustic_delay - 1][codebook]
            };
            if t == UNGENERATED {
                candle::bail!("internal error, ungenerated {} {codebook}", self.step_idx)
            }
            let t = Tensor::from_vec(vec![t; batch_size], (batch_size, 1), dev)?;
            codes.push(Some(t))
        }
        let text_token = match text_token {
            Some(text_token) => {
                Some(Tensor::from_vec(vec![text_token; batch_size], (batch_size, 1), dev)?)
            }
            None => None,
        };
        let (text_logits, ys) = match ca_src.as_ref() {
            None => {
                let (logits, ys) =
                    self.model.forward_cond(text_token, codes, conditions, &().into())?;
                let logits = match self.cfg_alpha {
                    None => logits.i((0, 0))?,
                    Some(a) => match logits.dim(0)? {
                        2 => ((logits.i((0, 0))? * a)? - (logits.i((1, 0))? * (a - 1.))?)?,
                        b_size => candle::bail!("unexpected batch size {b_size}"),
                    },
                };
                (logits, ys)
            }
            Some(ca_src) => {
                if self.cfg_alpha.is_some() {
                    candle::bail!("cfg is not supported with cross attention")
                }
                let (logits, ys) =
                    self.model.forward_ca(text_token, codes, ca_src, None, &().into())?;
                (logits.i((0, 0))?, ys)
            }
        };
        let text_logits = self.apply_repetition_penalty(text_logits)?;
        let text_token = match force_text_token {
            Some(tt) => tt,
            None => self.text_lp.sample_f(&text_logits, |prs| {
                if let Some(pad_mult) = self.pad_mult.as_ref() {
                    prs[self.config.text_pad_token as usize] *= f32::exp(*pad_mult);
                }
            })?,
        };
        self.text_tokens[self.step_idx] = text_token;
        let last_audio_tokens = match self.cfg_alpha {
            None => self.model.depformer_sample(
                &ys,
                Some(text_token),
                self.forced_audio_tokens.forced_tokens(self.step_idx),
                &mut self.audio_lp,
            )?,
            Some(cfg_alpha) => self.model.depformer_sample_cfg(
                &ys,
                cfg_alpha,
                Some(text_token),
                self.forced_audio_tokens.forced_tokens(self.step_idx),
                &mut self.audio_lp,
            )?,
        };
        let audio_pad_token = self.audio_pad_token();
        for c_idx in 0..self.config.generated_audio_codebooks {
            let delay = if c_idx == 0 || c_idx == self.config.generated_audio_codebooks {
                0
            } else {
                self.config.acoustic_delay
            };
            let pos = &mut self.audio_tokens[self.step_idx.saturating_sub(delay)][c_idx];
            // Overwrite existing positions even if there are non-UNGENERATED values. This
            // actually happens for the first few slices because of the saturating_sub.
            *pos = last_audio_tokens.as_ref().map_or(audio_pad_token, |l| l[c_idx]);
        }
        self.step_idx += 1;
        if self.step_idx >= self.audio_tokens.len() {
            candle::bail!("max step-idx reached")
        }
        Ok(text_token)
    }

    pub fn step_without_ca_src(
        &mut self,
        text_token: u32,
        input_audio_tokens: &[u32],
        force_text_token: Option<u32>,
    ) -> candle::Result<u32> {
        self.step_(Some(text_token), input_audio_tokens, force_text_token, None, None)
    }

    pub fn step(
        &mut self,
        text_token: u32,
        input_audio_tokens: &[u32],
        force_text_token: Option<u32>,
        ca_src: Option<&CaSrc>,
    ) -> candle::Result<u32> {
        self.step_(Some(text_token), input_audio_tokens, force_text_token, ca_src, None)
    }

    /// If include_all is set, all the time steps are returned. Otherwise only the timesteps that
    /// have been generated are handled.
    pub fn audio_tokens(&self, include_all: bool) -> &[Vec<u32>] {
        if include_all {
            &self.audio_tokens
        } else {
            let max_idx = usize::min(self.step_idx, self.audio_tokens.len());
            &self.audio_tokens[..max_idx]
        }
    }

    pub fn text_tokens(&self, include_all: bool) -> &[u32] {
        if include_all {
            &self.text_tokens
        } else {
            let max_idx = usize::min(self.step_idx, self.text_tokens.len());
            &self.text_tokens[..max_idx]
        }
    }

    pub fn last_audio_tokens(&self) -> Option<Vec<u32>> {
        if self.step_idx <= self.config.acoustic_delay {
            None
        } else {
            // step_idx is in advance by 1 + there is a 2 token delay on audio tokens.
            let audio_tokens = &self.audio_tokens[self.step_idx - self.config.acoustic_delay - 1];
            if audio_tokens.iter().any(|v| *v as usize >= self.config.audio_vocab_size - 1) {
                None
            } else {
                Some(audio_tokens.clone())
            }
        }
    }
}
