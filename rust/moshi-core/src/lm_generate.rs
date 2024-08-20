// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use candle::{IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;

const UNGENERATED: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub struct Config {
    pub audio_codebooks: usize,
    pub audio_vocab_size: usize,
    pub acoustic_delay: usize,
    pub text_bos_token: u32,
    pub text_eos_token: u32,
    pub text_pad_token: u32,
    pub text_start_token: u32,
}

impl Config {
    pub fn v0_1() -> Self {
        Self {
            audio_codebooks: 8,
            audio_vocab_size: 2049,
            acoustic_delay: 2,
            text_bos_token: 1,
            text_eos_token: 2,
            text_pad_token: 3,
            text_start_token: 32000,
        }
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn audio_codebooks(&self) -> usize {
        self.audio_codebooks
    }
}

pub struct State {
    model: crate::lm::LmModel,
    audio_tokens: Vec<Vec<u32>>,
    audio_lp: LogitsProcessor,
    text_lp: LogitsProcessor,
    step_idx: usize,
    config: Config,
    npads: i32,
}

impl State {
    pub fn new(
        model: crate::lm::LmModel,
        max_step_idx: usize,
        audio_lp: LogitsProcessor,
        text_lp: LogitsProcessor,
        config: Config,
    ) -> Self {
        let audio_tokens: Vec<Vec<u32>> =
            vec![vec![UNGENERATED; config.audio_codebooks]; max_step_idx + config.acoustic_delay];
        Self { model, audio_tokens, audio_lp, text_lp, step_idx: 0, npads: 0, config }
    }

    pub fn audio_codebooks(&self) -> usize {
        self.config.audio_codebooks
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.config.audio_pad_token()
    }

    pub fn step_gen_no_text(&mut self, force_text_token: Option<u32>) -> candle::Result<u32> {
        self.step(None, true, force_text_token)
    }

    pub fn step_gen(&mut self, prev_text_token: u32) -> candle::Result<u32> {
        self.step(Some(prev_text_token), true, None)
    }

    pub fn step_text_prompt(&mut self, id: u32) -> candle::Result<u32> {
        self.step(Some(id), false, None)
    }

    pub fn step_audio_prompt_(
        &mut self,
        codes: &[u32],
        text_token: Option<u32>,
    ) -> candle::Result<u32> {
        if codes.len() != self.audio_codebooks() {
            candle::bail!("unexpected codes length {} {}", codes.len(), self.audio_codebooks())
        }
        self.audio_tokens[self.step_idx].copy_from_slice(codes);
        let prev_text =
            if self.step_idx == 0 { Some(self.config.text_start_token) } else { text_token };
        self.step(prev_text, false, None)
    }

    pub fn step_audio_prompt(&mut self, codes: &[u32]) -> candle::Result<u32> {
        self.step_audio_prompt_(codes, None)
    }

    pub fn step_audio_prompt_with_text(&mut self, codes: &[u32], text: u32) -> candle::Result<u32> {
        self.step_audio_prompt_(codes, Some(text))
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

    pub fn audio_tokens(&self) -> Vec<Vec<u32>> {
        let l = self.step_idx - self.config.acoustic_delay - 1;
        self.audio_tokens[..l].to_vec()
    }

    // The acoustic tokens are written with a delay, so this can create "gaps" of UNGENERATED
    // tokens in the case where we call `step_audio_prompt` *after* `step`.
    fn step(
        &mut self,
        text_token: Option<u32>,
        gen_audio: bool,
        force_text_token: Option<u32>,
    ) -> candle::Result<u32> {
        let mut codes = Vec::with_capacity(self.audio_codebooks());
        let dev = self.model.device();
        for codebook in 0..self.audio_codebooks() {
            let t = if codebook == 0 {
                if self.step_idx == 0 {
                    self.audio_pad_token()
                } else {
                    self.audio_tokens[self.step_idx - 1][0]
                }
            } else if self.step_idx <= self.config.acoustic_delay {
                self.audio_pad_token()
            } else {
                self.audio_tokens[self.step_idx - self.config.acoustic_delay - 1][codebook]
            };
            if t == UNGENERATED {
                candle::bail!("internal error, ungenerated {}", self.step_idx)
            }
            let t = Tensor::new(&[t], dev)?.unsqueeze(0)?;
            codes.push(Some(t))
        }
        let text_token = match text_token {
            None => None,
            Some(text_token) => Some(Tensor::from_vec(vec![text_token], (1, 1), dev)?),
        };
        let (text_logits, ys) = self.model.forward(text_token, codes)?;
        let text_logits = text_logits.i((0, 0))?;
        let text_token = match force_text_token {
            None => self.text_lp.sample_f(&text_logits, |prs| {
                prs[self.config.text_bos_token as usize] = 1e-9;
                if self.npads > 40 {
                    let mul = 2f32.powi(self.npads - 40);
                    prs[self.config.text_eos_token as usize] *= mul;
                }
            })?,
            Some(t) => t,
        };
        if text_token == self.config.text_pad_token {
            self.npads += 1;
        } else {
            self.npads = 0;
        }

        let last_audio_tokens = if gen_audio {
            self.model.depformer_sample(self.step_idx, &ys, Some(text_token), &mut self.audio_lp)?
        } else {
            None
        };
        let audio_pad_token = self.audio_pad_token();
        for c_idx in 0..self.audio_codebooks() {
            let delay = if c_idx == 0 { 0 } else { self.config.acoustic_delay };
            let pos = &mut self.audio_tokens[self.step_idx.saturating_sub(delay)][c_idx];
            match last_audio_tokens.as_ref() {
                Some(lat) => *pos = lat[c_idx],
                None => {
                    if *pos == UNGENERATED {
                        *pos = audio_pad_token
                    }
                }
            }
        }
        self.step_idx += 1;
        if self.step_idx >= self.audio_tokens.len() {
            candle::bail!("max step-idx reached")
        }
        Ok(text_token)
    }
}
