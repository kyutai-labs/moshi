// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// The state struct in this module handles generation for a LM model:
// - Apply the audio delays.
// - Allow for teacher forcing of the audio/text tokens.
// - Support "literal-zeros" tokens for both text and audio.
// - Make no assumptions on the number of streams.
// - TODO: Handle batch size > 1
// - TODO: Support CFG.
// - TODO: Use CPU based tensors for storing the tokens?

use candle::{IndexOp, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Token {
    Set(u32),
    Ungenerated,
    LiteralZero,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub audio_delays: Vec<usize>,
    pub audio_vocab_size: usize,
    pub text_pad_token: u32,
    pub text_eop_token: u32,
    pub text_start_token: u32,
}

impl Config {
    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn audio_codebooks(&self) -> usize {
        self.audio_delays.len()
    }

    pub fn max_audio_delay(&self) -> usize {
        self.audio_delays.iter().max().cloned().unwrap_or(0)
    }
}

pub struct State {
    model: crate::lm::LmModel,
    audio_tokens: Vec<Vec<Token>>,
    text_tokens: Vec<Token>,
    audio_lp: LogitsProcessor,
    text_lp: LogitsProcessor,
    step_idx: usize,
    config: Config,
}

impl State {
    pub fn new(
        model: crate::lm::LmModel,
        max_step_idx: usize,
        audio_lp: LogitsProcessor,
        text_lp: LogitsProcessor,
        config: Config,
    ) -> Self {
        // TODO(laurent): handle a batch dimension.
        let total_len = max_step_idx + config.max_audio_delay();
        let audio_tokens = vec![vec![Token::Ungenerated; config.audio_codebooks()]; total_len];
        let text_tokens = vec![Token::Ungenerated; total_len];
        Self { model, audio_tokens, text_tokens, audio_lp, text_lp, step_idx: 0, config }
    }

    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.config.audio_pad_token()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn set_audio_tokens(&mut self, audio_tokens: &[Option<Token>]) -> Result<()> {
        for (s, at) in self.audio_tokens[self.step_idx].iter_mut().zip(audio_tokens.iter()) {
            if let Some(at) = at {
                *s = *at
            }
        }
        Ok(())
    }

    pub fn step(&mut self, conditions: Option<&crate::conditioner::Condition>) -> Result<()> {
        let dev = self.model.device();

        let mut forced_audio_tokens = Vec::with_capacity(self.config.audio_codebooks());
        for (codebook, &delay) in self.config.audio_delays.iter().enumerate() {
            let forced_token = if self.step_idx < delay {
                Some(self.audio_pad_token())
            } else {
                match self.audio_tokens[self.step_idx - delay][codebook] {
                    Token::Ungenerated | Token::LiteralZero => None,
                    Token::Set(v) => Some(v),
                }
            };
            forced_audio_tokens.push(forced_token);
        }

        let mut codes = Vec::with_capacity(self.config.audio_codebooks());
        for (codebook, &delay) in self.config.audio_delays.iter().enumerate() {
            let t = if self.step_idx <= delay {
                Some(self.audio_pad_token())
            } else {
                match self.audio_tokens[self.step_idx - delay - 1][codebook] {
                    Token::LiteralZero => None,
                    Token::Set(v) => Some(v),
                    Token::Ungenerated => {
                        candle::bail!("internal error, ungenerated {} {codebook}", self.step_idx)
                    }
                }
            };
            let t = match t {
                None => None,
                Some(t) => Some(Tensor::from_vec(vec![t; 1], (1, 1), dev)?),
            };
            codes.push(t)
        }
        let text_token = if self.step_idx == 0 {
            Some(self.config.text_start_token)
        } else {
            match self.text_tokens[self.step_idx - 1] {
                Token::LiteralZero => None,
                Token::Set(t) => Some(t),
                Token::Ungenerated => {
                    candle::bail!("internal error, ungenerated {} text", self.step_idx)
                }
            }
        };
        let text_token = match text_token {
            None => None,
            Some(t) => Some(Tensor::from_vec(vec![t; 1], (1, 1), dev)?),
        };
        let (text_logits, ys) =
            self.model.forward_cond(text_token, codes, conditions, &().into())?;
        let text_token = match self.text_tokens[self.step_idx] {
            Token::Ungenerated => {
                let t = self.text_lp.sample(&text_logits.i((0, 0))?)?;
                self.text_tokens[self.step_idx] = Token::Set(t);
                Some(t)
            }
            Token::Set(t) => Some(t),
            Token::LiteralZero => None,
        };
        let audio_tokens = self.model.depformer_sample(
            &ys,
            text_token,
            &forced_audio_tokens,
            &mut self.audio_lp,
        )?;
        if let Some(audio_tokens) = audio_tokens {
            for (codebook, audio_token) in audio_tokens.into_iter().enumerate() {
                let delay = self.config.audio_delays[codebook];
                if self.step_idx < delay {
                    continue;
                }
                let pos = &mut self.audio_tokens[self.step_idx - delay][codebook];
                if *pos == Token::Ungenerated {
                    *pos = Token::Set(audio_token)
                }
            }
        }
        self.step_idx += 1;
        if self.step_idx >= self.audio_tokens.len() {
            candle::bail!("max step-idx reached")
        }
        Ok(())
    }

    pub fn last_text_token(&self) -> Result<Option<u32>> {
        if self.step_idx == 0 {
            Ok(None)
        } else {
            match self.text_tokens[self.step_idx - 1] {
                Token::Set(t) => Ok(Some(t)),
                Token::LiteralZero => Ok(None),
                Token::Ungenerated => {
                    candle::bail!("internal error, ungenerated step {}, text", self.step_idx)
                }
            }
        }
    }

    pub fn last_audio_tokens(&self) -> Result<Option<Vec<u32>>> {
        let max_audio_delay = self.config.max_audio_delay();
        if self.step_idx <= max_audio_delay {
            Ok(None)
        } else {
            let mut audio_tokens = vec![];
            for (cb, audio_token) in
                self.audio_tokens[self.step_idx - max_audio_delay - 1].iter().enumerate()
            {
                match audio_token {
                    Token::LiteralZero => return Ok(None),
                    Token::Set(s) => audio_tokens.push(*s),
                    Token::Ungenerated => {
                        candle::bail!("internal error, ungenerated step {}, cb {cb}", self.step_idx)
                    }
                }
            }
            Ok(Some(audio_tokens))
        }
    }
}
