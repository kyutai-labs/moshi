// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
use crate::lm::LmModel;
use crate::mimi::Mimi;
use candle::{IndexOp, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Word {
    pub tokens: Vec<u32>,
    pub start_time: f64,
    pub stop_time: f64,
}

pub struct State {
    asr_delay_in_tokens: usize,
    text_token: u32,
    audio_tokenizer: Mimi,
    lm: LmModel,
    device: candle::Device,
    step_idx: usize,
    word_tokens: Vec<u32>,
    last_stop_time: f64,
    lp: LogitsProcessor,
}

impl State {
    pub fn new(asr_delay_in_tokens: usize, audio_tokenizer: Mimi, lm: LmModel) -> Result<Self> {
        let text_token = lm.text_start_token();
        let device = lm.device().clone();
        let mut s = Self {
            asr_delay_in_tokens,
            lm,
            audio_tokenizer,
            device,
            text_token,
            word_tokens: vec![],
            step_idx: 0,
            last_stop_time: 0.,
            lp: LogitsProcessor::new(42, None, None),
        };
        s.reset()?;
        Ok(s)
    }

    pub fn device(&self) -> &candle::Device {
        &self.device
    }

    pub fn reset(&mut self) -> Result<()> {
        self.step_idx = 0;
        self.lm.reset_state();
        self.audio_tokenizer.reset_state();
        self.word_tokens.clear();
        let text_start_token = self.lm.text_start_token();
        let audio_pad_token = self.lm.audio_pad_token();
        let text = Tensor::from_vec(vec![text_start_token], (1, 1), &self.device)?;
        let audio_token = Tensor::from_vec(vec![audio_pad_token], (1, 1), &self.device)?;
        let audio_tokens = vec![Some(audio_token); self.lm.in_audio_codebooks()];
        let (_, _) = self.lm.forward(Some(text), audio_tokens)?;
        Ok(())
    }

    pub fn step_pcm<F>(&mut self, pcm: Tensor, f: F) -> Result<Vec<Word>>
    where
        F: Fn(u32, Tensor) -> Result<()>,
    {
        let audio_tokens = self.audio_tokenizer.encode_step(&pcm.into())?;
        if let Some(audio_tokens) = audio_tokens.as_option() {
            self.step_tokens(audio_tokens, f)
        } else {
            Ok(vec![])
        }
    }

    pub fn step_tokens<F>(&mut self, audio_tokens: &Tensor, f: F) -> Result<Vec<Word>>
    where
        F: Fn(u32, Tensor) -> Result<()>,
    {
        let (_one, codebooks, steps) = audio_tokens.dims3()?;
        let mut words = vec![];
        for step in 0..steps {
            {
                let audio_tokens = audio_tokens.narrow(2, step, 1)?;
                f(self.text_token, audio_tokens)?;
            }

            let audio_tokens = (0..codebooks)
                .map(|idx| {
                    audio_tokens.i((0, idx, step)).and_then(|v| v.reshape((1, ()))).map(Some)
                })
                .collect::<Result<Vec<_>>>()?;

            let text = if self.step_idx >= self.asr_delay_in_tokens {
                let dev = self.lm.device();
                Some(Tensor::from_vec(vec![self.text_token], (1, 1), dev)?)
            } else {
                None
            };
            let (text_logits, _) = self.lm.forward(text, audio_tokens)?;
            self.step_idx += 1;
            let text_logits = text_logits.i((0, 0))?;
            self.text_token = self.lp.sample(&text_logits)?;
            if self.step_idx >= self.asr_delay_in_tokens {
                if self.text_token == 0 {
                    let mut tokens = vec![];
                    std::mem::swap(&mut self.word_tokens, &mut tokens);
                    let stop_time = (self.step_idx - self.asr_delay_in_tokens) as f64 / 12.5;
                    words.push(Word { tokens, start_time: self.last_stop_time, stop_time });
                    self.last_stop_time = stop_time;
                } else if self.text_token != 3 {
                    self.word_tokens.push(self.text_token)
                }
            }
        }
        Ok(words)
    }
}
