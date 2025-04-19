// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
use crate::lm::LmModel;
use crate::mimi::Mimi;
use candle::{IndexOp, Result, Tensor};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AsrMsg {
    Step { step_idx: usize, prs: Vec<Vec<f32>> },
    Word { tokens: Vec<u32>, start_time: f64, batch_idx: usize },
    EndWord { stop_time: f64, batch_idx: usize },
}

#[derive(Debug, Clone)]
pub struct ItemState {
    step_idx: usize,
    text_token: u32,
    word_tokens: Vec<u32>,
    unended_word: bool,
    last_stop_time: f64,
}

impl ItemState {
    fn reset(&mut self) {
        self.step_idx = 0;
        self.text_token = 0;
        self.word_tokens.clear();
        self.unended_word = false;
        self.last_stop_time = 0.;
    }

    pub fn text_token(&self) -> u32 {
        self.text_token
    }

    pub fn is_first_step(&self) -> bool {
        self.step_idx == 0
    }
}

pub struct State {
    asr_delay_in_tokens: usize,
    model_step_idx: usize,
    lm: LmModel,
    audio_tokenizer: Mimi,
    device: candle::Device,
    batch: Vec<ItemState>,
}

impl State {
    pub fn new(
        batch_size: usize,
        asr_delay_in_tokens: usize,
        audio_tokenizer: Mimi,
        lm: LmModel,
    ) -> Result<Self> {
        let text_token = lm.text_start_token();
        let device = lm.device().clone();
        let item_state = ItemState {
            text_token,
            word_tokens: vec![],
            unended_word: false,
            step_idx: 0,
            last_stop_time: 0.,
        };
        let mut s = Self {
            asr_delay_in_tokens,
            lm,
            model_step_idx: 0,
            audio_tokenizer,
            device,
            batch: vec![item_state; batch_size],
        };
        s.reset()?;
        Ok(s)
    }

    pub fn model_step_idx(&self) -> usize {
        self.model_step_idx
    }

    pub fn device(&self) -> &candle::Device {
        &self.device
    }

    pub fn batch_size(&self) -> usize {
        self.batch.len()
    }

    pub fn reset(&mut self) -> Result<()> {
        self.lm.reset_state();
        self.audio_tokenizer.reset_state();
        self.batch.iter_mut().for_each(|s| s.reset());
        Ok(())
    }

    pub fn step_pcm<F>(&mut self, pcm: Tensor, f: F) -> Result<Vec<AsrMsg>>
    where
        F: Fn(&[ItemState], Tensor) -> Result<()>,
    {
        let audio_tokens = self.audio_tokenizer.encode_step(&pcm.into())?;
        if let Some(audio_tokens) = audio_tokens.as_option() {
            self.step_tokens(audio_tokens, f)
        } else {
            Ok(vec![])
        }
    }

    fn text_tokens(&self) -> Result<Tensor> {
        let batch_size = self.batch_size();
        let text_start_token = self.lm.text_start_token();
        // We used to have literal 0s for the first asr_delay_in_tokens - 1 steps
        // This is not the case anymore.
        let dev = self.lm.device();
        let text_tokens = self
            .batch
            .iter()
            .map(|s| if s.is_first_step() { text_start_token } else { s.text_token() })
            .collect::<Vec<_>>();
        Tensor::from_vec(text_tokens, (batch_size, 1), dev)
    }

    pub fn step_tokens<F>(&mut self, audio_tokens: &Tensor, f: F) -> Result<Vec<AsrMsg>>
    where
        F: Fn(&[ItemState], Tensor) -> Result<()>,
    {
        let (batch_size, codebooks, steps) = audio_tokens.dims3()?;
        let audio_pad_token = self.lm.audio_pad_token();
        if batch_size != self.batch_size() {
            candle::bail!("batch size mismatch: {batch_size} != {}", self.batch_size());
        }
        let mut words = vec![];
        for step in 0..steps {
            let audio_tokens = audio_tokens.narrow(2, step, 1)?;
            f(self.batch.as_slice(), audio_tokens.clone())?;
            let audio_tokens = audio_tokens.reshape((batch_size, codebooks))?.to_vec2::<u32>()?;
            let audio_tokens = (0..codebooks)
                .map(|codebook_idx| {
                    let audio_tokens = audio_tokens
                        .iter()
                        .zip(self.batch.iter())
                        .map(|(audio_token, item)| {
                            if item.is_first_step() {
                                audio_pad_token
                            } else {
                                audio_token[codebook_idx]
                            }
                        })
                        .collect();
                    let audio_tokens =
                        Tensor::from_vec(audio_tokens, (batch_size, 1), self.device())?;
                    Ok(Some(audio_tokens))
                })
                .collect::<Result<Vec<_>>>()?;

            let text = self.text_tokens()?;
            let (text_logits, transformer_out) = self.lm.forward(Some(text), audio_tokens)?;
            self.model_step_idx += 1;
            let extra_heads = self.lm.extra_heads(&transformer_out)?;
            let mut prs = vec![];
            for extra_head in extra_heads.iter() {
                let prs_ =
                    // TODO(laurent): fix for batch size > 1
                    candle_nn::ops::softmax_last_dim(&extra_head.to_dtype(candle::DType::F32)?)?
                        .i((0, 0))?
                        .to_vec1::<f32>()?;
                prs.push(prs_);
            }
            // TODO: words.push(AsrMsg::Step { step_idx: self.step_idx(), prs });

            let text_tokens = text_logits.i((.., 0))?.argmax(candle::D::Minus1)?;
            let text_tokens = text_tokens.to_vec1::<u32>()?;
            for (batch_idx, (text_token, item)) in
                text_tokens.into_iter().zip(self.batch.iter_mut()).enumerate()
            {
                item.text_token = text_token;
                item.step_idx += 1;
                if item.step_idx >= self.asr_delay_in_tokens {
                    if text_token == 3 || text_token == 0 {
                        if !item.word_tokens.is_empty() {
                            let mut tokens = vec![];
                            std::mem::swap(&mut item.word_tokens, &mut tokens);
                            words.push(AsrMsg::Word {
                                tokens,
                                start_time: item.last_stop_time,
                                batch_idx,
                            });
                            item.unended_word = true;
                        }
                    } else {
                        item.word_tokens.push(item.text_token)
                    }
                    if item.text_token == 0 {
                        let stop_time = (item.step_idx - self.asr_delay_in_tokens) as f64 / 12.5;
                        if item.unended_word {
                            item.unended_word = false;
                            words.push(AsrMsg::EndWord { stop_time, batch_idx });
                        }
                        item.last_stop_time = stop_time;
                    }
                }
            }
        }
        Ok(words)
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        if batch_idx >= self.batch_size() {
            candle::bail!("batch index out of range: {batch_idx} >= {}", self.batch_size());
        }
        self.batch[batch_idx].reset();
        self.lm.reset_batch_idx(batch_idx, self.batch_size())?;
        self.audio_tokenizer.reset_batch_idx(batch_idx, self.batch_size())?;
        Ok(())
    }
}
