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
    audio_pad_token: u32,
    next_codebooks: Vec<u32>,
}

impl ItemState {
    fn reset(&mut self) {
        self.step_idx = 0;
        self.text_token = 0;
        self.word_tokens.clear();
        self.unended_word = false;
        self.last_stop_time = 0.;
        self.next_codebooks.fill(self.audio_pad_token);
    }

    pub fn text_token(&self) -> u32 {
        self.text_token
    }

    pub fn is_first_step(&self) -> bool {
        self.step_idx == 0
    }

    pub fn next_token(&mut self, codebook_idx: usize, token: u32) -> u32 {
        let v = self.next_codebooks[codebook_idx];
        self.next_codebooks[codebook_idx] = token;
        if self.is_first_step() {
            self.audio_pad_token
        } else {
            v
        }
    }
}

pub struct State {
    asr_delay_in_tokens: usize,
    model_step_idx: usize,
    temperature: f64,
    lm: LmModel,
    audio_tokenizer: Mimi,
    device: candle::Device,
    batch: Vec<ItemState>,
}

impl State {
    pub fn new(
        batch_size: usize,
        asr_delay_in_tokens: usize,
        temperature: f64,
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
            audio_pad_token: lm.audio_pad_token(),
            next_codebooks: vec![lm.audio_pad_token(); lm.in_audio_codebooks()],
        };
        let mut s = Self {
            asr_delay_in_tokens,
            lm,
            model_step_idx: 0,
            audio_tokenizer,
            temperature,
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

    pub fn asr_delay_in_tokens(&self) -> usize {
        self.asr_delay_in_tokens
    }

    pub fn reset(&mut self) -> Result<()> {
        self.lm.reset_state();
        self.audio_tokenizer.reset_state();
        self.batch.iter_mut().for_each(|s| s.reset());
        Ok(())
    }

    pub fn step_pcm<F>(
        &mut self,
        pcm: Tensor,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &crate::StreamMask,
        f: F,
    ) -> Result<Vec<AsrMsg>>
    where
        F: Fn(&[ItemState], &Tensor, &[Tensor]),
    {
        let audio_tokens = self.audio_tokenizer.encode_step(&pcm.into(), mask)?;
        if let Some(audio_tokens) = audio_tokens.as_option() {
            self.step_tokens(audio_tokens, conditions, mask, f)
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

    pub fn step_tokens<F>(
        &mut self,
        audio_tokens: &Tensor,
        conditions: Option<&crate::conditioner::Condition>,
        mask: &crate::StreamMask,
        f: F,
    ) -> Result<Vec<AsrMsg>>
    where
        F: Fn(&[ItemState], &Tensor, &[Tensor]),
    {
        let (batch_size, codebooks, steps) = audio_tokens.dims3()?;
        if batch_size != self.batch_size() {
            candle::bail!("batch size mismatch: {batch_size} != {}", self.batch_size());
        }
        let mut words = vec![];
        for step in 0..steps {
            let audio_tokens = audio_tokens.narrow(2, step, 1)?;
            let audio_tokens = audio_tokens.reshape((batch_size, codebooks))?.to_vec2::<u32>()?;
            let audio_tokens = (0..codebooks)
                .map(|codebook_idx| {
                    let audio_tokens = audio_tokens
                        .iter()
                        .zip(self.batch.iter_mut())
                        .enumerate()
                        .map(|(batch_idx, (audio_token, item))| {
                            if !mask.is_active(batch_idx) {
                                0
                            } else {
                                item.next_token(codebook_idx, audio_token[codebook_idx])
                            }
                        })
                        .collect();
                    let audio_tokens =
                        Tensor::from_vec(audio_tokens, (batch_size, 1), self.device())?;
                    Ok(audio_tokens)
                })
                .collect::<Result<Vec<_>>>()?;
            let text = self.text_tokens()?;
            f(self.batch.as_slice(), &text, &audio_tokens);
            let audio_tokens = audio_tokens.into_iter().map(Some).collect::<Vec<_>>();
            let (text_logits, transformer_out) =
                self.lm.forward_cond(Some(text), audio_tokens, conditions, mask)?;
            self.model_step_idx += 1;
            let extra_heads = self.lm.extra_heads(&transformer_out)?;
            let mut prs = vec![];
            for extra_head in extra_heads.iter() {
                // Only retrieve the first element for each extra-head.
                let prs_ =
                    candle_nn::ops::softmax_last_dim(&extra_head.to_dtype(candle::DType::F32)?)?
                        .i((.., 0, 0))?
                        .to_vec1::<f32>()?;
                prs.push(prs_);
            }
            if !prs.is_empty() {
                words.push(AsrMsg::Step { step_idx: self.model_step_idx(), prs });
            }

            let text_tokens = if self.temperature <= 0.0 {
                text_logits.i((.., 0))?.argmax(candle::D::Minus1)?
            } else {
                candle_nn::sampling::gumbel_softmax(
                    &text_logits.i((.., 0))?.to_dtype(candle::DType::F32)?,
                    self.temperature,
                    candle::D::Minus1,
                )?
            };
            let text_tokens = text_tokens.to_vec1::<u32>()?;
            for (batch_idx, (text_token, item)) in
                text_tokens.into_iter().zip(self.batch.iter_mut()).enumerate()
            {
                if !mask.is_active(batch_idx) {
                    continue;
                }
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
