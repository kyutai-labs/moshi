// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use candle::{IndexOp, Result, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::transformer::CaSrc;

pub const UNGENERATED: u32 = u32::MAX;

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct Config {
    pub acoustic_delay: usize,
    pub text_pad_token: u32,
    pub text_bos_token: u32,
    pub text_eos_token: u32,
    pub text_eop_token: u32,
    pub text_start_token: u32,
    pub text_audio_delay_in_tokens: usize,
    pub max_consecutive_pads: usize,
    pub extra_steps: usize,
    pub speaker_cond_duration_s: f64,
    pub speaker_cond_dim: usize,
    pub speaker_cond_n_speakers: usize,
}

impl Config {
    pub fn v202501() -> Self {
        Self {
            acoustic_delay: 2,
            text_eop_token: 0,
            text_bos_token: 1,
            text_eos_token: 2,
            text_pad_token: 3,
            text_start_token: 8000,
            text_audio_delay_in_tokens: 25, // aka interleaver_delay = 2s
            max_consecutive_pads: 10,
            extra_steps: 5,
            speaker_cond_duration_s: 10.,
            speaker_cond_dim: 2048,
            speaker_cond_n_speakers: 5,
        }
    }
}

pub struct State {
    model: crate::lm::LmModel,
    ca_src: Option<CaSrc>,
    audio_tokens: Vec<Vec<u32>>,
    text_tokens: Vec<u32>,
    consecutive_pads: usize,
    audio_lp: LogitsProcessor,
    text_lp: LogitsProcessor,
    step_idx: usize,
    forced_audio_tokens: crate::lm::ForcedAudioTokens,
    cfg_alpha: Option<f64>,
    config: Config,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowedTokens {
    Text(u32),
    Pad,
    PadOrEpad,
}

impl State {
    pub fn new(
        model: crate::lm::LmModel,
        ca_src: Option<CaSrc>,
        max_step_idx: usize,
        audio_lp: LogitsProcessor,
        text_lp: LogitsProcessor,
        cfg_alpha: Option<f64>,
        config: Config,
    ) -> Self {
        let audio_tokens: Vec<Vec<u32>> = vec![
            vec![UNGENERATED; model.generated_audio_codebooks()];
            max_step_idx + config.acoustic_delay
        ];
        let text_tokens = vec![UNGENERATED; max_step_idx + config.acoustic_delay];
        let forced_audio_tokens = crate::lm::ForcedAudioTokens::new(
            config.acoustic_delay,
            model.audio_pad_token(),
            &[model.generated_audio_codebooks()],
        );
        Self {
            model,
            ca_src,
            audio_tokens,
            text_tokens,
            consecutive_pads: 0,
            audio_lp,
            text_lp,
            step_idx: 0,
            forced_audio_tokens,
            cfg_alpha,
            config,
        }
    }

    pub fn step_idx(&self) -> usize {
        self.step_idx
    }

    fn audio_pad_token(&self) -> u32 {
        self.model.audio_pad_token()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    // The acoustic tokens are written with a delay, so this can create "gaps" of UNGENERATED
    // tokens in the case where we call `step_audio_prompt` *after* `step`.
    pub fn step(
        &mut self,
        prev_text_token: u32,
        allowed_tokens: AllowedTokens,
        conditions: Option<&crate::conditioner::Condition>,
    ) -> Result<u32> {
        let mut codes = Vec::with_capacity(self.model.generated_audio_codebooks());
        let dev = self.model.device();
        let batch_size = if self.cfg_alpha.is_some() { 2 } else { 1 };
        for codebook in 0..self.model.generated_audio_codebooks() {
            let t = if codebook == 0 {
                if self.step_idx == 0 {
                    Some(self.audio_pad_token())
                } else if self.step_idx <= self.config.text_audio_delay_in_tokens {
                    // The delayed pattern for TTS is a bit special, the audio-pad tokens are used
                    // in the same way as usual, i.e. for the first slice and until the acoustic
                    // delay for semantic tokens.
                    // However for the first couple seconds (set by `text_audio_delay_in_tokens`),
                    // the tokens that are *not* audio-pad are replaced by "literal zeros".
                    None
                } else {
                    Some(self.audio_tokens[self.step_idx - 1][codebook])
                }
            } else if self.step_idx <= self.config.acoustic_delay {
                Some(self.audio_pad_token())
            } else if self.step_idx
                <= self.config.text_audio_delay_in_tokens + self.config.acoustic_delay
            {
                // The same comment as above applies here.
                None
            } else {
                Some(self.audio_tokens[self.step_idx - self.config.acoustic_delay - 1][codebook])
            };
            if t == Some(UNGENERATED) {
                candle::bail!("internal error, ungenerated {}", self.step_idx)
            }
            let t = match t {
                Some(t) => Some(Tensor::from_vec(vec![t; batch_size], (batch_size, 1), dev)?),
                None => None,
            };
            codes.push(t)
        }
        let prev_text_token =
            Some(Tensor::from_vec(vec![prev_text_token; batch_size], (batch_size, 1), dev)?);
        let (text_logits, ys) = match self.ca_src.as_ref() {
            None => self.model.forward_cond(prev_text_token, codes, conditions, &().into())?,
            Some(ca_src) => {
                self.model.forward_ca(prev_text_token, codes, ca_src, conditions, &().into())?
            }
        };
        let text_logits = match self.cfg_alpha {
            None => text_logits.i((0, 0))?,
            Some(a) => match text_logits.dim(0)? {
                2 => ((text_logits.i((0, 0))? * a)? - (text_logits.i((1, 0))? * (a - 1.))?)?,
                b_size => candle::bail!("unexpected batch size {b_size}"),
            },
        };
        // When in tts mode, there are only two possible outcomes corresponding to tokens 0 and 3.
        // 0 -> EOP or the next text token, this is ambiguous, a list of consecutive 0s correspond to
        //   word + EOP + word + EOP ...
        // 3 -> pad.
        // This will change when the simplerleaver lands.
        let text_token = match allowed_tokens {
            AllowedTokens::Text(v) => v,
            AllowedTokens::Pad => self.config.text_pad_token,
            AllowedTokens::PadOrEpad => {
                if self.consecutive_pads > self.config.max_consecutive_pads {
                    self.config.text_eop_token
                } else {
                    let text_token = self.text_lp.sample(&text_logits)?;
                    if text_token == self.config.text_pad_token {
                        self.config.text_pad_token
                    } else {
                        self.config.text_eop_token
                    }
                }
            }
        };
        if text_token == self.config.text_pad_token {
            self.consecutive_pads += 1
        } else {
            self.consecutive_pads = 0
        }
        self.text_tokens[self.step_idx] = text_token;
        let last_audio_tokens = if self.step_idx < self.config.text_audio_delay_in_tokens {
            None
        } else {
            match self.cfg_alpha {
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
            }
        };
        let audio_pad_token = self.audio_pad_token();
        for c_idx in 0..self.model.generated_audio_codebooks() {
            let delay = if c_idx == 0 { 0 } else { self.config.acoustic_delay };
            let pos = &mut self.audio_tokens[self.step_idx.saturating_sub(delay)][c_idx];
            match last_audio_tokens.as_ref() {
                Some(lat) => {
                    if *pos == UNGENERATED {
                        *pos = lat[c_idx]
                    }
                }
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

    pub fn overwrite_last_text_token(&mut self, text_token: u32) -> Result<()> {
        if self.step_idx == 0 {
            candle::bail!("cannot overwrite first token")
        }
        if text_token == UNGENERATED {
            candle::bail!("cannot overwrite with UNGENERATED")
        }
        self.text_tokens[self.step_idx - 1] = text_token;
        Ok(())
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
            if audio_tokens.iter().any(|v| *v >= self.audio_pad_token()) {
                None
            } else {
                Some(audio_tokens.clone())
            }
        }
    }

    pub fn audio_codebooks(&self) -> usize {
        self.model.generated_audio_codebooks()
    }

    pub fn device(&self) -> &candle::Device {
        self.model.device()
    }

    pub fn dtype(&self) -> candle::DType {
        self.model.dtype()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Main,
    Other,
}

pub fn tokenize_prompt<E>(
    text: &[String],
    text_bos_token: u32,
    text_eos_token: u32,
    encode: impl Fn(&str) -> std::result::Result<Vec<u32>, E>,
) -> std::result::Result<Vec<(Vec<u32>, Speaker)>, E> {
    let mut prompt = vec![];
    for (turn_idx, turn) in text.iter().enumerate() {
        let (speaker, turn_token) = if turn_idx % 2 == 0 {
            (Speaker::Main, text_bos_token)
        } else {
            (Speaker::Other, text_eos_token)
        };
        for (word_idx, word) in turn.split(' ').enumerate() {
            let mut word = encode(word)?.into_iter().collect::<Vec<_>>();
            if word_idx == 0 && speaker == Speaker::Main {
                word.insert(0, turn_token)
            }
            if !word.is_empty() {
                prompt.push((word, speaker))
            }
        }
    }
    Ok(prompt)
}

#[derive(Debug, Clone)]
pub struct SpeakerEncoder {
    mimi: crate::mimi::Mimi,
    learnt_padding: Tensor,
    proj: candle_nn::Linear,
    n_speakers: usize,
    cond_dim: usize,
    device: candle::Device,
    dtype: candle::DType,
}

impl SpeakerEncoder {
    pub fn new(
        mimi: crate::mimi::Mimi,
        speaker_cond_dim: usize,
        speaker_cond_n_speakers: usize,
        dtype: candle::DType,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self> {
        let learnt_padding = vb.get(
            (1, 1, speaker_cond_dim),
            "condition_provider.conditioners.speaker_wavs.learnt_padding",
        )?;
        let mimi_dim = mimi.config().seanet.dimension;
        let proj = candle_nn::linear_no_bias(
            mimi_dim,
            speaker_cond_dim,
            vb.pp("condition_provider.conditioners.speaker_wavs.output_proj"),
        )?;
        Ok(Self {
            mimi,
            learnt_padding,
            proj,
            n_speakers: speaker_cond_n_speakers,
            cond_dim: speaker_cond_dim,
            device: vb.device().clone(),
            dtype,
        })
    }

    pub fn device(&self) -> &candle::Device {
        &self.device
    }

    pub fn sample_rate(&self) -> f64 {
        self.mimi.config().sample_rate
    }

    pub fn encode(&self, speakers: &[Tensor]) -> Result<Tensor> {
        if speakers.is_empty() {
            candle::bail!("empty speakers in encode")
        }
        let mut pcms = vec![];
        for pcm in speakers.iter().take(self.n_speakers) {
            let stdev = pcm.broadcast_sub(&pcm.mean_all()?)?.sqr()?.mean_all()?.sqrt()?;
            let pcm = (pcm * 0.08)?.broadcast_div(&stdev)?;
            pcms.push(pcm)
        }
        let n_speakers = pcms.len();
        let pcm = Tensor::cat(&pcms, 0)?;
        let mut mimi = self.mimi.clone();
        mimi.reset_state();
        let embeddings = mimi.encode_pre_quantize(&pcm)?.t()?.apply(&self.proj)?;
        let embeddings = if n_speakers < self.n_speakers {
            let lp =
                embeddings.narrow(0, 0, 1)?.zeros_like()?.broadcast_add(&self.learnt_padding)?;
            let mut embs = vec![embeddings];
            embs.resize(self.n_speakers - n_speakers + 1, lp);
            Tensor::cat(&embs, 0)?
        } else {
            embeddings
        };
        let embeddings = embeddings.flatten(0, 1)?.unsqueeze(0)?;
        let embeddings = crate::tts::add_sin_embeddings(&embeddings)?;
        embeddings.to_dtype(self.dtype)
    }

    pub fn empty(&self) -> Result<Tensor> {
        let embeddings =
            self.learnt_padding.broadcast_as((1, self.n_speakers * 125, self.cond_dim))?;
        let embeddings = crate::tts::add_sin_embeddings(&embeddings)?;
        embeddings.to_dtype(self.dtype)
    }
}
