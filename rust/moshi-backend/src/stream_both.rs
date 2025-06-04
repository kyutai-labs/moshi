// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use axum::extract::ws;
use futures_util::{
    stream::{SplitSink, SplitStream, StreamExt},
    SinkExt,
};
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub instance_name: String,
    #[serde(default)]
    pub hf_repo: String,
    pub lm_model_file: String,
    pub log_dir: String,
    pub text_tokenizer_file: String,
    pub mimi_model_file: String,
    pub mimi_num_codebooks: usize,
    pub lm_config: Option<moshi::lm_generate_multistream::Config>,
    #[serde(default = "default_false")]
    pub use_cpu_for_mimi: bool,
    pub asr_delay_in_tokens: Option<usize>,
}

fn default_false() -> bool {
    false
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = serde_json::from_str(&config)?;
        config.log_dir = crate::utils::replace_env_vars(&config.log_dir);
        config.text_tokenizer_file = crate::utils::replace_env_vars(&config.text_tokenizer_file);
        config.mimi_model_file = crate::utils::replace_env_vars(&config.mimi_model_file);
        config.lm_model_file = crate::utils::replace_env_vars(&config.lm_model_file);
        Ok(config)
    }

    /// Check if all modelling files are available on machine.
    pub fn requires_model_download(&self) -> bool {
        [&self.lm_model_file, &self.mimi_model_file, &self.text_tokenizer_file]
            .iter()
            .any(|file| !std::path::Path::new(file).exists())
    }
}

pub type AppState = Arc<AppStateInner>;
pub struct AppStateInner {
    pub lm_model: moshi::lm::LmModel,
    pub mimi_model: moshi::mimi::Mimi,
    pub text_tokenizer: sentencepiece::SentencePieceProcessor,
    pub device: candle::Device,
    pub config: Config,
}

impl AppStateInner {
    fn text(
        &self,
        prev_text_token: u32,
        text_token: u32,
        config: &moshi::lm_generate_multistream::Config,
    ) -> Option<String> {
        if text_token != config.text_start_token
            && text_token != config.text_pad_token
            && text_token != config.text_eop_token
        {
            if prev_text_token == config.text_start_token {
                self.text_tokenizer.decode_piece_ids(&[text_token]).ok()
            } else {
                let prev_ids = self.text_tokenizer.decode_piece_ids(&[prev_text_token]).ok();
                let ids = self.text_tokenizer.decode_piece_ids(&[prev_text_token, text_token]).ok();
                prev_ids.and_then(|prev_ids| {
                    ids.map(|ids| {
                        if ids.len() > prev_ids.len() {
                            ids[prev_ids.len()..].to_string()
                        } else {
                            String::new()
                        }
                    })
                })
            }
        } else {
            None
        }
    }
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct SessionConfigReq {
    pub text_temperature: Option<f64>,
    pub text_topk: Option<usize>,
    pub audio_temperature: Option<f64>,
    pub audio_topk: Option<usize>,
    pub max_steps: Option<usize>,
    pub audio_seed: Option<u64>,
    pub text_seed: Option<u64>,
    pub email: Option<String>,
    pub pad_mult: Option<f32>,
    pub repetition_penalty_context: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

#[derive(serde::Serialize, Debug, Clone)]
pub struct SessionConfig {
    pub text_temperature: f64,
    pub text_topk: usize,
    pub audio_temperature: f64,
    pub audio_topk: usize,
    pub max_steps: usize,
    pub audio_seed: u64,
    pub text_seed: u64,
    pub pad_mult: Option<f32>,
    pub repetition_penalty: Option<(usize, f32)>,
    pub email: Option<String>,
    pub user_feedback: Option<usize>,
}

#[derive(serde::Serialize, Debug, Clone)]
struct SessionSummary<'a> {
    #[serde(flatten)]
    session_config: &'a SessionConfig,
    last_step_idx: usize,
    transcript: String,
    addr: Option<String>,
    lm_model_file: &'a str,
    mimi_model_file: &'a str,
    #[serde(flatten)]
    lm_config: &'a Option<moshi::lm_generate_multistream::Config>,
}

impl SessionConfigReq {
    fn into_session_config(self) -> SessionConfig {
        use rand::Rng;

        let repetition_penalty = self.repetition_penalty_context.zip(self.repetition_penalty);
        SessionConfig {
            text_temperature: self.text_temperature.unwrap_or(0.8),
            text_topk: self.text_topk.unwrap_or(250),
            text_seed: self.text_seed.unwrap_or_else(|| rand::thread_rng().gen()),
            audio_temperature: self.audio_temperature.unwrap_or(0.8),
            audio_topk: self.audio_topk.unwrap_or(250),
            audio_seed: self.audio_seed.unwrap_or_else(|| rand::thread_rng().gen()),
            email: self.email,
            user_feedback: None,
            max_steps: self.max_steps.unwrap_or(4500).min(4500),
            pad_mult: self.pad_mult,
            repetition_penalty,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct MetaData {
    text_temperature: f64,
    text_topk: usize,
    audio_temperature: f64,
    audio_topk: usize,
    pad_mult: f32,
    repetition_penalty_context: usize,
    repetition_penalty: f32,
    lm_model_file: String,
    mimi_model_file: String,
    build_info: crate::utils::BuildInfo,
    instance_name: String,
}

#[derive(Debug, Clone)]
pub enum StreamOut {
    Ready,
    InputPcm { pcm_len: usize },
    MetaData { metadata: Box<MetaData> },
    StepStart { step: usize },
    StepPostSampling { step: usize },
    Text { text: String },
    Pcm { pcm: Vec<f32> },
}

// This must be an allowed value among 120, 240, 480, 960, 1920, and 2880.
// Using a different value would result in a BadArg "invalid argument" error when calling encode.
// https://opus-codec.org/docs/opus_api-1.2/group__opus__encoder.html#ga4ae9905859cd241ef4bb5c59cd5e5309
const OPUS_ENCODER_FRAME_SIZE: usize = 960;

#[derive(Debug, Clone, Copy)]
pub enum MsgType {
    Handshake,
    Audio,
    Text,
    Control,
    Metadata,
    Error,
    Ping,
}

impl MsgType {
    pub fn from_u8(v: u8) -> Result<Self> {
        let s = match v {
            0 => MsgType::Handshake,
            1 => MsgType::Audio,
            2 => MsgType::Text,
            3 => MsgType::Control,
            4 => MsgType::Metadata,
            5 => MsgType::Error,
            6 => MsgType::Ping,
            _ => anyhow::bail!("unexpected msg type {v}"),
        };
        Ok(s)
    }

    pub fn to_u8(self) -> u8 {
        match self {
            MsgType::Handshake => 0,
            MsgType::Audio => 1,
            MsgType::Text => 2,
            MsgType::Control => 3,
            MsgType::Metadata => 4,
            MsgType::Error => 5,
            MsgType::Ping => 6,
        }
    }
}

pub struct MsgSender {
    pw: ogg::PacketWriter<'static, Vec<u8>>,
    encoder: opus::Encoder,
    out_pcm: std::collections::VecDeque<f32>,
    out_pcm_buf: Vec<u8>,
    total_data: usize,
    sender: SplitSink<ws::WebSocket, ws::Message>,
}

impl MsgSender {
    fn new(sender: SplitSink<ws::WebSocket, ws::Message>) -> Result<Self> {
        let encoder = opus::Encoder::new(24000, opus::Channels::Mono, opus::Application::Voip)?;
        // Not sure what the appropriate buffer size would be here.
        let out_pcm_buf = vec![0u8; 50_000];
        let out_pcm = std::collections::VecDeque::with_capacity(2 * OPUS_ENCODER_FRAME_SIZE);

        let all_data = Vec::new();
        let mut pw = ogg::PacketWriter::new(all_data);
        let mut head = Vec::new();
        crate::audio::write_opus_header(&mut head)?;
        pw.write_packet(head, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
        let mut tags = Vec::new();
        crate::audio::write_opus_tags(&mut tags)?;
        pw.write_packet(tags, 42, ogg::PacketWriteEndInfo::EndPage, 0)?;
        Ok(Self { pw, encoder, out_pcm, out_pcm_buf, total_data: 0, sender })
    }

    async fn send_text(&mut self, text: String) -> Result<()> {
        let msg: Vec<u8> = [&[MsgType::Text.to_u8()], text.as_bytes()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    async fn send_ready(&mut self) -> Result<()> {
        // The payload is made of two fields.
        // 1. Protocol version (`u32`) - always 0 for now.
        // 2. Model version (`u32`).
        let msg: Vec<u8> = [&[MsgType::Handshake.to_u8()], [0u8; 8].as_slice()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    async fn send_metadata(&mut self, md: Box<MetaData>) -> Result<()> {
        let bytes = serde_json::to_vec(&md)?;
        let msg: Vec<u8> = [&[MsgType::Metadata.to_u8()], bytes.as_slice()].concat();
        let msg = ws::Message::Binary(msg.into());
        self.sender.send(msg).await?;
        Ok(())
    }

    async fn send_pcm(&mut self, pcm: Vec<f32>) -> Result<()> {
        self.out_pcm.extend(pcm.iter());
        self.total_data += pcm.len();
        let nchunks = self.out_pcm.len() / OPUS_ENCODER_FRAME_SIZE;
        for _chunk_id in 0..nchunks {
            let mut chunk = Vec::with_capacity(OPUS_ENCODER_FRAME_SIZE);
            for _i in 0..OPUS_ENCODER_FRAME_SIZE {
                let v = match self.out_pcm.pop_front() {
                    None => anyhow::bail!("unexpected err popping from pcms"),
                    Some(v) => v,
                };
                chunk.push(v)
            }
            let size = self.encoder.encode_float(&chunk, &mut self.out_pcm_buf)?;
            if size > 0 {
                let msg = self.out_pcm_buf[..size].to_vec();
                self.pw.write_packet(
                    msg,
                    42,
                    ogg::PacketWriteEndInfo::EndPage,
                    self.total_data as u64,
                )?
            } else {
                tracing::error!("OPUS SIZE 0")
            }
            let data = self.pw.inner_mut();
            if !data.is_empty() {
                let msg: Vec<u8> = [&[MsgType::Audio.to_u8()], data.as_slice()].concat();
                let msg = ws::Message::Binary(msg.into());
                self.sender.send(msg).await?;
                self.sender.flush().await?;
                data.clear();
            } else {
                tracing::error!("OGG SIZE 0")
            }
        }
        Ok(())
    }
}

pub struct StreamingModel {
    state: AppState,
    device: candle::Device,
    config: moshi::lm_generate_multistream::Config,
    session_config: SessionConfig,
}

impl StreamingModel {
    fn run_with_state_asr(
        &self,
        state: &mut moshi::lm_generate_multistream::State,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        asr_delay_in_tokens: usize,
    ) -> Result<()> {
        use candle::IndexOp;

        let app_state = &self.state;

        let mut mimi = app_state.mimi_model.clone();
        let config = state.config().clone();

        mimi.reset_state();
        tracing::info!("processing loop");
        let mut prev_text_token = config.text_start_token;
        let mimi_device =
            if self.state.config.use_cpu_for_mimi { &candle::Device::Cpu } else { &self.device };
        mimi_device.synchronize()?;
        sender.send(StreamOut::Ready)?;
        while let Ok(in_pcm) = receiver.recv() {
            if in_pcm.is_empty() {
                continue;
            }
            let pcm_len = in_pcm.len();
            sender.send(StreamOut::InputPcm { pcm_len })?;
            let pcms = candle::Tensor::from_vec(in_pcm, (1, 1, pcm_len), mimi_device)?;
            let audio_tokens = mimi.encode_step(&pcms.into(), &().into())?;
            let audio_tokens = match audio_tokens.as_option() {
                None => continue,
                Some(audio_tokens) => audio_tokens,
            };
            let (_one, _codebooks, steps) = audio_tokens.dims3()?;

            for step in 0..steps {
                let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                // For the ASR, we don't provide text tokens during the initial steps except the
                // initial one.
                if state.step_idx() > 0 && state.step_idx() < asr_delay_in_tokens {
                    prev_text_token = state.step_(None, &codes, None, None, None)?;
                } else {
                    sender.send(StreamOut::StepStart { step })?;
                    let text_token = state.step(prev_text_token, &codes, None, None)?;
                    sender.send(StreamOut::StepPostSampling { step })?;
                    if let Some(text) = app_state.text(prev_text_token, text_token, &config) {
                        sender.send(StreamOut::Text { text })?;
                    }
                    prev_text_token = text_token;
                }
            }
        }
        tracing::info!("finished the processing loop");
        Ok(())
    }

    fn run_with_state(
        &self,
        state: &mut moshi::lm_generate_multistream::State,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
    ) -> Result<()> {
        use candle::IndexOp;

        let app_state = &self.state;

        let mut mimi = app_state.mimi_model.clone();
        let config = state.config().clone();

        mimi.reset_state();
        tracing::info!("processing loop");
        let mut prev_text_token = config.text_start_token;
        let mut tensor_tokens = vec![];
        let mimi_device =
            if self.state.config.use_cpu_for_mimi { &candle::Device::Cpu } else { &self.device };
        mimi_device.synchronize()?;
        sender.send(StreamOut::Ready)?;
        while let Ok(in_pcm) = receiver.recv() {
            if in_pcm.is_empty() {
                continue;
            }
            let pcm_len = in_pcm.len();
            sender.send(StreamOut::InputPcm { pcm_len })?;
            let pcms = candle::Tensor::from_vec(in_pcm, (1, 1, pcm_len), mimi_device)?;
            let audio_tokens = mimi.encode_step(&pcms.into(), &().into())?;
            let audio_tokens = match audio_tokens.as_option() {
                None => continue,
                Some(audio_tokens) => audio_tokens,
            };
            let (_one, _codebooks, steps) = audio_tokens.dims3()?;

            for step in 0..steps {
                let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                sender.send(StreamOut::StepStart { step })?;
                let text_token = state.step(prev_text_token, &codes, None, None)?;
                sender.send(StreamOut::StepPostSampling { step })?;
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    let audio_tokens = {
                        let cb = app_state.config.mimi_num_codebooks;
                        candle::Tensor::from_slice(&audio_tokens[..cb], (1, cb, 1), mimi_device)?
                    };
                    tensor_tokens.push(audio_tokens.clone());
                    let pcm = mimi.decode_step(&audio_tokens.into(), &().into())?;
                    if let Some(pcm) = pcm.as_option() {
                        let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                        sender.send(StreamOut::Pcm { pcm })?;
                    }
                }
                if let Some(text) = app_state.text(prev_text_token, text_token, &config) {
                    sender.send(StreamOut::Text { text })?;
                }
                prev_text_token = text_token;
            }
        }
        tracing::info!("finished the processing loop");
        Ok(())
    }

    fn run_with_state_mt(
        &self,
        state: &mut moshi::lm_generate_multistream::State,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
    ) -> Result<()> {
        use candle::IndexOp;

        let app_state = &self.state;

        let mut mimi = app_state.mimi_model.clone();
        let config = state.config().clone();

        mimi.reset_state();
        tracing::info!("processing loop");
        let mut prev_text_token = config.text_start_token;
        let mut tensor_tokens = vec![];
        let (tx_i, rx_i) = std::sync::mpsc::channel::<(Vec<u32>, usize)>();
        let (tx_o, rx_o) = std::sync::mpsc::channel::<Vec<u32>>();
        let sender = Arc::new(sender);
        let status = std::thread::scope(|s| {
            s.spawn({
                let mut mimi = mimi.clone();
                let sender = sender.clone();
                move || {
                    'outer: while let Ok(in_pcm) = receiver.recv() {
                        if in_pcm.is_empty() {
                            continue;
                        }
                        let pcm_len = in_pcm.len();
                        sender.send(StreamOut::InputPcm { pcm_len })?;
                        let pcms = candle::Tensor::from_vec(
                            in_pcm,
                            (1, 1, pcm_len),
                            &candle::Device::Cpu,
                        )?;
                        let audio_tokens = mimi.encode_step(&pcms.into(), &().into())?;
                        let audio_tokens = match audio_tokens.as_option() {
                            None => continue,
                            Some(audio_tokens) => audio_tokens,
                        };
                        let (_one, _codebooks, steps) = audio_tokens.dims3()?;
                        for step in 0..steps {
                            let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                            if tx_i.send((codes, step)).is_err() {
                                break 'outer;
                            }
                        }
                    }
                    Ok::<_, anyhow::Error>(())
                }
            });
            s.spawn({
                let cb = app_state.config.mimi_num_codebooks;
                let sender = sender.clone();
                move || {
                    while let Ok(audio_tokens) = rx_o.recv() {
                        let audio_tokens = {
                            candle::Tensor::from_slice(
                                &audio_tokens[..cb],
                                (1, cb, 1),
                                &candle::Device::Cpu,
                            )?
                        };
                        tensor_tokens.push(audio_tokens.clone());
                        let pcm = mimi.decode_step(&audio_tokens.into(), &().into())?;
                        if let Some(pcm) = pcm.as_option() {
                            let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                            sender.send(StreamOut::Pcm { pcm })?;
                        }
                    }
                    Ok::<_, anyhow::Error>(())
                }
            });
            sender.send(StreamOut::Ready)?;
            while let Ok((codes, step)) = rx_i.recv() {
                tracing::info!("received codes");
                sender.send(StreamOut::StepStart { step })?;
                let text_token = state.step(prev_text_token, &codes, None, None);
                sender.send(StreamOut::StepPostSampling { step })?;
                tracing::info!(?text_token, "codes");
                if text_token.is_err() {
                    drop(rx_i);
                    drop(tx_o);
                    break;
                }
                let text_token = text_token?;
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    tx_o.send(audio_tokens)?
                }
                if let Some(text) = app_state.text(prev_text_token, text_token, &config) {
                    sender.send(StreamOut::Text { text })?;
                }
                prev_text_token = text_token;
            }
            Ok::<_, anyhow::Error>(())
        });
        match status {
            Ok(()) => tracing::info!("finished the processing loop"),
            Err(err) => tracing::error!(?err, "processing loop"),
        };
        Ok(())
    }

    pub fn new(state: &AppState, session_config: SessionConfigReq) -> Self {
        let config = match state.config.lm_config.as_ref() {
            None => moshi::lm_generate_multistream::Config::v0_1(),
            Some(config) => config.clone(),
        };
        let session_config = session_config.into_session_config();
        Self { state: state.clone(), device: state.device.clone(), config, session_config }
    }

    pub fn run(
        &self,
        receiver: std::sync::mpsc::Receiver<Vec<f32>>,
        sender: tokio::sync::mpsc::UnboundedSender<StreamOut>,
        addr: Option<String>,
    ) -> Result<()> {
        let app_state = &self.state;
        let (repetition_penalty_context, repetition_penalty) =
            self.session_config.repetition_penalty.unwrap_or((32, 1.));
        let metadata = MetaData {
            text_temperature: self.session_config.text_temperature,
            text_topk: self.session_config.text_topk,
            audio_temperature: self.session_config.audio_temperature,
            audio_topk: self.session_config.audio_topk,
            pad_mult: self.session_config.pad_mult.unwrap_or(0.),
            repetition_penalty,
            repetition_penalty_context,
            lm_model_file: self.state.config.lm_model_file.to_string(),
            mimi_model_file: self.state.config.mimi_model_file.to_string(),
            build_info: crate::utils::BuildInfo::new(),
            instance_name: self.state.config.instance_name.to_string(),
        };
        sender.send(StreamOut::MetaData { metadata: Box::new(metadata) })?;
        let lm_model = app_state.lm_model.clone();
        let audio_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            self.session_config.audio_seed,
            candle_transformers::generation::Sampling::TopK {
                k: self.session_config.audio_topk,
                temperature: self.session_config.audio_temperature,
            },
        );
        let text_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            self.session_config.text_seed,
            candle_transformers::generation::Sampling::TopK {
                k: self.session_config.text_topk,
                temperature: self.session_config.text_temperature,
            },
        );
        let mut state = moshi::lm_generate_multistream::State::new(
            lm_model,
            self.session_config.max_steps,
            audio_lp,
            text_lp,
            self.session_config.pad_mult,
            self.session_config.repetition_penalty,
            None,
            self.config.clone(),
        );

        // We want to log the output even if the run function returns an error.
        let run_result = if self.state.config.use_cpu_for_mimi {
            self.run_with_state_mt(&mut state, receiver, sender)
        } else if let Some(asr_delay_in_tokens) = self.state.config.asr_delay_in_tokens {
            self.run_with_state_asr(&mut state, receiver, sender, asr_delay_in_tokens)
        } else {
            self.run_with_state(&mut state, receiver, sender)
        };
        {
            let text_tokens = state.text_tokens(false);
            let transcript = {
                let text_tokens = text_tokens
                    .iter()
                    .filter_map(|v| {
                        let v = *v;
                        if v != moshi::lm_generate_multistream::UNGENERATED
                            && v != self.config.text_pad_token
                            && v != self.config.text_eop_token
                            && v != self.config.text_start_token
                        {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                self.state
                    .text_tokenizer
                    .decode_piece_ids(&text_tokens)
                    .unwrap_or_else(|_| String::new())
            };
            let audio_tokens = state.audio_tokens(false);
            let audio_tokens = audio_tokens
                .iter()
                .map(|v| {
                    v.iter()
                        .map(|v| {
                            if *v == moshi::lm_generate_multistream::UNGENERATED {
                                -1
                            } else {
                                *v as i64
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let text_tokens = candle::Tensor::new(text_tokens, &candle::Device::Cpu)?
                .to_dtype(candle::DType::I64)?;
            let audio_tokens = candle::Tensor::new(audio_tokens, &candle::Device::Cpu)?;
            let since_epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
            let (secs, us) = (since_epoch.as_secs(), since_epoch.subsec_micros());
            let log_dir = &app_state.config.log_dir;
            let base_path = format!("{log_dir}/{}-{secs}-{us}", app_state.config.instance_name);
            let json_filename = format!("{base_path}.json");
            let json_content = serde_json::to_string_pretty(&SessionSummary {
                session_config: &self.session_config,
                last_step_idx: state.step_idx(),
                transcript,
                addr,
                mimi_model_file: &self.state.config.mimi_model_file,
                lm_model_file: &self.state.config.lm_model_file,
                lm_config: &self.state.config.lm_config,
            })?;
            std::fs::write(json_filename, json_content)?;
            let st_filename = format!("{base_path}.safetensors");
            let st_content =
                std::collections::HashMap::from([("text", text_tokens), ("audio", audio_tokens)]);
            candle::safetensors::save(&st_content, st_filename)?;
        }
        run_result
    }
}

type Handle = tokio::task::JoinHandle<Result<()>>;

fn spawn_recv_loops(
    mut receiver: SplitStream<ws::WebSocket>,
    sender: std::sync::mpsc::Sender<Vec<f32>>,
) -> Result<(Handle, Handle)> {
    use tokio::io::AsyncWriteExt;

    let (mut tx, rx) = tokio::io::duplex(100_000);
    let mut pr = ogg::reading::async_api::PacketReader::new(rx);
    let mut decoder = opus::Decoder::new(24000, opus::Channels::Mono)?;
    let handle1 = tokio::spawn({
        async move {
            loop {
                match receiver.next().await {
                    None => {
                        // The close logic is that if this loop exits, then tx gets dropped so pr
                        // gets closed and the second thread gets dropped resulting in sender
                        // getting dropped.
                        break;
                    }
                    Some(v) => {
                        let v = v?.into_data();
                        if v.is_empty() {
                            continue;
                        }
                        let msg_type = MsgType::from_u8(v[0])?;
                        match msg_type {
                            MsgType::Metadata => {}
                            MsgType::Handshake => {}
                            MsgType::Control => {}
                            MsgType::Text => {}
                            MsgType::Error => {}
                            MsgType::Ping => {}
                            MsgType::Audio => tx.write_all(&v[1..]).await?,
                        }
                    }
                }
            }
            tracing::info!("socket closed");
            Ok::<_, anyhow::Error>(())
        }
    });
    let handle2 = tokio::spawn(async move {
        // TODO: dynamic sizing?
        let mut pcm_buf = vec![0f32; 24_000 * 10];
        let mut size_in_buf = 0;
        loop {
            match pr.next().await {
                None => {
                    break;
                }
                Some(packet) => {
                    let packet = packet?;
                    if packet.data.starts_with(b"OpusHead") || packet.data.starts_with(b"OpusTags")
                    {
                        continue;
                    }
                    let read_size = decoder.decode_float(
                        &packet.data,
                        &mut pcm_buf[size_in_buf..],
                        /* Forward Error Correction */ false,
                    )?;
                    size_in_buf += read_size;
                    // flush the data every half timestep
                    if size_in_buf >= 24_000 / 25 {
                        if sender.send(pcm_buf[..size_in_buf].to_vec()).is_err() {
                            break;
                        }
                        size_in_buf = 0;
                    }
                }
            }
        }
        tracing::info!("decoder closed");
        Ok::<_, anyhow::Error>(())
    });
    Ok((handle1, handle2))
}

async fn sender_loop(
    mut stream_out_rx: tokio::sync::mpsc::UnboundedReceiver<StreamOut>,
    mut sender: MsgSender,
) -> Result<()> {
    // It is important for the recv here to be an async enabled one. Otherwise this could lead
    // to some weird deadlocks.
    while let Some(v) = stream_out_rx.recv().await {
        match v {
            StreamOut::Pcm { pcm } => sender.send_pcm(pcm).await?,
            StreamOut::Ready => sender.send_ready().await?,
            StreamOut::MetaData { metadata } => sender.send_metadata(metadata).await?,
            StreamOut::Text { text } => sender.send_text(text).await?,
            StreamOut::InputPcm { .. }
            | StreamOut::StepStart { .. }
            | StreamOut::StepPostSampling { .. } => {}
        }
    }
    Ok::<_, anyhow::Error>(())
}

pub async fn handle_socket(
    socket: ws::WebSocket,
    sm: StreamingModel,
    addr: Option<String>,
) -> Result<()> {
    tracing::info!("accepted websocket connection");
    let (sender, receiver) = socket.split();
    let sender = MsgSender::new(sender)?;

    tracing::info!("starting streaming");

    let (in_pcm_tx, in_pcm_rx) = std::sync::mpsc::channel();
    let (stream_out_tx, stream_out_rx) = tokio::sync::mpsc::unbounded_channel();
    let (loop1, loop2) = spawn_recv_loops(receiver, in_pcm_tx)?;
    std::thread::spawn(move || {
        if let Err(err) = sm.run(in_pcm_rx, stream_out_tx, addr) {
            tracing::error!("{err}")
        }
    });
    let sender_loop = tokio::spawn(async move {
        match sender_loop(stream_out_rx, sender).await {
            Ok(()) => tracing::info!("sender closed"),
            Err(err) => {
                // Using the Display trait rather than the Debug one so as not to include the backtrace.
                let err = format!("{err}");
                tracing::info!(err, "sender err")
            }
        }
    });

    let sleep = tokio::time::sleep(std::time::Duration::from_secs(360));
    tokio::pin!(sleep);
    // select should ensure that all the threads get aborted on timeout.
    tokio::select! {
        _ = &mut sleep => {
            tracing::error!("reached timeout");
        }
        r = loop1 => {
            tracing::error!(?r, "loop1 ended")
        }
        r = loop2 => {
            tracing::error!(?r, "loop2 ended")
        }
        r = sender_loop => {
            tracing::error!(?r, "sender loop ended")
        }
    }
    Ok(())
}
