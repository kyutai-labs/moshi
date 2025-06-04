// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::{Context, Result};
use axum::extract::ws;
use candle::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use moshi::tts_streaming::Speaker;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WordWithTimestamps {
    pub text: String,
    pub start_s: f64,
    pub stop_s: f64,
}

pub struct Model {
    lm: moshi::lm::LmModel,
    audio_tokenizer: moshi::mimi::Mimi,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
    speaker_encoder: moshi::tts_streaming::SpeakerEncoder,
    ca_srcs: std::collections::HashMap<String, Tensor>,
    tts_config: moshi::tts_streaming::Config,
    instance_name: String,
    voice_dir: std::path::PathBuf,
    log_dir: std::path::PathBuf,
    // Dummy way to ensure that only a single inference can happen.
    pub(crate) mutex: tokio::sync::Mutex<()>,
}

pub enum Encoder {
    OggOpus(kaudio::ogg_opus::Encoder),
    OggOpusMessagePack(kaudio::ogg_opus::Encoder),
    Pcm,
    PcmMessagePack,
}

enum LogMessage {
    Text(String),
    Slice(u32, Tensor),
}

#[derive(serde::Serialize)]
struct QueryWithTexts<'a, Q: serde::Serialize> {
    #[serde(flatten)]
    query: &'a Q,
    texts: Vec<String>,
}

#[derive(Clone)]
struct LogSender(std::sync::mpsc::Sender<LogMessage>);
struct Logger(std::sync::mpsc::Receiver<LogMessage>);

fn logger() -> (LogSender, Logger) {
    let (log_tx, log_rx) = std::sync::mpsc::channel();
    (LogSender(log_tx), Logger(log_rx))
}

impl LogSender {
    fn send(&self, msg: LogMessage) {
        let _err = self.0.send(msg);
    }

    fn send_text(&self, text: String) {
        self.send(LogMessage::Text(text));
    }

    fn send_slice(&self, idx: u32, slice: Tensor) {
        self.send(LogMessage::Slice(idx, slice));
    }
}

impl Logger {
    fn save<P: AsRef<std::path::Path>, T: serde::Serialize>(
        self,
        query: &T,
        log_dir: P,
        instance_name: &str,
    ) -> Result<()> {
        // Use log_rx.iter() to wait on the process loop being done.

        let mut text_tokens = vec![];
        let mut audio_tokens = vec![];
        let mut texts = vec![];
        for elem in self.0.into_iter() {
            match elem {
                LogMessage::Text(text) => {
                    texts.push(text);
                }
                LogMessage::Slice(idx, slice) => {
                    audio_tokens.push(slice);
                    text_tokens.push(idx);
                }
            }
        }
        let text_tokens = text_tokens.into_iter().map(|v| (v, Speaker::Main)).collect::<Vec<_>>();
        let audio_tokens = Tensor::cat(&audio_tokens, candle::D::Minus1)?;
        let since_epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
        let (secs, us) = (since_epoch.as_secs(), since_epoch.subsec_micros());
        let base_path = log_dir.as_ref().join(format!("{}-tts-{secs}-{us}", instance_name));
        let json_filename = base_path.with_extension("json");
        let query = QueryWithTexts { query, texts };
        let json_content = serde_json::to_string_pretty(&query)?;
        std::fs::write(json_filename, json_content)?;
        let st_filename = base_path.with_extension("safetensors");
        let text_tokens: Vec<_> = text_tokens.iter().map(|v| v.0 as i64).collect();
        let text_len = text_tokens.len();
        let text_tokens = candle::Tensor::from_vec(text_tokens, text_len, &candle::Device::Cpu)?
            .to_dtype(DType::I64)?;
        let audio_tokens = audio_tokens.to_device(&Device::Cpu)?.to_dtype(DType::I64)?;
        let st_content =
            std::collections::HashMap::from([("text", text_tokens), ("audio", audio_tokens)]);
        candle::safetensors::save(&st_content, st_filename)?;
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum OutMsg {
    Text { text: String, start_s: f64, stop_s: f64 },
    Audio { pcm: Vec<f32> },
    OggOpus { data: Vec<u8> },
    Error { message: String },
    Ready,
}

impl Encoder {
    pub fn new(format: crate::StreamingOutput) -> Result<Self> {
        match format {
            crate::StreamingOutput::OggOpus => Self::ogg_opus(24000),
            crate::StreamingOutput::OggOpusMessagePack => Self::ogg_opus_message_pack(24000),
            crate::StreamingOutput::Pcm => Ok(Self::pcm()),
            crate::StreamingOutput::PcmMessagePack => Ok(Self::pcm_message_pack()),
        }
    }

    fn ogg_opus(sample_rate: usize) -> Result<Self> {
        Ok(Self::OggOpus(kaudio::ogg_opus::Encoder::new(sample_rate)?))
    }

    fn ogg_opus_message_pack(sample_rate: usize) -> Result<Self> {
        Ok(Self::OggOpusMessagePack(kaudio::ogg_opus::Encoder::new(sample_rate)?))
    }

    fn pcm_message_pack() -> Self {
        Self::PcmMessagePack
    }

    fn pcm() -> Self {
        Self::Pcm
    }

    pub fn header(&self) -> Result<Option<Vec<u8>>> {
        let header = match self {
            Self::OggOpus(oo) => Some(oo.header_data().to_vec()),
            Self::OggOpusMessagePack(oo) => {
                use serde::Serialize;
                let msg = OutMsg::OggOpus { data: oo.header_data().to_vec() };
                let mut buf = vec![];
                msg.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                Some(buf)
            }
            Self::Pcm => None,
            Self::PcmMessagePack => None,
        };
        Ok(header)
    }

    pub fn encode_word(&self, wwts: WordWithTimestamps) -> Result<Option<Vec<u8>>> {
        if wwts.text.is_empty() {
            return Ok(None);
        }
        let buf = match self {
            Self::Pcm | Self::OggOpus(_) => None,
            Self::OggOpusMessagePack(_) | Self::PcmMessagePack => {
                use serde::Serialize;
                let mut buf = vec![];
                OutMsg::Text { text: wwts.text, start_s: wwts.start_s, stop_s: wwts.stop_s }
                    .serialize(
                        &mut rmp_serde::Serializer::new(&mut buf)
                            .with_human_readable()
                            .with_struct_map(),
                    )?;
                Some(buf)
            }
        };
        Ok(buf)
    }

    pub fn encode(&mut self, pcm: Vec<f32>) -> Result<Vec<u8>> {
        use serde::Serialize;
        let buf = match self {
            Self::OggOpus(oo) => oo.encode_page(&pcm)?,
            Self::OggOpusMessagePack(oo) => {
                let data = oo.encode_page(&pcm)?;
                let mut buf = vec![];
                OutMsg::OggOpus { data }.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                buf
            }
            Self::PcmMessagePack => {
                let mut buf = vec![];
                OutMsg::Audio { pcm }.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                buf
            }
            Self::Pcm => {
                use byteorder::ByteOrder;
                let mut buf = vec![0u8; std::mem::size_of_val(pcm.as_slice())];
                byteorder::LittleEndian::write_f32_into(&pcm, &mut buf);
                buf
            }
        };
        Ok(buf)
    }

    pub fn encode_msg(&mut self, msg: OutMsg) -> Result<Option<Vec<u8>>> {
        use serde::Serialize;
        let buf = match self {
            Self::OggOpus(_) | Self::Pcm => None,
            Self::OggOpusMessagePack(_) | Self::PcmMessagePack => {
                let mut buf = vec![];
                msg.serialize(
                    &mut rmp_serde::Serializer::new(&mut buf)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                Some(buf)
            }
        };
        Ok(buf)
    }
}

impl Model {
    pub fn new(tts: &crate::TtsConfig, config: &crate::Config, dev: &Device) -> Result<Self> {
        let dtype = dev.bf16_default_to_f32();
        let model_config = &tts.model;
        let audio_codebooks = model_config.audio_codebooks;
        let audio_tokenizer =
            moshi::mimi::load(&tts.audio_tokenizer_file, Some(audio_codebooks), dev)?;
        let speaker_tokenizer = if tts.speaker_tokenizer_file == tts.audio_tokenizer_file {
            audio_tokenizer.clone()
        } else if tts.speaker_tokenizer_file.is_empty() {
            let vb_lm = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&tts.lm_model_file], DType::F32, dev)?
            };
            let cfg = moshi::mimi::Config::v0_1(None);
            moshi::mimi::Mimi::new(
                cfg,
                vb_lm.pp("condition_provider.conditioners.speaker_wavs.compression_model"),
            )?
        } else {
            moshi::mimi::load(&tts.speaker_tokenizer_file, None, dev)?
        };
        let vb_lm =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&tts.lm_model_file], dtype, dev)? };
        let speaker_encoder = moshi::tts_streaming::SpeakerEncoder::new(
            speaker_tokenizer,
            tts.generation.speaker_cond_dim,
            tts.generation.speaker_cond_n_speakers,
            dtype,
            vb_lm.to_dtype(DType::F32),
        )?;
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&tts.text_tokenizer_file)
            .with_context(|| tts.text_tokenizer_file.clone())?;
        let mut ca_srcs = std::collections::HashMap::new();
        for (name, path) in tts.voices.iter() {
            let ca_src = match candle::safetensors::load(path, dev)?.get("ca_src") {
                Some(ca_src) => ca_src.clone(),
                None => anyhow::bail!("missing ca_src tensor in {path}"),
            };
            let ca_src = ca_src.narrow(0, 0, 1)?.to_dtype(dtype)?;
            ca_srcs.insert(name.to_string(), ca_src);
        }
        let lm = moshi::lm::LmModel::new(
            model_config,
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        Ok(Self {
            lm,
            audio_tokenizer,
            text_tokenizer: std::sync::Arc::new(text_tokenizer),
            speaker_encoder,
            ca_srcs,
            tts_config: tts.generation.clone(),
            instance_name: config.instance_name.to_string(),
            log_dir: config.log_dir.clone().into(),
            voice_dir: tts.voice_dir.clone().into(),
            mutex: tokio::sync::Mutex::new(()),
        })
    }

    pub async fn handle_socket(
        &self,
        socket: ws::WebSocket,
        query: crate::TtsStreamingQuery,
    ) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};

        let _guard = self.mutex.lock().await;
        let config = &self.tts_config;
        let (log_tx, log_rx) = logger();
        let log_tx2 = log_tx.clone();
        let sampling = if query.temperature <= 0. || query.top_k <= 1 {
            candle_transformers::generation::Sampling::ArgMax
        } else {
            candle_transformers::generation::Sampling::TopK {
                k: query.top_k,
                temperature: query.temperature,
            }
        };

        let text_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
            query.seed,
            sampling.clone(),
        );
        let audio_lp =
            candle_transformers::generation::LogitsProcessor::from_sampling(query.seed, sampling);
        let conditions = match self.lm.condition_provider() {
            None => None,
            Some(cp) => {
                let conditions = cp.condition_lut("control", "also_good")?;
                tracing::info!(?conditions, "generated conditions");
                Some(conditions)
            }
        };

        let mut last_text_token = config.text_start_token;
        let ca_src = self.voice_ca_src(query.voice.as_ref(), query.voices.as_ref())?;
        ca_src.device().synchronize()?;
        let ca_src = if query.cfg_alpha.is_some() {
            let lp = self.speaker_encoder.empty()?;
            Tensor::cat(&[ca_src, lp], 0)?
        } else {
            ca_src
        };
        let max_seq_len = query.max_seq_len.unwrap_or(2048);
        let mut state = moshi::tts_streaming::State::new(
            self.lm.clone(),
            Some(moshi::transformer::CaSrc::Tokens(ca_src)),
            max_seq_len,
            audio_lp,
            text_lp,
            query.cfg_alpha,
            config.clone(),
        );
        let text_tokenizer = self.text_tokenizer.clone();

        let (mut sender, mut receiver) = socket.split();
        let (in_tx, in_rx) = std::sync::mpsc::channel();
        let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel();
        let text_bos_token = state.config().text_bos_token;
        let recv_loop = tokio::task::spawn(async move {
            let mut inserted_bos = false;
            while let Some(msg) = receiver.next().await {
                let msg = match msg? {
                    ws::Message::Text(x) => x,
                    ws::Message::Binary(x) => {
                        // End of stream, we do not exit the loop so as not to close
                        // the connection.
                        if x.as_ref() == b"\0" {
                            log::info!("received end of stream");
                            in_tx.send(None)?;
                        }
                        continue;
                    }
                    // ping messages are automatically answered by tokio-tungstenite as long as
                    // the connection is read from.
                    ws::Message::Ping(_) | ws::Message::Pong(_) => continue,
                    ws::Message::Close(_) => break,
                };

                let msg: String = msg.to_string();
                for word in msg.split(' ') {
                    if word.is_empty() {
                        continue;
                    }
                    let mut word_tokens: Vec<_> =
                        text_tokenizer.encode(word)?.into_iter().map(|v| v.id).collect();
                    if !inserted_bos {
                        inserted_bos = true;
                        word_tokens.insert(0, text_bos_token)
                    }
                    log_tx2.send_text(word.to_string());
                    in_tx.send(Some(word_tokens))?;
                }
            }
            tracing::info!("recv loop exited - connection closed");
            Ok::<(), anyhow::Error>(())
        });
        let mut audio_tokenizer = self.audio_tokenizer.clone();
        audio_tokenizer.reset_state();
        let text_tokenizer = self.text_tokenizer.clone();
        let format = query.format;
        let process_loop = tokio::task::spawn_blocking(move || {
            let err = (|| {
                tracing::info!("starting the inference loop");
                let text_audio_delay_in_tokens = state.config().text_audio_delay_in_tokens;
                let acoustic_delay = state.config().acoustic_delay;
                let text_eop_token = state.config().text_eop_token;
                let text_pad_token = state.config().text_pad_token;
                let extra_steps = state.config().extra_steps;

                let mut token_idx = 0;
                let mut step_past_last_token = 0;
                // Start with an empty list to trigger the first bos.
                let mut word_tokens = Some(vec![]);

                let mut encoder = Encoder::new(format)?;
                if let Some(header) = encoder.header()? {
                    out_tx.send(header)?
                }
                let mut last_epad_index = 0usize;
                for step_idx in 0..max_seq_len {
                    let allowed_tokens = match word_tokens.as_ref() {
                        None => {
                            step_past_last_token += 1;
                            if step_past_last_token > extra_steps + text_audio_delay_in_tokens {
                                break;
                            }
                            moshi::tts_streaming::AllowedTokens::Pad
                        }
                        Some(word_tokens) => match word_tokens.get(token_idx) {
                            None => moshi::tts_streaming::AllowedTokens::PadOrEpad,
                            Some(id) => moshi::tts_streaming::AllowedTokens::Text(*id),
                        },
                    };
                    last_text_token =
                        state.step(last_text_token, allowed_tokens, conditions.as_ref())?;
                    if last_text_token == text_eop_token {
                        if let Some(vs) = word_tokens {
                            if let Ok(text) = text_tokenizer.decode_piece_ids(&vs) {
                                let start_s = last_epad_index as f64 / 12.5;
                                let stop_s = step_idx as f64 / 12.5;
                                let wwts = WordWithTimestamps { text, start_s, stop_s };
                                if let Some(oo) = encoder.encode_word(wwts)? {
                                    out_tx.send(oo)?;
                                }
                            }
                        }
                        last_epad_index = step_idx;
                        word_tokens = in_rx.recv()?;
                        if word_tokens.is_none() {
                            // We teacher force a pad instead of tho eop for the last word.
                            state.overwrite_last_text_token(text_pad_token)?;
                        }
                        token_idx = 0;
                    } else if last_text_token != text_pad_token {
                        token_idx += 1;
                    }
                    if let Some(audio_tokens) = state.last_audio_tokens() {
                        let cb = audio_tokens.len();
                        let audio_tokens =
                            candle::Tensor::from_vec(audio_tokens, (1, cb, 1), state.device())?;
                        if step_idx >= text_audio_delay_in_tokens + acoustic_delay {
                            let pcm = audio_tokenizer
                                .decode_step(&audio_tokens.clone().into(), &().into())?;
                            if let Some(pcm) = pcm.as_option() {
                                let pcm = pcm.flatten_all()?.to_vec1::<f32>()?;
                                let oo = encoder.encode(pcm)?;
                                out_tx.send(oo)?;
                            }
                        }
                        log_tx.send_slice(last_text_token, audio_tokens)
                    } else {
                        let cb = state.audio_codebooks();
                        let audio_tokens =
                            candle::Tensor::zeros((1, cb, 1), DType::U32, state.device())?;
                        log_tx.send_slice(last_text_token, audio_tokens)
                    }
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
                Ok::<(), anyhow::Error>(())
            })();
            match err {
                Err(err) => tracing::error!(?err, "process loop exited"),
                Ok(()) => tracing::info!("process loop exited"),
            }
        });
        let send_loop = tokio::task::spawn(async move {
            use tokio::time::{timeout, Duration};
            loop {
                // The recv method is cancel-safe so can be wrapped in a timeout.
                let msg = timeout(Duration::from_secs(10), out_rx.recv()).await;
                let msg = match msg {
                    Ok(Some(msg)) => ws::Message::binary(msg),
                    Ok(None) => break,
                    Err(_) => ws::Message::Ping(vec![].into()),
                };
                sender.send(msg).await?;
            }
            tracing::info!("send loop exited - connection closed");
            sender.close().await?;
            tracing::info!("send loop exited - connection really closed");
            drop(sender);
            Ok::<(), anyhow::Error>(())
        });
        // select should ensure that all the threads get aborted on timeout.
        // TODO(laurent): this actually doesn't work as expected, and the background threads don't
        // appear to be cancelled properly (at least the websocket connection remains open.
        let sleep = tokio::time::sleep(std::time::Duration::from_secs(360));
        tokio::pin!(sleep);
        tokio::select! {
            _ = sleep => {
                tracing::error!("reached timeout");
            }
            res = recv_loop => {
                match res {
                    Err(err) => tracing::error!(?err, "recv loop ended"),
                    Ok(Err(err)) => tracing::error!(?err, "recv loop err"),
                    Ok(Ok(())) => tracing::info!("recv loop ended"),
                }
            }
            p = process_loop => {
                match p {
                    Err(err) => tracing::error!(?err, "process loop ended"),
                    Ok(()) => tracing::info!("process loop ended"),
                }
            }
            res = send_loop => {
                match res {
                    Err(err) => tracing::error!(?err, "send loop ended"),
                    Ok(Err(err)) => tracing::error!(?err, "send loop err"),
                    Ok(Ok(())) => tracing::info!("send loop ended"),
                }
            }
        }
        tracing::info!("exiting handle-socket");
        if let Err(err) = log_rx.save(&query, &self.log_dir, &self.instance_name) {
            tracing::error!(?err, "cannot save logs")
        };
        Ok(())
    }

    pub fn voice_ca_src(
        &self,
        voice: Option<&String>,
        voices: Option<&Vec<String>>,
    ) -> Result<Tensor> {
        match (voice, voices) {
            (None, None) => anyhow::bail!("either voice or voices has to be set"),
            (Some(_), Some(_)) => {
                anyhow::bail!("voice and voices should not be set at the same time")
            }
            (Some(voice), None) => match self.ca_srcs.get(voice) {
                None => {
                    let voice_dir = std::fs::canonicalize(&self.voice_dir)?;
                    let mut pcms = vec![];
                    let (voice, speaker_cond_start_s) = match voice.split_once('+') {
                        None => (voice.as_str(), 0.0),
                        Some((v, delay)) => {
                            let delay = match delay.parse::<f64>() {
                                Ok(delay) => delay,
                                Err(_) => anyhow::bail!(
                                    "unexpected format for delay in {voice}: '{delay}'"
                                ),
                            };
                            (v, delay)
                        }
                    };
                    let path = std::fs::canonicalize(voice_dir.join(voice))?;
                    if !path.starts_with(&voice_dir) {
                        tracing::error!(?voice_dir, ?path, "unable to access voice file");
                        anyhow::bail!("unknown voice file '{voice}'")
                    }
                    let pcm = speaker_pcm(
                        self.speaker_encoder.sample_rate(),
                        speaker_cond_start_s,
                        self.tts_config.speaker_cond_duration_s,
                        path,
                        self.lm.device(),
                    )?;
                    pcms.push(pcm.clone());
                    pcms.push(pcm);
                    Ok(self.speaker_encoder.encode(&pcms)?)
                }
                Some(v) => Ok(v.clone()),
            },
            (None, Some(voices)) => {
                let voice_dir = std::fs::canonicalize(&self.voice_dir)?;
                let mut pcms = vec![];
                for voice in voices.iter() {
                    let (voice, speaker_cond_start_s) = match voice.split_once('+') {
                        None => (voice.as_str(), 0.0),
                        Some((v, delay)) => {
                            let delay = match delay.parse::<f64>() {
                                Ok(delay) => delay,
                                Err(_) => anyhow::bail!(
                                    "unexpected format for delay in {voice}: '{delay}'"
                                ),
                            };
                            (v, delay)
                        }
                    };
                    let path = std::fs::canonicalize(voice_dir.join(voice))?;
                    if !path.starts_with(&voice_dir) {
                        tracing::error!(?voice_dir, ?path, "unable to access voice file");
                        anyhow::bail!("unknown voice file '{voice}'")
                    }
                    let pcm = speaker_pcm(
                        self.speaker_encoder.sample_rate(),
                        speaker_cond_start_s,
                        self.tts_config.speaker_cond_duration_s,
                        path,
                        self.lm.device(),
                    )?;
                    pcms.push(pcm)
                }
                Ok(self.speaker_encoder.encode(&pcms)?)
            }
        }
    }

    pub fn run(&self, query: &crate::TtsQuery) -> Result<(Vec<u8>, Vec<WordWithTimestamps>)> {
        let config = &self.tts_config;
        let text_audio_delay_in_tokens = config.text_audio_delay_in_tokens;
        let text_bos_token = config.text_bos_token;
        let text_eos_token = config.text_eos_token;
        let text_eop_token = config.text_eop_token;
        let text_pad_token = config.text_pad_token;
        let mut prompt = moshi::tts_streaming::tokenize_prompt(
            &query.text,
            text_bos_token,
            text_eos_token,
            |s| self.text_tokenizer.encode(s).map(|v| v.into_iter().map(|v| v.id).collect()),
        )?;
        // Insert an empty word to start with and trigger the first bos.
        prompt.insert(0, (vec![], Speaker::Other));
        tracing::info!(?prompt, "starting tts");
        let mut transcript = vec![];
        let (log_tx, log_rx) = logger();
        let all_audio_tokens = {
            let start_time = std::time::Instant::now();
            let sampling = if query.temperature <= 0. || query.top_k <= 1 {
                candle_transformers::generation::Sampling::ArgMax
            } else {
                candle_transformers::generation::Sampling::TopK {
                    k: query.top_k,
                    temperature: query.temperature,
                }
            };

            let text_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
                query.seed,
                sampling.clone(),
            );
            let audio_lp = candle_transformers::generation::LogitsProcessor::from_sampling(
                query.seed, sampling,
            );
            let conditions = match self.lm.condition_provider() {
                None => None,
                Some(cp) => {
                    let conditions = cp.condition_lut("control", "also_good")?;
                    tracing::info!(?conditions, "generated conditions");
                    Some(conditions)
                }
            };

            let mut last_text_token = config.text_start_token;
            let ca_src = self.voice_ca_src(query.voice.as_ref(), query.voices.as_ref())?;
            let ca_src = if query.cfg_alpha.is_some() {
                let lp = self.speaker_encoder.empty()?;
                Tensor::cat(&[ca_src, lp], 0)?
            } else {
                ca_src
            };
            let max_seq_len = query.max_seq_len.unwrap_or(2048);
            let config = config.clone();
            let mut state = moshi::tts_streaming::State::new(
                self.lm.clone(),
                Some(moshi::transformer::CaSrc::Tokens(ca_src)),
                max_seq_len,
                audio_lp,
                text_lp,
                query.cfg_alpha,
                config.clone(),
            );
            let mut all_audio_tokens = vec![];
            tracing::info!("starting the inference loop");
            let mut word_idx = 0;
            let mut token_idx = 0;
            let mut step_past_last_token = 0;
            let mut last_epad_index = 0usize;
            for step_idx in 0..max_seq_len {
                let word_tokens = prompt.get(word_idx);
                let allowed_tokens = match word_tokens.as_ref() {
                    None => {
                        step_past_last_token += 1;
                        if step_past_last_token > 5 + text_audio_delay_in_tokens {
                            break;
                        }
                        moshi::tts_streaming::AllowedTokens::Pad
                    }
                    Some(word_tokens) => match word_tokens.0.get(token_idx) {
                        None => moshi::tts_streaming::AllowedTokens::PadOrEpad,
                        Some(id) => moshi::tts_streaming::AllowedTokens::Text(*id),
                    },
                };
                last_text_token =
                    state.step(last_text_token, allowed_tokens, conditions.as_ref())?;
                if last_text_token == text_eop_token {
                    if let Some(vs) = word_tokens {
                        if let Ok(text) = self.text_tokenizer.decode_piece_ids(&vs.0) {
                            let start_s = last_epad_index as f64 / 12.5;
                            let stop_s = step_idx as f64 / 12.5;
                            transcript.push(WordWithTimestamps { text, start_s, stop_s })
                        }
                    }
                    last_epad_index = step_idx;
                    word_idx += 1;
                    token_idx = 0;
                } else if last_text_token != text_pad_token {
                    token_idx += 1;
                }
                if let Some(audio_tokens) = state.last_audio_tokens() {
                    let cb = audio_tokens.len();
                    let audio_tokens =
                        candle::Tensor::from_vec(audio_tokens, (1, cb, 1), state.device())?;
                    if step_idx >= text_audio_delay_in_tokens {
                        all_audio_tokens.push(audio_tokens.clone())
                    }
                    log_tx.send_slice(last_text_token, audio_tokens)
                } else {
                    let cb = state.audio_codebooks();
                    let audio_tokens =
                        candle::Tensor::zeros((1, cb, 1), DType::U32, state.device())?;
                    log_tx.send_slice(last_text_token, audio_tokens)
                }
            }
            let dt = start_time.elapsed().as_secs_f64();
            let total = all_audio_tokens.len();
            tracing::info!(
                "processed {total} total steps in {dt:.2}s, {:.2} steps/s",
                total as f64 / dt
            );
            Tensor::cat(&all_audio_tokens, candle::D::Minus1)?
        };
        let (_one, _codebooks, total_steps) = all_audio_tokens.dims3()?;
        let mut all_pcm_chunks = vec![];
        let chunk_by = 25;
        let mut mimi = self.audio_tokenizer.clone();
        for start_step in (0..total_steps).step_by(chunk_by) {
            let chunk_steps = usize::min(chunk_by, total_steps - start_step);
            let pcm = mimi.decode_step(
                &all_audio_tokens.narrow(2, start_step, chunk_steps)?.into(),
                &().into(),
            )?;
            if let Some(pcm) = pcm.as_option() {
                all_pcm_chunks.push(pcm.clone())
            }
        }
        // Close the log stream so that log_rx.save does not block.
        std::mem::drop(log_tx);
        if let Err(err) = log_rx.save(&query, &self.log_dir, &self.instance_name) {
            tracing::error!(?err, "cannot save logs")
        };

        let pcm = Tensor::cat(&all_pcm_chunks, 2)?;
        let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
        let mut wav = vec![];
        moshi::wav::write_pcm_as_wav(&mut wav, &pcm, 24_000)?;
        Ok((wav, transcript))
    }
}

pub fn speaker_pcm<P: AsRef<std::path::Path>>(
    mimi_sample_rate: f64,
    speaker_cond_start_s: f64,
    speaker_cond_duration_s: f64,
    speaker: P,
    dev: &Device,
) -> Result<Tensor> {
    let (pcm, sample_rate) = kaudio::pcm_decode(speaker)?;
    let pcm = if sample_rate != mimi_sample_rate as u32 {
        kaudio::resample(&pcm, sample_rate as usize, mimi_sample_rate as usize)?
    } else {
        pcm
    };
    let start_pos = (speaker_cond_start_s * mimi_sample_rate) as usize;
    let sample_len = (speaker_cond_duration_s * mimi_sample_rate) as usize;
    let pcm = &pcm[start_pos..start_pos + sample_len];
    let pcm = Tensor::new(pcm, dev)?.reshape((1, 1, ()))?;
    Ok(pcm)
}
