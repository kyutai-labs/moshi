// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
use crate::protocol::MsgType;
use anyhow::{Context, Result};
use axum::extract::ws;
use candle::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use kaudio::ogg_opus;

struct TextDecoder {
    gen_config: moshi::lm_generate_multistream::Config,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
}

impl TextDecoder {
    fn text(&self, prev_text_token: u32, text_token: u32) -> Option<String> {
        let config = &self.gen_config;
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

pub struct Lm {
    dev: Device,
    gen_config: moshi::lm_generate_multistream::Config,
    lm: moshi::lm::LmModel,
    audio_tokenizer: moshi::mimi::Mimi,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
    instance_name: String,
    log_dir: std::path::PathBuf,
}

enum WsEvent {
    Text(String),
    Pcm(Vec<f32>),
}

enum LogEvent {
    TextToken(u32),
    AudioTokens(Vec<u32>),
}

impl Lm {
    pub fn new(lm: &crate::LmConfig, config: &crate::Config, dev: &Device) -> Result<Self> {
        let dtype = dev.bf16_default_to_f32();
        let model_config = &lm.model;
        let gen_config = lm.gen.clone();
        let audio_tokenizer = moshi::mimi::load(&lm.audio_tokenizer_file, Some(8), dev)?;
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&lm.text_tokenizer_file)
            .with_context(|| lm.text_tokenizer_file.clone())?;
        let vb_lm =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&lm.lm_model_file], dtype, dev)? };
        let lm = moshi::lm::LmModel::new(
            model_config,
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        Ok(Self {
            audio_tokenizer,
            lm,
            gen_config,
            dev: dev.clone(),
            log_dir: config.log_dir.clone().into(),
            instance_name: config.instance_name.clone(),
            text_tokenizer: text_tokenizer.into(),
        })
    }

    pub async fn handle_socket(&self, socket: ws::WebSocket) -> Result<()> {
        use futures_util::StreamExt;

        tracing::info!("connected");
        let (opus_in_tx, mut opus_in_rx) = tokio::sync::mpsc::unbounded_channel();
        let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel();
        let (event_tx, event_rx) = std::sync::mpsc::channel();
        let (mut ws_sender, mut ws_receiver) = socket.split();
        let ws_recv_handle = tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                let (msg_type, payload) = match msg? {
                    ws::Message::Binary(b) => {
                        if b.is_empty() {
                            continue;
                        }
                        let msg_type = MsgType::from_u8(b[0])?;
                        let payload = b[1..].to_vec();
                        (msg_type, payload)
                    }
                    _ => continue,
                };
                match msg_type {
                    MsgType::Audio => {
                        opus_in_tx.send(payload)?;
                    }
                    t => {
                        tracing::warn!("unexpected msg type {t:?}");
                        continue;
                    }
                }
            }
            Ok::<_, anyhow::Error>(())
        });
        let dev = self.dev.clone();
        let mut audio_tokenizer = self.audio_tokenizer.clone();
        audio_tokenizer.reset_state();
        let text_lp = LogitsProcessor::from_sampling(
            299792458,
            candle_transformers::generation::Sampling::TopK { k: 25, temperature: 0.8 },
        );
        let audio_lp = LogitsProcessor::from_sampling(
            299792458,
            candle_transformers::generation::Sampling::TopK { k: 250, temperature: 0.8 },
        );
        let conditions = match self.lm.condition_provider() {
            None => None,
            Some(cp) => {
                let conditions = cp.condition_lut("description", "very_good")?;
                tracing::info!(?conditions, "generated conditions");
                Some(conditions)
            }
        };

        let mut state = moshi::lm_generate_multistream::State::new(
            self.lm.clone(),
            /* max_steps = */ 4096,
            audio_lp,
            text_lp,
            None,
            None,
            None,
            self.gen_config.clone(),
        );
        let text_decoder = TextDecoder {
            gen_config: self.gen_config.clone(),
            text_tokenizer: self.text_tokenizer.clone(),
        };
        let mut decoder = ogg_opus::Decoder::new(24000, 1920)?;
        let pcm_recv_handle = tokio::spawn(async move {
            let mut prev_text_token = state.config().text_start_token;
            tracing::info!("starting pcm recv loop");
            while let Some(opus) = opus_in_rx.recv().await {
                if let Some(pcm) = decoder.decode(&opus)? {
                    let pcm = Tensor::new(pcm, &dev)?.reshape((1, 1, ()))?;
                    let audio_tokens = audio_tokenizer.encode_step(&pcm.into(), &().into())?;
                    let audio_tokens = match audio_tokens.as_option() {
                        None => continue,
                        Some(audio_tokens) => audio_tokens,
                    };
                    let (_one, _codebooks, steps) = audio_tokens.dims3()?;

                    for step in 0..steps {
                        let codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
                        let text_token = state.step_(
                            Some(prev_text_token),
                            &codes,
                            None,
                            None,
                            conditions.as_ref(),
                        )?;

                        if let Some(text) = text_decoder.text(prev_text_token, text_token) {
                            out_tx.send(WsEvent::Text(text))?
                        }
                        event_tx.send(LogEvent::TextToken(text_token))?;
                        tracing::info!(text_token, "sampled text token");
                        if let Some(audio_tokens) = state.last_audio_tokens() {
                            let audio_tokens_t = {
                                let cb = state.config().generated_audio_codebooks;
                                Tensor::from_slice(&audio_tokens[..cb], (1, cb, 1), &dev)?
                            };
                            event_tx.send(LogEvent::AudioTokens(audio_tokens))?;
                            let pcm =
                                audio_tokenizer.decode_step(&audio_tokens_t.into(), &().into())?;
                            if let Some(pcm) = pcm.as_option() {
                                let pcm = pcm.i((0, 0))?.to_vec1::<f32>()?;
                                out_tx.send(WsEvent::Pcm(pcm))?;
                            }
                        }
                        prev_text_token = text_token
                    }
                }
            }
            Ok::<_, anyhow::Error>(())
        });
        let send_handle = tokio::spawn(async move {
            use futures_util::SinkExt;

            let mut encoder = ogg_opus::Encoder::new(24000)?;
            let mut handshake = vec![MsgType::Handshake.to_u8()];
            handshake.resize(9, 0u8);
            if let Err(err) = ws_sender.send(ws::Message::binary(handshake)).await {
                tracing::error!("error sending header {err:?}");
                return Ok(());
            }
            {
                let msg: Vec<u8> = [&[MsgType::Audio.to_u8()], encoder.header_data()].concat();
                let msg = ws::Message::Binary(msg.into());
                ws_sender.send(msg).await?;
            }
            while let Some(evt) = out_rx.recv().await {
                let msg: Vec<u8> = match evt {
                    WsEvent::Pcm(pcm) => {
                        let ogg = encoder.encode_page(&pcm)?;
                        [&[MsgType::Audio.to_u8()], ogg.as_slice()].concat()
                    }
                    WsEvent::Text(text) => [&[MsgType::Text.to_u8()], text.as_bytes()].concat(),
                };
                let msg = ws::Message::Binary(msg.into());
                ws_sender.send(msg).await?
            }
            Ok::<_, anyhow::Error>(())
        });
        let sleep = tokio::time::sleep(std::time::Duration::from_secs(360));
        tokio::pin!(sleep);
        // select should ensure that all the threads get aborted on timeout.
        tokio::select! {
            _ = &mut sleep => {
                tracing::error!("reached timeout");
            }
            r = pcm_recv_handle => {
                tracing::error!(?r, "pcm recv loop ended")
            }
            r = ws_recv_handle => {
                tracing::error!(?r, "ws recv loop ended")
            }
            r = send_handle => {
                tracing::error!(?r, "ws send loop ended")
            }
        };
        let events: Vec<_> = event_rx.try_iter().collect();
        self.save_logs((), events)?;
        Ok(())
    }

    fn save_logs(&self, query: (), events: Vec<LogEvent>) -> Result<()> {
        let cpu = &Device::Cpu;
        let since_epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
        let (secs, us) = (since_epoch.as_secs(), since_epoch.subsec_micros());
        let base_path = self.log_dir.join(format!("{}-lm-{secs}-{us}", self.instance_name));
        let json_filename = base_path.with_extension("json");
        let json_content = serde_json::to_string_pretty(&query)?;
        std::fs::write(json_filename, json_content)?;
        let st_filename = base_path.with_extension("safetensors");
        let text_tokens: Vec<i64> = events
            .iter()
            .filter_map(|v| match v {
                LogEvent::TextToken(v) => Some(*v as i64),
                LogEvent::AudioTokens(_) => None,
            })
            .collect();
        let text_len = text_tokens.len();
        let text_tokens =
            Tensor::from_vec(text_tokens, text_len, cpu)?.to_dtype(candle::DType::I64)?;
        let audio_tokens: Vec<_> = events
            .iter()
            .filter_map(|v| match v {
                LogEvent::TextToken(_) => None,
                LogEvent::AudioTokens(a) => {
                    let a = a.iter().map(|v| *v as i64).collect::<Vec<_>>();
                    Some(Tensor::from_slice(&a, (1, a.len(), 1), cpu))
                }
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let audio_tokens = Tensor::cat(&audio_tokens, 2)?;
        let st_content =
            std::collections::HashMap::from([("text", text_tokens), ("audio", audio_tokens)]);
        candle::safetensors::save(&st_content, st_filename)?;
        Ok(())
    }
}
