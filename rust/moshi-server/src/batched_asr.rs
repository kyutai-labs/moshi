// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::asr::{InMsg, OutMsg};
use crate::metrics::asr as metrics;
use crate::AsrStreamingQuery as Query;
use anyhow::{Context, Result};
use axum::extract::ws;
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::task;
use tokio::time::{timeout, Duration};

const FRAME_SIZE: usize = 1920;
const SEND_PING_EVERY: Duration = Duration::from_secs(10);

#[derive(Debug, PartialEq, Eq, Clone)]
struct Marker {
    channel_id: ChannelId,
    batch_idx: usize,
    step_idx: usize,
    marker_id: i64,
}

impl std::cmp::PartialOrd for Marker {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Marker {
    // We use reverse ordering as this will be embedded in a max heap.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.step_idx.cmp(&other.step_idx).reverse()
    }
}

type InSend = std::sync::mpsc::Sender<InMsg>;
type InRecv = std::sync::mpsc::Receiver<InMsg>;
type OutSend = tokio::sync::mpsc::UnboundedSender<OutMsg>;
type OutRecv = tokio::sync::mpsc::UnboundedReceiver<OutMsg>;

/// Unique identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChannelId(usize);

impl ChannelId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct Channel {
    id: ChannelId,
    in_rx: InRecv,
    out_tx: OutSend,
    data: VecDeque<f32>,
    decoder: kaudio::ogg_opus::Decoder,
    steps: usize,
}

impl Channel {
    fn new(in_rx: InRecv, out_tx: OutSend) -> Result<Self> {
        metrics::OPEN_CHANNELS.inc();
        let decoder = kaudio::ogg_opus::Decoder::new(24000, FRAME_SIZE)?;
        Ok(Self { id: ChannelId::new(), in_rx, out_tx, data: VecDeque::new(), decoder, steps: 0 })
    }

    fn extend_data(&mut self, mut pcm: Vec<f32>) -> Option<Vec<f32>> {
        if self.data.is_empty() && pcm.len() >= FRAME_SIZE {
            self.data.extend(&pcm[FRAME_SIZE..]);
            pcm.truncate(FRAME_SIZE);
            Some(pcm)
        } else {
            self.data.extend(&pcm);
            if self.data.len() >= FRAME_SIZE {
                Some(self.data.drain(..FRAME_SIZE).collect())
            } else {
                None
            }
        }
    }

    fn send(&self, msg: OutMsg, ref_channel_id: Option<ChannelId>) -> Result<()> {
        // If the channel id has changed compared to the reference. Return Ok(())
        // so as not to disconnect the new user.
        if Some(self.id) != ref_channel_id {
            return Ok(());
        }
        self.out_tx.send(msg)?;
        Ok(())
    }
}

impl Drop for Channel {
    fn drop(&mut self) {
        metrics::OPEN_CHANNELS.dec();
        metrics::CONNECTION_NUM_STEPS.observe(self.steps as f64);
    }
}

struct Logger {
    base_path: std::path::PathBuf,
    log_tx: std::sync::mpsc::Sender<(Tensor, Tensor)>,
    log_rx: std::sync::mpsc::Receiver<(Tensor, Tensor)>,
    log_frequency_s: f64,
}

impl Logger {
    fn new<P: AsRef<std::path::Path>>(
        instance_name: &str,
        log_dir: P,
        log_frequency_s: f64,
    ) -> Result<Self> {
        let since_epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
        let (secs, us) = (since_epoch.as_secs(), since_epoch.subsec_micros());
        let base_path = log_dir.as_ref().join(format!("{}-asr-{secs}-{us}", instance_name));
        let (log_tx, log_rx) = std::sync::mpsc::channel::<(Tensor, Tensor)>();
        Ok(Self { base_path, log_tx, log_rx, log_frequency_s })
    }

    fn log_loop(self) {
        tracing::info!(?self.base_path, "starting log loop");
        task::spawn_blocking(move || {
            let mut cnt = 0usize;
            loop {
                std::thread::sleep(std::time::Duration::from_secs_f64(self.log_frequency_s));
                let tokens: Vec<_> = self.log_rx.try_iter().collect();
                if tokens.is_empty() {
                    tracing::info!("no tokens to log");
                    continue;
                }
                let st_filename = self.base_path.with_extension(format!("{cnt}.safetensors"));
                tracing::info!(?st_filename, "writing logs");
                let (text_tokens, audio_tokens): (Vec<_>, Vec<_>) = tokens.into_iter().unzip();
                let write = || {
                    let text_tokens = Tensor::cat(&text_tokens, candle::D::Minus1)?;
                    let audio_tokens = Tensor::cat(&audio_tokens, candle::D::Minus1)?;
                    let st_content = std::collections::HashMap::from([
                        ("text", text_tokens),
                        ("audio", audio_tokens),
                    ]);
                    candle::safetensors::save(&st_content, st_filename)?;
                    Ok::<_, anyhow::Error>(())
                };
                if let Err(err) = write() {
                    tracing::error!(?err, "failed to write safetensors");
                }
                cnt += 1;
            }
        });
    }
}

struct BatchedAsrInner {
    channels: Channels,
    asr_delay_in_tokens: usize,
    temperature: f64,
    lm: moshi::lm::LmModel,
    audio_tokenizer: moshi::mimi::Mimi,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
}

fn warmup(
    state: &mut moshi::asr::State,
    conditions: Option<&moshi::conditioner::Condition>,
) -> Result<()> {
    let dev = state.device().clone();
    let pcm = vec![0f32; FRAME_SIZE * state.batch_size()];
    let pcm = Tensor::from_vec(pcm, (state.batch_size(), 1, FRAME_SIZE), &dev)?;
    let mask = moshi::StreamMask::new(vec![true; state.batch_size()], &dev)?;
    for _ in 0..2 {
        let _asr_msgs = state.step_pcm(pcm.clone(), conditions, &mask, |_, _, _| ())?;
    }
    dev.synchronize()?;
    Ok(())
}

impl BatchedAsrInner {
    fn start_model_loop(
        self,
        conditioning_delay: Option<f32>,
        conditioning_learnt_padding: bool,
        batch_size: usize,
        logger: Option<&Logger>,
    ) -> Result<()> {
        let conditions = match self.lm.condition_provider() {
            None => None,
            Some(cp) => match (conditioning_delay, conditioning_learnt_padding) {
                (Some(delay), false) => {
                    let conditions = cp.condition_cont("delay", -delay)?;
                    tracing::info!(?conditions, "generated conditions");
                    Some(conditions)
                }
                (None, true) => {
                    let conditions = cp.learnt_padding("delay")?;
                    tracing::info!(?conditions, "generated conditions");
                    Some(conditions)
                }
                (Some(_), true) => anyhow::bail!(
                    "conditioning_delay/conditioning_learnt_padding cannot be both set"
                ),
                (None, false) => {
                    anyhow::bail!("conditioning_delay/conditioning_learnt_padding is required")
                }
            },
        };
        let mut state = moshi::asr::State::new(
            batch_size,
            self.asr_delay_in_tokens,
            self.temperature,
            self.audio_tokenizer.clone(),
            self.lm.clone(),
        )?;
        let log_tx = logger.map(|v| v.log_tx.clone());
        let dev = state.device().clone();
        let model_loop: task::JoinHandle<Result<()>> = task::spawn_blocking(move || {
            tracing::info!("warming-up the asr");
            warmup(&mut state, conditions.as_ref())?;
            tracing::info!("starting asr loop {batch_size}");
            // Store the markers in a double ended queue
            let mut markers = BinaryHeap::new();
            // This loop runs in real-time.
            let mut step_idx = 0;
            loop {
                let (batch_pcm, mask, ref_channel_ids) =
                    self.pre_process(&mut state, step_idx, &mut markers);
                let with_data = mask.iter().filter(|v| **v).count();
                if with_data > 0 {
                    let mask = moshi::StreamMask::new(mask, &dev)?;
                    let pcm =
                        Tensor::new(batch_pcm.as_slice(), &dev)?.reshape((batch_size, 1, ()))?;
                    let start_time = std::time::Instant::now();
                    let asr_msgs = state.step_pcm(
                        pcm,
                        conditions.as_ref(),
                        &mask,
                        |_, text_tokens, audio_tokens| {
                            let res = || {
                                if let Some(log_tx) = log_tx.as_ref() {
                                    let text_tokens = text_tokens.to_device(&Device::Cpu)?;
                                    let audio_tokens: Vec<Tensor> = audio_tokens
                                        .iter()
                                        .map(|t| t.to_device(&Device::Cpu))
                                        .collect::<candle::Result<Vec<_>>>()?;
                                    let audio_tokens = Tensor::stack(&audio_tokens, 1)?;
                                    if let Err(err) = log_tx.send((text_tokens, audio_tokens)) {
                                        tracing::error!(?err, "failed to send log");
                                    };
                                }
                                Ok::<_, anyhow::Error>(())
                            };
                            if let Err(err) = res() {
                                tracing::error!(?err, "failed to send log");
                            }
                        },
                    )?;
                    let elapsed = start_time.elapsed().as_secs_f64();
                    metrics::MODEL_STEP_DURATION.observe(elapsed);
                    tracing::info!(step_idx, with_data, "{:.2}ms", elapsed * 1000.);
                    step_idx += 1;
                    self.post_process(asr_msgs, step_idx, &mut markers, &mask, &ref_channel_ids)?;
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
            }
        });
        task::spawn(async {
            match model_loop.await {
                Err(err) => tracing::error!(?err, "model loop join err"),
                Ok(Err(err)) => tracing::error!(?err, "model loop err"),
                Ok(Ok(())) => tracing::info!("model loop exited"),
            }
        });
        Ok(())
    }

    fn pre_process(
        &self,
        state: &mut moshi::asr::State,
        step_idx: usize,
        markers: &mut BinaryHeap<Marker>,
    ) -> (Vec<f32>, Vec<bool>, Vec<Option<ChannelId>>) {
        use rayon::prelude::*;
        enum Todo {
            Reset(usize),
            Marker(Marker),
        }

        let mut mask = vec![false; state.batch_size()];
        let mut channels = self.channels.lock().unwrap();
        let mut batch_pcm = vec![0f32; FRAME_SIZE * channels.len()];
        let channel_ids = channels.iter().map(|c| c.as_ref().map(|c| c.id)).collect::<Vec<_>>();
        let todo = batch_pcm
            .par_chunks_mut(FRAME_SIZE)
            .zip(channels.par_iter_mut())
            .zip(mask.par_iter_mut())
            .enumerate()
            .flat_map(|(bid, ((out_pcm, channel), mask))| -> Option<Todo> {
                let c = channel.as_mut()?;
                if c.out_tx.is_closed() {
                    *channel = None;
                    None
                } else {
                    use std::sync::mpsc::TryRecvError;
                    match c.in_rx.try_recv() {
                        Ok(InMsg::Init) => {
                            if c.out_tx.send(OutMsg::Ready).is_err() {
                                *channel = None;
                            }
                            Some(Todo::Reset(bid))
                        }
                        Ok(InMsg::Marker { id }) => {
                            tracing::info!(bid, id, "received marker");
                            // The marker only gets sent back once all the current data has been
                            // processed and the asr delay has passed.
                            let current_data = c.data.len() / FRAME_SIZE;
                            let step_idx = step_idx + state.asr_delay_in_tokens() + current_data;
                            let marker = Marker {
                                channel_id: c.id,
                                batch_idx: bid,
                                step_idx,
                                marker_id: id,
                            };
                            Some(Todo::Marker(marker))
                        }
                        Ok(InMsg::OggOpus { data }) => {
                            match c.decoder.decode(&data) {
                                Err(err) => tracing::error!(?err, "oggopus not supported"),
                                Ok(None) => {}
                                Ok(Some(pcm)) => {
                                    out_pcm.copy_from_slice(pcm);
                                    c.steps += 1;
                                    *mask = true;
                                }
                            }
                            None
                        }
                        Ok(InMsg::Audio { pcm }) => {
                            if let Some(bpcm) = c.extend_data(pcm) {
                                out_pcm.copy_from_slice(&bpcm);
                                c.steps += 1;
                                *mask = true;
                            }
                            None
                        }
                        Err(TryRecvError::Empty) => {
                            // Even if we haven't received new data, we process the existing one.
                            if let Some(bpcm) = c.extend_data(vec![]) {
                                out_pcm.copy_from_slice(&bpcm);
                                c.steps += 1;
                                *mask = true;
                            }
                            None
                        }
                        Err(TryRecvError::Disconnected) => {
                            *channel = None;
                            None
                        }
                    }
                }
            })
            .collect::<Vec<_>>();
        todo.into_iter().for_each(|t| match t {
            Todo::Reset(bid) => {
                if let Err(err) = state.reset_batch_idx(bid) {
                    tracing::error!(?err, bid, "failed to reset batch");
                }
            }
            Todo::Marker(m) => markers.push(m),
        });
        (batch_pcm, mask, channel_ids)
    }

    fn post_process(
        &self,
        asr_msgs: Vec<moshi::asr::AsrMsg>,
        step_idx: usize,
        markers: &mut BinaryHeap<Marker>,
        mask: &moshi::StreamMask,
        ref_channel_ids: &[Option<ChannelId>],
    ) -> Result<()> {
        let mut channels = self.channels.lock().unwrap();
        for asr_msg in asr_msgs.into_iter() {
            match asr_msg {
                moshi::asr::AsrMsg::Word { tokens, start_time, batch_idx } => {
                    let msg = OutMsg::Word {
                        text: self.text_tokenizer.decode_piece_ids(&tokens)?,
                        start_time,
                    };
                    if let Some(c) = channels[batch_idx].as_ref() {
                        if c.send(msg, ref_channel_ids[batch_idx]).is_err() {
                            channels[batch_idx] = None;
                        }
                    }
                }
                moshi::asr::AsrMsg::EndWord { stop_time, batch_idx } => {
                    let msg = OutMsg::EndWord { stop_time };
                    if let Some(c) = channels[batch_idx].as_ref() {
                        if c.send(msg, ref_channel_ids[batch_idx]).is_err() {
                            channels[batch_idx] = None;
                        }
                    }
                }
                moshi::asr::AsrMsg::Step { step_idx, prs } => {
                    for (batch_idx, c) in channels.iter_mut().enumerate() {
                        if !mask.is_active(batch_idx) {
                            continue;
                        }
                        if let Some(ch) = c.as_mut() {
                            let prs = prs.iter().map(|p| p[batch_idx]).collect();
                            let msg = OutMsg::Step { step_idx, prs, buffered_pcm: ch.data.len() };
                            if ch.send(msg, ref_channel_ids[batch_idx]).is_err() {
                                *c = None;
                            }
                        }
                    }
                }
            }
        }
        while let Some(m) = markers.peek() {
            if m.step_idx <= step_idx {
                if let Some(c) = channels[m.batch_idx].as_ref() {
                    if c.send(OutMsg::Marker { id: m.marker_id }, Some(m.channel_id)).is_err() {
                        channels[m.batch_idx] = None;
                    }
                }
                markers.pop();
            } else {
                break;
            }
        }
        Ok(())
    }
}

type Channels = Arc<Mutex<Vec<Option<Channel>>>>;

pub struct BatchedAsr {
    channels: Channels,
    config: crate::AsrConfig,
    batch_size: usize,
}

impl BatchedAsr {
    pub fn new(
        batch_size: usize,
        asr: &crate::AsrConfig,
        config: &crate::Config,
        dev: &Device,
    ) -> Result<Self> {
        let dtype = dev.bf16_default_to_f32();
        let vb_lm =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&asr.lm_model_file], dtype, dev)? };
        let lm = moshi::lm::LmModel::batched(
            batch_size,
            &asr.model,
            moshi::nn::MaybeQuantizedVarBuilder::Real(vb_lm),
        )?;
        let audio_tokenizer = {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[&asr.audio_tokenizer_file],
                    DType::F32,
                    dev,
                )?
            };
            let mut cfg = moshi::mimi::Config::v0_1(Some(asr.model.audio_codebooks));
            // The mimi transformer runs at 25Hz.
            cfg.transformer.max_seq_len = asr.model.transformer.max_seq_len * 2;
            moshi::mimi::Mimi::batched(batch_size, cfg, vb)?
        };
        let text_tokenizer = sentencepiece::SentencePieceProcessor::open(&asr.text_tokenizer_file)
            .with_context(|| asr.text_tokenizer_file.clone())?;
        let channels = (0..batch_size).map(|_| None).collect::<Vec<_>>();
        let channels = Arc::new(Mutex::new(channels));
        let asr_delay_in_tokens =
            asr.conditioning_delay.map_or(asr.asr_delay_in_tokens, |v| (v * 12.5) as usize + 1);
        let batched_asr = BatchedAsrInner {
            asr_delay_in_tokens,
            temperature: asr.temperature.unwrap_or(0.0),
            lm,
            audio_tokenizer,
            text_tokenizer: text_tokenizer.into(),
            channels: channels.clone(),
        };
        let logger = match asr.log_frequency_s {
            Some(s) => Some(Logger::new(&config.instance_name, &config.log_dir, s)?),
            None => None,
        };
        batched_asr.start_model_loop(
            asr.conditioning_delay,
            asr.conditioning_learnt_padding,
            batch_size,
            logger.as_ref(),
        )?;
        if let Some(logger) = logger {
            logger.log_loop()
        }
        Ok(Self { channels, config: asr.clone(), batch_size })
    }

    fn channels(&self) -> Result<(usize, InSend, OutRecv)> {
        let mut channels = self.channels.lock().unwrap();
        // Linear scan to find an available channel. This is fairly inefficient, instead we should
        // probably have a queue of available slots.
        for (batch_idx, channel) in channels.iter_mut().enumerate() {
            if channel.is_none() {
                let (in_tx, in_rx) = std::sync::mpsc::channel::<InMsg>();
                let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<OutMsg>();
                let c = Channel::new(in_rx, out_tx)?;
                *channel = Some(c);
                return Ok((batch_idx, in_tx, out_rx));
            }
        }
        anyhow::bail!("no free channels");
    }

    pub async fn handle_socket(&self, socket: ws::WebSocket, query: Query) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};
        use serde::Serialize;

        tracing::info!(?query, "batched-asr query");
        metrics::CONNECT.inc();

        let (mut sender, receiver) = socket.split();
        let (batch_idx, in_tx, mut out_rx) = match self.channels() {
            Ok(v) => v,
            Err(err) => {
                tracing::error!(?err, "no free channels");
                let mut msg = vec![];
                OutMsg::Error { message: "no free channels".into() }.serialize(
                    &mut rmp_serde::Serializer::new(&mut msg)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                sender.send(ws::Message::binary(msg)).await?;
                sender.close().await?;
                return Err(err);
            }
        };
        tracing::info!(batch_idx, "batched-asr channel");
        in_tx.send(InMsg::Init)?;

        let recv_loop = task::spawn(async move {
            let mut receiver = receiver;
            // There are two timeouts here:
            // - The short timeout handles the case where the client does not answer the regular pings.
            // - The long timeout handles the case where the client does not send valid data for a
            // long time.
            let mut last_message_received = std::time::Instant::now();
            let short_timeout_duration = SEND_PING_EVERY * 2;
            let long_timeout_duration = std::time::Duration::from_secs(120);
            loop {
                use ws::Message;
                let msg = match timeout(short_timeout_duration, receiver.next()).await {
                    Ok(Some(msg)) => msg,
                    Ok(None) => break,
                    Err(_) => {
                        tracing::info!(?batch_idx, "recv loop short timeout");
                        break;
                    }
                };
                if last_message_received.elapsed() > long_timeout_duration {
                    tracing::info!(?batch_idx, "recv loop long timeout");
                    break;
                }
                let msg = match msg? {
                    Message::Binary(x) => x,
                    // ping messages are automatically answered by tokio-tungstenite as long as
                    // the connection is read from.
                    Message::Ping(_) | Message::Pong(_) | Message::Text(_) => continue,
                    Message::Close(_) => break,
                };
                last_message_received = std::time::Instant::now();
                let msg: InMsg = rmp_serde::from_slice(&msg)?;
                in_tx.send(msg)?;
            }
            Ok::<_, anyhow::Error>(())
        });
        let send_loop = task::spawn(async move {
            let mut sender = sender;
            loop {
                // The recv method is cancel-safe so can be wrapped in a timeout.
                let msg = timeout(SEND_PING_EVERY, out_rx.recv()).await;
                let msg = match msg {
                    Ok(None) => break,
                    Err(_) => ws::Message::Ping(vec![].into()),
                    Ok(Some(msg)) => {
                        let mut buf = vec![];
                        msg.serialize(
                            &mut rmp_serde::Serializer::new(&mut buf)
                                .with_human_readable()
                                .with_struct_map(),
                        )?;
                        ws::Message::Binary(buf.into())
                    }
                };
                sender.send(msg).await?;
            }
            Ok::<(), anyhow::Error>(())
        });

        // Keep track of the outputs of the different threads.
        task::spawn(async {
            match send_loop.await {
                Err(err) => tracing::error!(?err, "send loop join err"),
                Ok(Err(err)) => tracing::error!(?err, "send loop err"),
                Ok(Ok(())) => tracing::info!("send loop exited"),
            }
        });
        task::spawn(async {
            match recv_loop.await {
                Err(err) => tracing::error!(?err, "recv loop join err"),
                Ok(Err(err)) => tracing::error!(?err, "recv loop err"),
                Ok(Ok(())) => tracing::info!("recv loop exited"),
            }
        });

        Ok(())
    }

    pub fn config(&self) -> &crate::AsrConfig {
        &self.config
    }

    pub fn total_slots(&self) -> usize {
        self.batch_size
    }

    pub fn used_slots(&self) -> usize {
        self.channels.lock().unwrap().iter().filter(|v| v.is_some()).count()
    }
}
