// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::asr::{InMsg, OutMsg};
use crate::metrics::asr as metrics;
use crate::py_module::{init, toml_to_py, VerbosePyErr};
use crate::PyAsrStreamingQuery as Query;
use anyhow::{Context, Result};
use axum::extract::ws;
use moshi::asr::AsrMsg;
use pyo3::prelude::*;
use pyo3_ffi::c_str;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::task;
use tokio::time::{timeout, Duration};

const FRAME_SIZE: usize = 1920;
const SEND_PING_EVERY: Duration = Duration::from_secs(10);
const POST_RETRY_DELAY: Duration = Duration::from_millis(100);
const POST_MAX_RETRIES: usize = 1000;
const MASK_ACTIVE: u8 = 1 << 0;
const MASK_MARKER_RECEIVED: u8 = 1 << 1;
const MASK_END_OF_STREAM: u8 = 1 << 2;

const NODATA: i32 = 0;
const ACTIVE: i32 = -1;
const RESET: i32 = -2;

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

type InSend = std::sync::mpsc::Sender<InMsg>;
type InRecv = std::sync::mpsc::Receiver<InMsg>;
type OutSend = tokio::sync::mpsc::UnboundedSender<OutMsg>;
type OutRecv = tokio::sync::mpsc::UnboundedReceiver<OutMsg>;

struct Channel {
    id: ChannelId,
    in_rx: InRecv,
    out_tx: OutSend,
    data: VecDeque<f32>,
    steps: usize,
    decoder: kaudio::ogg_opus::Decoder,
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

type Channels = Arc<Mutex<Vec<Option<Channel>>>>;

struct Inner {
    channels: Channels,
    app: PyObject,
    text_tokenizer: Arc<sentencepiece::SentencePieceProcessor>,
    asr_delay_in_tokens: usize,
}

impl Inner {
    fn pre_process(
        &self,
        step_idx: usize,
        batch_size: usize,
        batch_pcm: &mut [f32],
        markers: &mut BinaryHeap<Marker>,
    ) -> Result<(Vec<i32>, Vec<Option<ChannelId>>)> {
        use rayon::prelude::*;
        let mut channels = self.channels.lock().unwrap();
        let mut updates = vec![NODATA; batch_size];
        let channel_ids = channels.iter().map(|c| c.as_ref().map(|c| c.id)).collect::<Vec<_>>();
        let todo: Vec<Marker> = channels
            .par_iter_mut()
            .zip(batch_pcm.par_chunks_mut(FRAME_SIZE))
            .zip(updates.par_iter_mut())
            .enumerate()
            .flat_map(|(batch_idx, ((channel, pcm_out), update))| {
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
                            *update = RESET;
                            None
                        }
                        Ok(InMsg::Marker { id }) => {
                            tracing::info!(batch_idx, id, "received marker");
                            // The marker only gets sent back once all the current data has been
                            // processed and the asr delay has passed.
                            let current_data = c.data.len() / FRAME_SIZE;
                            let step_idx = step_idx + self.asr_delay_in_tokens + current_data;

                            let marker =
                                Marker { channel_id: c.id, batch_idx, step_idx, marker_id: id };
                            *update = current_data as i32;
                            Some(marker)
                        }
                        Ok(InMsg::OggOpus { data }) => {
                            let mut bpcm = vec![];
                            match c.decoder.decode(&data) {
                                Err(err) => tracing::error!(?err, "oggopus not supported"),
                                Ok(None) => {}
                                Ok(Some(pcm)) => {
                                    bpcm = pcm.to_vec();
                                }
                            }
                            if let Some(bpcm) = c.extend_data(bpcm) {
                                pcm_out.copy_from_slice(&bpcm);
                                *update = ACTIVE;
                            }
                            None
                        }
                        Ok(InMsg::Audio { pcm }) => {
                            if let Some(bpcm) = c.extend_data(pcm) {
                                pcm_out.copy_from_slice(&bpcm);
                                *update = ACTIVE;
                            }
                            None
                        }
                        Err(TryRecvError::Empty) => {
                            // Even if we haven't received new data, we process the existing one.
                            if let Some(bpcm) = c.extend_data(vec![]) {
                                pcm_out.copy_from_slice(&bpcm);
                                *update = ACTIVE;
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
        todo.into_iter().for_each(|marker| markers.push(marker));
        Ok((updates, channel_ids))
    }

    fn start_model_loop(self, batch_size: usize) -> Result<()> {
        use numpy::{PyArrayMethods, ToPyArray};
        use rayon::prelude::*;
        use std::ops::DerefMut;
        let model_loop: task::JoinHandle<Result<()>> = task::spawn_blocking(move || {
            tracing::info!("starting-up the py asr model loop");
            struct MsgsOut {
                msgs: Vec<AsrMsg>,
            }
            let mut markers = BinaryHeap::new();
            let mask = numpy::ndarray::Array1::<u8>::zeros([batch_size]);
            let mask = Python::with_gil(|py| mask.to_pyarray(py).unbind());
            let tokens = numpy::ndarray::Array1::<u32>::zeros([batch_size]);
            let tokens = Python::with_gil(|py| tokens.to_pyarray(py).unbind());
            let extra_heads = numpy::ndarray::Array2::<f32>::zeros((batch_size, 4));
            let extra_heads = Python::with_gil(|py| extra_heads.to_pyarray(py).unbind());
            let batch_pcm_py = numpy::ndarray::Array1::<f32>::zeros(batch_size * FRAME_SIZE);
            let batch_pcm_py = Python::with_gil(|py| batch_pcm_py.to_pyarray(py).unbind());
            let mut current_word: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
            let mut words_start_step = vec![0_usize; batch_size];
            let mut channel_ids = vec![None; batch_size];
            let mut updates: Vec<i32> = vec![NODATA; batch_size];
            let mut with_data: usize = 0;
            let mut step_idx = 0;
            loop {
                // We store the channel ids here to check that they have not changed when sending
                // the data back to the user.
                let start_time = std::time::Instant::now();
                let mut asr_msgs: Vec<AsrMsg> = vec![];
                Python::with_gil(|py| -> Result<()> {
                    let mut batch_pcm = batch_pcm_py.bind(py).readwrite();
                    let batch_pcm = batch_pcm
                        .as_slice_mut()
                        .context("pcm is not contiguous or not writable")?;
                    (updates, channel_ids) =
                        self.pre_process(step_idx, batch_size, batch_pcm, &mut markers)?;
                    self.app
                        .call_method1(
                            py,
                            "step",
                            (&batch_pcm_py, &mask, &tokens, &extra_heads, updates.clone()),
                        )
                        .map_err(VerbosePyErr::from)?;
                    let mask = mask.bind(py).readonly();
                    let tokens_data = tokens.bind(py).readonly();
                    let mask = mask.as_slice().context("mask is not contiguous")?;
                    let tokens_data = tokens_data.as_slice().context("tokens is not contiguous")?;
                    let extra_heads_data = extra_heads.bind(py).readonly();
                    let extra_heads_data = extra_heads_data.as_array();
                    with_data = mask
                        .iter()
                        .filter(|&&x| x == MASK_ACTIVE || x == MASK_MARKER_RECEIVED)
                        .count();
                    if with_data == 0 {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        return Ok(());
                    }
                    let mut channels = self.channels.lock().unwrap();
                    let c = channels.deref_mut();
                    let todo = c
                        .par_iter_mut()
                        .zip(current_word.par_iter_mut())
                        .zip(words_start_step.par_iter_mut())
                        .enumerate()
                        .flat_map(|(batch_idx, ((channel, word), start_step))| -> Option<MsgsOut> {
                            if let Some(c) = channel.as_mut() {
                                let mask = mask[batch_idx];
                                let tokens_data = tokens_data[batch_idx];

                                // The channel has changed so skip the update.
                                if Some(c.id) != channel_ids[batch_idx] {
                                    return None;
                                }
                                if (mask & MASK_ACTIVE) > 0 || (mask & MASK_MARKER_RECEIVED) > 0 {
                                    c.steps += 1;
                                    match tokens_data {
                                        0 | 3 => {
                                            if !word.is_empty() {
                                                let msgs = vec![
                                                    moshi::asr::AsrMsg::Word {
                                                        tokens: std::mem::take(word),
                                                        start_time: *start_step as f64 / 12.5,
                                                        batch_idx,
                                                    },
                                                    moshi::asr::AsrMsg::EndWord {
                                                        stop_time: c.steps as f64 / 12.5,
                                                        batch_idx,
                                                    },
                                                ];
                                                Some(MsgsOut { msgs })
                                            } else {
                                                word.clear();
                                                None
                                            }
                                        }
                                        _ => {
                                            if word.is_empty() {
                                                *start_step = c.steps;
                                            }
                                            word.push(tokens_data);
                                            None
                                        }
                                    }
                                } else if (mask & MASK_END_OF_STREAM) > 0 {
                                    word.clear();
                                    None
                                } else {
                                    None
                                }
                            } else {
                                // The channel has been closed, so we skip the update.
                                None
                            }
                        })
                        .collect::<Vec<_>>();
                    todo.into_iter().for_each(|t| asr_msgs.extend(t.msgs));
                    asr_msgs.push(moshi::asr::AsrMsg::Step {
                        step_idx: (step_idx),
                        prs: extra_heads_data
                            .outer_iter()
                            .map(|row| row.iter().copied().collect())
                            .collect(),
                    });

                    let elapsed = start_time.elapsed().as_secs_f64();
                    metrics::MODEL_STEP_DURATION.observe(elapsed);
                    tracing::info!(step_idx, with_data, "{:.2}ms", elapsed * 1000.);
                    Ok(())
                })?;
                if with_data > 0 {
                    self.post_process(asr_msgs, step_idx, &mut markers, &channel_ids)?;
                    step_idx += 1;
                }
            }
        });
        task::spawn(async {
            match model_loop.await {
                Err(err) => tracing::error!(?err, "model loop join err"),
                Ok(Err(err)) => tracing::error!(?err, "model loop or post-process err"),
                Ok(Ok(())) => tracing::info!("model loop exited"),
            }
        });
        Ok(())
    }
    fn post_process(
        &self,
        asr_msgs: Vec<moshi::asr::AsrMsg>,
        step_idx: usize,
        markers: &mut BinaryHeap<Marker>,
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
                        if let Some(ch) = c.as_mut() {
                            let prs = prs[batch_idx].clone();
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
                channels[m.batch_idx] = None;
                markers.pop();
            } else {
                break;
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct M {
    channels: Channels,
    config: crate::PyAsrConfig,
}

impl M {
    pub fn new(config: crate::PyAsrConfig) -> Result<Self> {
        init()?;
        let batch_size = config.batch_size;
        let asr_delay_in_tokens = config.asr_delay_in_tokens;
        let (script, script_name) = match &config.script {
            None => {
                let script_name = std::ffi::CString::new("batched_asr.py")?;
                let script = std::ffi::CString::new(crate::ASR_PY)?;
                (script, script_name)
            }
            Some(script) => {
                let script_name = std::ffi::CString::new(script.as_bytes())?;
                let script =
                    std::fs::read_to_string(script).with_context(|| format!("{script:?}"))?;
                let script = std::ffi::CString::new(script)?;
                (script, script_name)
            }
        };
        let app = Python::with_gil(|py| -> Result<_> {
            let py_config = pyo3::types::PyDict::new(py);
            if let Some(cfg) = config.py.as_ref() {
                for (key, value) in cfg.iter() {
                    py_config.set_item(key, toml_to_py(py, value)?)?;
                }
            }
            let app =
                PyModule::from_code(py, script.as_c_str(), script_name.as_c_str(), c_str!("foo"))
                    .map_err(VerbosePyErr::from)?
                    .getattr("init")?
                    .call1((batch_size.into_pyobject(py)?, py_config))
                    .map_err(VerbosePyErr::from)?;
            Ok(app.unbind())
        })?;
        let text_tokenizer =
            sentencepiece::SentencePieceProcessor::open(&config.text_tokenizer_file)
                .with_context(|| config.text_tokenizer_file.clone())?;
        let channels = (0..batch_size).map(|_| None).collect::<Vec<_>>();
        let channels = Arc::new(Mutex::new(channels));
        let inner = Inner {
            app,
            channels: channels.clone(),
            text_tokenizer: text_tokenizer.into(),
            asr_delay_in_tokens,
        };

        inner.start_model_loop(batch_size)?;
        Ok(Self { channels, config })
    }
    // Returns None if no channel is available at the moment.
    fn channels(&self) -> Result<Option<(usize, InSend, OutRecv)>> {
        let mut channels = self.channels.lock().unwrap();
        // Linear scan to find an available channel. This is fairly inefficient, instead we should
        // probably have a queue of available slots.
        for (batch_idx, channel) in channels.iter_mut().enumerate() {
            if channel.is_none() {
                let (in_tx, in_rx) = std::sync::mpsc::channel::<InMsg>();
                let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<OutMsg>();
                let c = Channel::new(in_rx, out_tx)?;
                *channel = Some(c);
                return Ok(Some((batch_idx, in_tx, out_rx)));
            }
        }
        Ok(None)
    }

    // TODO: Add a proper batch variant that would enqueue the task so that it can be processed
    // when there is a free channel.
    pub async fn handle_query(&self, query: axum::body::Bytes) -> Result<Vec<OutMsg>> {
        tracing::info!("py-asr handle-query");
        metrics::CONNECT.inc();
        let (batch_idx, in_tx, mut out_rx) = {
            let mut num_tries = 0;
            loop {
                match self.channels() {
                    Ok(Some(x)) => break x,
                    Ok(None) => {
                        num_tries += 1;
                        if num_tries > POST_MAX_RETRIES {
                            tracing::error!("no free channels after 1000 tries");
                            anyhow::bail!("no free channels");
                        }
                        tokio::time::sleep(POST_RETRY_DELAY).await;
                    }
                    Err(err) => {
                        tracing::error!(?err, "no free channels");
                        Err(err)?
                    }
                }
            }
        };
        tracing::info!(batch_idx, "batched-py-asr channel");
        in_tx.send(InMsg::Init)?;
        let (pcm, sample_rate) = crate::utils::pcm_decode(query)?;
        let pcm = if sample_rate == 24000 {
            pcm
        } else {
            kaudio::resample(&pcm, sample_rate as usize, 24000)?
        };
        in_tx.send(InMsg::Audio { pcm })?;
        in_tx.send(InMsg::Marker { id: 0 })?;
        in_tx.send(InMsg::Audio { pcm: vec![0f32; 240000] })?;
        let mut msgs = vec![];
        while let Some(msg) = out_rx.recv().await {
            match msg {
                OutMsg::Marker { .. } => break,
                OutMsg::Error { .. } | OutMsg::Word { .. } | OutMsg::EndWord { .. } => {
                    msgs.push(msg)
                }
                OutMsg::Ready | OutMsg::Step { .. } => {}
            }
        }
        Ok(msgs)
    }

    pub async fn handle_socket(&self, socket: ws::WebSocket, query: Query) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};
        use serde::Serialize;
        tracing::info!(?query, "py-asr handle-socket");
        metrics::CONNECT.inc();
        let (mut sender, receiver) = socket.split();
        let (bidx, in_tx, mut out_rx) = match self.channels()? {
            Some(x) => x,
            None => {
                tracing::error!("no free channels");
                let mut msg = vec![];
                OutMsg::Error { message: "no free channels".into() }.serialize(
                    &mut rmp_serde::Serializer::new(&mut msg)
                        .with_human_readable()
                        .with_struct_map(),
                )?;
                sender.send(ws::Message::binary(msg)).await?;
                sender.close().await?;
                anyhow::bail!("no free channels")
            }
        };
        tracing::info!(?bidx, "batched-py channel");
        in_tx.send(InMsg::Init)?;
        let recv_loop = task::spawn(async move {
            // There are two timeouts here:
            // - The short timeout handles the case where the client does not answer the regular pings.
            // - The long timeout handles the case where the client does not send valid data for a
            // long time.
            let short_timeout_duration = SEND_PING_EVERY * 3;
            let long_timeout_duration = std::time::Duration::from_secs(120);
            let mut last_message_received = std::time::Instant::now();
            let mut receiver = receiver;
            loop {
                use ws::Message;
                let msg = match timeout(short_timeout_duration, receiver.next()).await {
                    Ok(Some(msg)) => msg,
                    Ok(None) => break,
                    Err(_) => {
                        tracing::info!(?bidx, "recv loop short timeout");
                        break;
                    }
                };
                if last_message_received.elapsed() > long_timeout_duration {
                    tracing::info!(?bidx, "recv loop long timeout");
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

    pub fn config(&self) -> &crate::PyAsrConfig {
        &self.config
    }

    pub fn total_slots(&self) -> usize {
        self.config.batch_size
    }

    pub fn used_slots(&self) -> usize {
        self.channels.lock().unwrap().iter().filter(|v| v.is_some()).count()
    }
}
