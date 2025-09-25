// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::metrics::py as metrics;
use crate::PyStreamingQuery as Query;
use crate::StreamingOutput;
use anyhow::{Context, Result};
use axum::extract::ws;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3_ffi::c_str;
use std::sync::{Arc, Mutex};
use tokio::time::{timeout, Duration};

const FRAME_SIZE: usize = 1920;
const MASK_HAS_PCM: u8 = 1 << 0;
const MASK_IS_EOS: u8 = 1 << 1;
const MASK_WORD_FINISHED: u8 = 1 << 2;
const MASK_AR_STEP: u8 = 1 << 3;
const MASK_MISSING_WORDS: u8 = 1 << 4;

const SEND_PING_EVERY: Duration = Duration::from_secs(10);
const POST_RETRY_DELAY: Duration = Duration::from_millis(100);
const POST_MAX_RETRIES: usize = 1000;

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct TtsQuery {
    text: String,
    voice: String,
}

pub struct VerbosePyErr {
    err: PyErr,
}

impl From<PyErr> for VerbosePyErr {
    fn from(err: PyErr) -> Self {
        Self { err }
    }
}

fn get_traceback(py: Python<'_>, err: &PyErr) -> Result<String> {
    let traceback_mod = PyModule::import(py, "traceback")?;
    let func = traceback_mod.getattr("format_exception")?;
    let traceback_obj = func.call1((err.get_type(py), err.value(py), err.traceback(py)))?;
    let lines = traceback_obj.extract::<Vec<String>>()?;
    Ok(lines.join(""))
}

impl std::error::Error for VerbosePyErr {}

impl std::fmt::Display for VerbosePyErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Python::with_gil(|py| {
            let traceback = match get_traceback(py, &self.err) {
                Err(_) => "no traceback".to_string(),
                Ok(traceback) => traceback,
            };
            write!(f, "{}\n{}", self.err, traceback)
        })
    }
}

impl std::fmt::Debug for VerbosePyErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum InMsg {
    Text { text: String },
    Voice { embeddings: Vec<f32>, shape: Vec<usize> },
    Eos,
}

#[derive(Debug, Clone)]
pub enum Msg {
    Text(String, Vec<u32>),
    Voice { embeddings: Vec<f32>, shape: Vec<usize> },
    Eos,
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

type InSend = std::sync::mpsc::Sender<Msg>;
type InRecv = std::sync::mpsc::Receiver<Msg>;
type OutSend = tokio::sync::mpsc::UnboundedSender<Vec<u8>>;
type OutRecv = tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>;

struct Channel {
    id: ChannelId,
    in_rx: InRecv,
    out_tx: OutSend,
    encoder: crate::tts::Encoder,
    voice: Option<Voice>,
    sent_init: bool,
    words: std::collections::VecDeque<String>,
    steps: usize,
    prev_word_steps: usize,
}

impl Channel {
    fn new(
        in_rx: InRecv,
        out_tx: OutSend,
        encoder: crate::tts::Encoder,
        voice: Option<String>,
    ) -> Self {
        metrics::OPEN_CHANNELS.inc();
        let words = std::collections::VecDeque::new();
        Self {
            id: ChannelId::new(),
            in_rx,
            out_tx,
            encoder,
            words,
            voice: voice.map(Voice::File),
            sent_init: false,
            steps: 0,
            prev_word_steps: 0,
        }
    }

    fn on_end_of_word(&mut self) {
        if let Some(text) = self.words.pop_front() {
            let wwts = crate::tts::WordWithTimestamps {
                text,
                start_s: self.prev_word_steps as f64 / 12.5,
                stop_s: self.steps as f64 / 12.5,
            };
            match self.encoder.encode_word(wwts) {
                Ok(Some(msg)) => {
                    let _ = self.out_tx.send(msg).is_err();
                }
                Ok(None) => {}
                Err(err) => {
                    tracing::error!(?err, "encoder word error")
                }
            }
        }
    }
}

impl Drop for Channel {
    fn drop(&mut self) {
        metrics::CONNECTION_NUM_STEPS.observe(self.steps as f64);
        metrics::OPEN_CHANNELS.dec();
    }
}

pub fn init() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| -> PyResult<()> {
        let signal = py.import("signal")?;
        // Set SIGINT to have the default action rather than triggering a Python exception
        signal.getattr("signal")?.call1((signal.getattr("SIGINT")?, signal.getattr("SIG_DFL")?))?;
        Ok(())
    })?;
    Ok(())
}

type Channels = Arc<Mutex<Vec<Option<Channel>>>>;

struct Inner {
    channels: Channels,
    app: PyObject,
}

enum Voice {
    File(String),
    Embeddings { embeddings: Vec<f32>, shape: Vec<usize> },
}

impl<'py> IntoPyObject<'py> for Voice {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> std::result::Result<Self::Output, Self::Error> {
        use numpy::ToPyArray;
        let go = |s| -> PyResult<_> {
            let any = match s {
                Voice::File(v) => v.into_pyobject(py)?.into_any(),
                Voice::Embeddings { embeddings, shape } => match *shape.as_slice() {
                    [dim1] => embeddings.to_pyarray(py).reshape((dim1,))?.into_any(),
                    [dim1, dim2] => embeddings.to_pyarray(py).reshape((dim1, dim2))?.into_any(),
                    [d1, d2, d3] => embeddings.to_pyarray(py).reshape((d1, d2, d3))?.into_any(),
                    _ => return Ok(py.None().into_bound(py)),
                },
            };
            Ok(any)
        };
        // We convert errors to None, this should result in using the default voice rather than
        // crashing the whole process.
        match go(self) {
            Ok(any) => Ok(any),
            Err(_) => Ok(py.None().into_bound(py)),
        }
    }
}

// The arguments passed to the python step function, for now this is:
// (batch_idx, tokens, voice)
// tokens can include a -1 to indicate a new user, and a -2 to indicate
// end of stream.
type PyInput = (usize, Vec<i32>, Option<Voice>);

impl Inner {
    fn pre_process(&self, _step_idx: usize) -> Result<(Vec<PyInput>, Vec<Option<ChannelId>>)> {
        let mut channels = self.channels.lock().unwrap();
        let mut in_data = vec![];
        let mut channel_ids = Vec::with_capacity(channels.len());
        for (batch_idx, channel) in channels.iter_mut().enumerate() {
            channel_ids.push(channel.as_ref().map(|c| c.id));
            if let Some(c) = channel.as_mut() {
                if c.out_tx.is_closed() {
                    *channel = None;
                } else {
                    use std::sync::mpsc::TryRecvError;
                    match c.in_rx.try_recv() {
                        Ok(Msg::Text(word, tokens)) => {
                            c.words.push_back(word);
                            let mut t = Vec::with_capacity(tokens.len() + 1);
                            if !c.sent_init {
                                t.push(-1);
                                c.sent_init = true;
                            }
                            for &v in tokens.iter() {
                                t.push(v as i32);
                            }
                            in_data.push((batch_idx, t, c.voice.take()));
                        }
                        Ok(Msg::Voice { embeddings, shape }) => {
                            c.voice = Some(Voice::Embeddings { embeddings, shape });
                        }
                        Ok(Msg::Eos) => {
                            if c.sent_init {
                                in_data.push((batch_idx, vec![-2], None));
                            } else {
                                *channel = None
                            }
                        }
                        Err(TryRecvError::Empty) => {}
                        Err(TryRecvError::Disconnected) => *channel = None,
                    }
                };
            }
        }
        Ok((in_data, channel_ids))
    }

    fn start_model_loop(self, batch_size: usize) -> Result<()> {
        use numpy::{PyArrayMethods, ToPyArray};
        use rayon::prelude::*;
        use std::ops::DerefMut;

        crate::utils::spawn_blocking("model_loop", move || {
            // Maybe the model loop could just always hold the gil?
            tracing::info!("starting-up the py model loop");
            let pcm_data = numpy::ndarray::Array2::<f32>::zeros([batch_size, FRAME_SIZE]);
            let pcm_data = Python::with_gil(|py| pcm_data.to_pyarray(py).unbind());
            let mask = numpy::ndarray::Array1::<u8>::zeros([batch_size]);
            let mask = Python::with_gil(|py| mask.to_pyarray(py).unbind());
            let tokens = numpy::ndarray::Array2::<i32>::zeros([batch_size, 33]);
            let tokens = Python::with_gil(|py| tokens.to_pyarray(py).unbind());

            for step_idx in 0.. {
                // We store the channel ids here to check that they have not changed when sending
                // the data back to the user.
                let (in_data, channel_ids) = self.pre_process(step_idx)?;
                if channel_ids.iter().all(|c| c.is_none()) {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    continue;
                }
                let start_time = std::time::Instant::now();
                Python::with_gil(|py| -> Result<()> {
                    self.app
                        .call_method1(py, "step", (in_data, &pcm_data, &mask, &tokens))
                        .map_err(VerbosePyErr::from)?;
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let pcm = pcm_data.bind(py).readonly();
                    let mask = mask.bind(py).readonly();
                    let tokens = tokens.bind(py).readonly();
                    let pcm = pcm.as_slice().context("pcm is not contiguous")?;
                    let mask = mask.as_slice().context("mask is not contiguous")?;
                    let _tokens = tokens.as_slice().context("tokens is not contiguous")?;

                    // Only store the sample is something was actually done.
                    if mask.iter().any(|&x| (x & MASK_AR_STEP) > 0) {
                        metrics::MODEL_STEP_DURATION.observe(elapsed);
                        metrics::ACTIVE_STEPS.inc();
                    }
                    metrics::TOTAL_STEPS.inc();

                    let mut channels = self.channels.lock().unwrap();
                    let c = channels.deref_mut();

                    c.par_iter_mut().enumerate().for_each(|(batch_idx, channel)| {
                        if let Some(c) = channel.as_mut() {
                            let mask = mask[batch_idx];
                            // The channel has changed so skip the update.
                            if Some(c.id) != channel_ids[batch_idx] || !c.sent_init {
                                return;
                            }
                            if (mask & MASK_AR_STEP) > 0 {
                                c.steps += 1;
                            }
                            if (mask & MASK_MISSING_WORDS) > 0 {
                                metrics::MISSING_WORDS_STEPS.inc();
                            } else {
                                metrics::COULD_HAVE_RUN_STEPS.inc();
                            }
                            if (mask & MASK_WORD_FINISHED) > 0 {
                                // MASK_WORD_FINISHED currently indicates the beggining of a new
                                // word rather than the end of one.
                                if c.prev_word_steps > 0 {
                                    c.on_end_of_word();
                                }
                                c.prev_word_steps = c.steps;
                            }
                            if (mask & MASK_HAS_PCM) > 0 {
                                let pcm = pcm[batch_idx * FRAME_SIZE..(batch_idx + 1) * FRAME_SIZE]
                                    .to_vec();
                                match c.encoder.encode(pcm) {
                                    Ok(msg) => {
                                        let _ = c.out_tx.send(msg).is_err();
                                    }
                                    Err(err) => {
                                        tracing::error!(?err, ?batch_idx, "encoder error")
                                    }
                                }
                            }
                            // The TTS has finished generating so we close the channel, this should
                            // drop out_tx and result in the websock closing.
                            if (mask & MASK_IS_EOS) > 0 {
                                c.on_end_of_word();
                                tracing::info!(?batch_idx, "tts finished");
                                *channel = None;
                            }
                        }
                    });
                    Ok(())
                })?;
            }
            Ok(())
        });
        Ok(())
    }
}

#[derive(Clone)]
pub struct M {
    channels: Channels,
    text_tokenizer: std::sync::Arc<sentencepiece::SentencePieceProcessor>,
    config: crate::PyConfig,
}

pub(crate) fn toml_to_py<'a>(py: Python<'a>, value: &toml::Value) -> Result<Bound<'a, PyAny>> {
    let value = match value {
        toml::Value::Float(v) => v.into_pyobject(py)?.into_any(),
        toml::Value::Integer(v) => v.into_pyobject(py)?.into_any(),
        toml::Value::String(v) => v.into_pyobject(py)?.into_any(),
        toml::Value::Boolean(v) => v.into_pyobject(py)?.to_owned().into_any(),
        toml::Value::Table(table) => {
            let v = pyo3::types::PyDict::new(py);
            for (key, value) in table.iter() {
                v.set_item(key, toml_to_py(py, value)?)?;
            }
            v.into_any()
        }
        toml::Value::Array(vs) => {
            let v = pyo3::types::PyList::empty(py);
            for value in vs.iter() {
                v.append(toml_to_py(py, value)?)?;
            }
            v.into_any()
        }
        toml::Value::Datetime(_) => {
            anyhow::bail!("unsupported value type DateTime")
        }
    };
    Ok(value)
}

impl M {
    pub fn new(config: crate::PyConfig) -> Result<Self> {
        init()?;
        let text_tokenizer =
            sentencepiece::SentencePieceProcessor::open(&config.text_tokenizer_file)
                .with_context(|| config.text_tokenizer_file.clone())?;
        let batch_size = config.batch_size;
        let (script, script_name) = match &config.script {
            None => {
                let script_name = std::ffi::CString::new("tts.py")?;
                let script = std::ffi::CString::new(crate::TTS_PY)?;
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
        let channels = (0..batch_size).map(|_| None).collect::<Vec<_>>();
        let channels = Arc::new(Mutex::new(channels));
        let text_tokenizer = Arc::new(text_tokenizer);
        let inner = Inner { app, channels: channels.clone() };
        inner.start_model_loop(batch_size)?;
        Ok(Self { config, channels, text_tokenizer })
    }

    // Returns None if no channel is available at the moment.
    fn channels(
        &self,
        format: StreamingOutput,
        voice: Option<String>,
    ) -> Result<Option<(usize, InSend, OutRecv)>> {
        let mut channels = self.channels.lock().unwrap();
        // Linear scan to find an available channel. This is fairly inefficient, instead we should
        // probably have a queue of available slots.
        for (batch_idx, channel) in channels.iter_mut().enumerate() {
            if channel.is_none() {
                let (in_tx, in_rx) = std::sync::mpsc::channel::<Msg>();
                let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<Vec<u8>>();
                let mut encoder = crate::tts::Encoder::new(format)?;
                if let Some(msg) = encoder.encode_msg(crate::tts::OutMsg::Ready)? {
                    out_tx.send(msg)?
                }
                if let Some(header) = encoder.header()? {
                    out_tx.send(header)?
                }
                let c = Channel::new(in_rx, out_tx, encoder, voice.clone());
                *channel = Some(c);
                return Ok(Some((batch_idx, in_tx, out_rx)));
            }
        }
        Ok(None)
    }

    // TODO: Add a proper batch variant that would enqueue the task so that it can be processed
    // when there is a free channel.
    pub async fn handle_query(
        &self,
        query: &TtsQuery,
    ) -> Result<tokio_stream::wrappers::ReceiverStream<Result<bytes::Bytes, std::io::Error>>> {
        use bytes::Bytes;
        use std::io;
        use tokio::sync::mpsc;
        use tokio_stream::wrappers::ReceiverStream;

        tracing::info!("py handle-query");
        metrics::CONNECT.inc();
        let (batch_idx, in_tx, mut out_rx) = {
            let mut num_tries = 0;
            loop {
                match self.channels(StreamingOutput::Pcm, Some(query.voice.clone())) {
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
        tracing::info!(batch_idx, "batched-py channel");
        let mut text_tokenizer = crate::tts_preprocess::Tokenizer::new(
            self.text_tokenizer.clone(),
            self.config().text_bos_token,
        );
        let wwts = text_tokenizer.preprocess(&query.text)?;
        for wwt in wwts.into_iter() {
            in_tx.send(Msg::Text(wwt.word, wwt.tokens))?;
        }
        in_tx.send(Msg::Eos)?;

        let (tx, rx) = mpsc::channel::<Result<Bytes, io::Error>>(32);

        crate::utils::spawn("send_loop", async move {
            let mut wav_buffer = Vec::new();
            moshi::wav::write_wav_header(&mut wav_buffer, 24_000, 0xFFFF_FFFFu32, 0xFFFF_FFFFu32)?;
            tx.send(Ok(Bytes::from(wav_buffer.clone()))).await?;
            while let Some(data) = out_rx.recv().await {
                wav_buffer.clear();
                let pcm = {
                    use byteorder::ByteOrder;
                    let mut buf = vec![0f32; data.len() / std::mem::size_of::<f32>()];
                    byteorder::LittleEndian::read_f32_into(&data, &mut buf);
                    buf
                };

                moshi::wav::write_pcm_in_wav(&mut wav_buffer, &pcm)?;
                tx.send(Ok(Bytes::from(wav_buffer.clone()))).await?;
            }

            // We want in_tx to stay open while the send_loop is running, so we capture it
            std::mem::drop(in_tx);
            Ok::<_, anyhow::Error>(())
        });
        Ok(ReceiverStream::new(rx))
    }

    pub async fn handle_socket(&self, socket: ws::WebSocket, query: Query) -> Result<()> {
        use futures_util::{SinkExt, StreamExt};

        tracing::info!(?query, "py query");
        metrics::CONNECT.inc();

        let (mut sender, receiver) = socket.split();
        let (bidx, in_tx, mut out_rx) = match self.channels(query.format, query.voice.clone())? {
            Some(x) => x,
            None => {
                tracing::error!("no free channels");
                let mut encoder = crate::tts::Encoder::new(query.format)?;
                let msg = crate::tts::OutMsg::Error { message: "no free channels".into() };
                if let Some(msg) = encoder.encode_msg(msg)? {
                    sender.send(ws::Message::binary(msg)).await?;
                    sender.close().await?;
                }
                anyhow::bail!("no free channels")
            }
        };
        tracing::info!(?bidx, "batched-py channel");
        let mut text_tokenizer = crate::tts_preprocess::Tokenizer::new(
            self.text_tokenizer.clone(),
            self.config().text_bos_token,
        );

        crate::utils::spawn("recv_loop", async move {
            let timeout_duration = SEND_PING_EVERY * 3;
            let mut receiver = receiver;
            let mut send_text = |msg: &str| -> Result<()> {
                let wwts = text_tokenizer.preprocess(msg)?;
                for wwt in wwts.into_iter() {
                    in_tx.send(Msg::Text(wwt.word, wwt.tokens))?;
                }
                Ok(())
            };
            loop {
                use ws::Message;
                let msg = match timeout(timeout_duration, receiver.next()).await {
                    Ok(Some(msg)) => msg,
                    Ok(None) => break,
                    Err(_) => {
                        tracing::info!(?bidx, "recv loop short timeout");
                        break;
                    }
                };
                match msg? {
                    Message::Text(text) => send_text(&text)?,
                    Message::Binary(msg) => {
                        if msg.as_ref() == b"\0" {
                            tracing::info!(?bidx, "received end of stream");
                            in_tx.send(Msg::Eos)?
                        } else {
                            let msg: InMsg = rmp_serde::from_slice(&msg)?;
                            match msg {
                                InMsg::Eos => in_tx.send(Msg::Eos)?,
                                InMsg::Text { text } => send_text(&text)?,
                                InMsg::Voice { embeddings, shape } => {
                                    in_tx.send(Msg::Voice { embeddings, shape })?
                                }
                            }
                        }
                    }
                    // ping messages are automatically answered by tokio-tungstenite as long as
                    // the connection is read from.
                    Message::Ping(_) | Message::Pong(_) => {}
                    Message::Close(_) => break,
                };
            }
            Ok::<_, anyhow::Error>(())
        });
        crate::utils::spawn("send_loop", async move {
            let mut sender = sender;
            let mut last_ping_sent = std::time::Instant::now();
            loop {
                // The recv method is cancel-safe so can be wrapped in a timeout.
                let msg = timeout(SEND_PING_EVERY, out_rx.recv()).await;
                let now = std::time::Instant::now();
                if now.duration_since(last_ping_sent) > SEND_PING_EVERY {
                    last_ping_sent = now;
                    sender.send(ws::Message::Ping(vec![].into())).await?;
                }
                if let Ok(msg) = msg {
                    match msg {
                        None => break,
                        Some(msg) => {
                            let msg = ws::Message::binary(msg);
                            sender.send(msg).await?;
                        }
                    }
                };
            }
            sender.close().await?;
            drop(sender);
            Ok::<(), anyhow::Error>(())
        });
        Ok(())
    }

    pub fn config(&self) -> &crate::PyConfig {
        &self.config
    }

    pub fn total_slots(&self) -> usize {
        self.config.batch_size
    }

    pub fn used_slots(&self) -> usize {
        self.channels.lock().unwrap().iter().filter(|v| v.is_some()).count()
    }
}
