// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use axum::{http::StatusCode, response::IntoResponse, response::Response};
use candle::Device;
use std::str::FromStr;
use std::sync::Arc;

mod asr;
mod batched_asr;
mod legacy_tts;
mod lm;
mod metrics;
mod mimi;
mod protocol;
mod py_module;
mod py_module_post;
mod tts;
mod utils;

const ID_HEADER: &str = "kyutai-api-key";
const ROOM_ID_HEADER: &str = "room_id";

#[derive(clap::Parser, Debug)]
struct WorkerArgs {
    #[clap(short = 'l', long = "log", default_value = "info")]
    log_level: String,

    #[clap(short = 'a', long = "addr", default_value = "0.0.0.0")]
    addr: String,

    #[clap(short = 'p', long = "port", default_value = "8080")]
    port: u16,

    #[clap(long)]
    cpu: bool,

    #[clap(long)]
    config: String,

    #[clap(long)]
    silent: bool,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Validate { configs: Vec<String> },
    Worker(WorkerArgs),
}

#[derive(clap::Parser, Debug)]
#[clap(name = "server", about = "Kyutai moshi server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TtsConfig {
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub speaker_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub voices: std::collections::HashMap<String, String>,
    pub voice_dir: String,
    pub model: moshi::lm::Config,
    pub generation: moshi::tts_streaming::Config,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct AsrConfig {
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub model: moshi::lm::Config,
    pub asr_delay_in_tokens: usize,
    #[serde(default)]
    pub log_frequency_s: Option<f64>,
    #[serde(default)]
    pub conditioning_delay: Option<f32>,
    // The default for bools in rust is false.
    #[serde(default)]
    pub conditioning_learnt_padding: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MimiConfig {
    pub audio_tokenizer_file: String,
    pub auth_recv: bool,
    pub rooms: Vec<String>,
    pub default_room: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LmConfig {
    pub lm_model_file: String,
    pub text_tokenizer_file: String,
    pub audio_tokenizer_file: String,
    pub model: moshi::lm::Config,
    pub gen: moshi::lm_generate_multistream::Config,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PyConfig {
    pub script: String,
    pub batch_size: usize,
    pub text_tokenizer_file: String,
    pub text_bos_token: u32,
    #[serde(default)]
    pub py: Option<toml::Table>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PyPostConfig {
    pub script: String,
    #[serde(default)]
    pub py: Option<toml::Table>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ModuleConfig {
    Tts {
        path: String,
        #[serde(flatten)]
        config: TtsConfig,
    },
    Asr {
        path: String,
        #[serde(flatten)]
        config: AsrConfig,
    },
    BatchedAsr {
        path: String,
        #[serde(flatten)]
        config: AsrConfig,
        batch_size: usize,
    },
    Mimi {
        send_path: String,
        recv_path: String,
        #[serde(flatten)]
        config: MimiConfig,
    },
    Lm {
        path: String,
        #[serde(flatten)]
        config: LmConfig,
    },
    Py {
        path: String,
        #[serde(flatten)]
        config: PyConfig,
    },
    PyPost {
        path: String,
        #[serde(flatten)]
        config: PyPostConfig,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub static_dir: String,
    pub log_dir: String,
    pub instance_name: String,
    #[serde(default)]
    pub modules: std::collections::HashMap<String, ModuleConfig>,
    pub authorized_ids: std::collections::HashSet<String>,
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        use utils::resolve_or_download as rod;
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = toml::from_str(&config)?;
        for (_, c) in config.modules.iter_mut() {
            match c {
                ModuleConfig::Mimi { send_path: _, recv_path: _, config: c } => {
                    c.audio_tokenizer_file = rod(&c.audio_tokenizer_file)?;
                }
                ModuleConfig::Tts { path: _, config: c } => {
                    c.lm_model_file = rod(&c.lm_model_file)?;
                    c.text_tokenizer_file = rod(&c.text_tokenizer_file)?;
                    c.speaker_tokenizer_file = rod(&c.speaker_tokenizer_file)?;
                    c.audio_tokenizer_file = rod(&c.audio_tokenizer_file)?;
                    for (_, v) in c.voices.iter_mut() {
                        *v = rod(v)?
                    }
                    c.voice_dir = rod(&c.voice_dir)?;
                }
                ModuleConfig::BatchedAsr { path: _, config: c, batch_size: _ } => {
                    c.lm_model_file = rod(&c.lm_model_file)?;
                    c.text_tokenizer_file = rod(&c.text_tokenizer_file)?;
                    c.audio_tokenizer_file = rod(&c.audio_tokenizer_file)?;
                }
                ModuleConfig::Asr { path: _, config: c } => {
                    c.lm_model_file = rod(&c.lm_model_file)?;
                    c.text_tokenizer_file = rod(&c.text_tokenizer_file)?;
                    c.audio_tokenizer_file = rod(&c.audio_tokenizer_file)?;
                }
                ModuleConfig::Lm { path: _, config: c } => {
                    c.audio_tokenizer_file = rod(&c.audio_tokenizer_file)?;
                    c.text_tokenizer_file = rod(&c.text_tokenizer_file)?;
                    c.lm_model_file = rod(&c.lm_model_file)?;
                }
                ModuleConfig::Py { path: _, config: c } => {
                    c.script = rod(&c.script)?;
                    c.text_tokenizer_file = rod(&c.text_tokenizer_file)?;
                    if let Some(t) = c.py.as_mut() {
                        crate::utils::resolve_or_download_toml(t)?;
                    }
                }
                ModuleConfig::PyPost { path: _, config: c } => {
                    c.script = rod(&c.script)?;
                    if let Some(t) = c.py.as_mut() {
                        crate::utils::resolve_or_download_toml(t)?;
                    }
                }
            }
        }
        config.static_dir = rod(&config.static_dir)?;
        config.log_dir = rod(&config.log_dir)?;
        config.instance_name = rod(&config.instance_name)?;
        Ok(config)
    }
}

fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

#[allow(unused)]
enum Module {
    Tts { path: String, m: Arc<tts::Model> },
    Asr { path: String, m: Arc<asr::Asr> },
    BatchedAsr { path: String, m: Arc<batched_asr::BatchedAsr> },
    Mimi { send_path: String, recv_path: String, m: Arc<mimi::Mimi> },
    Lm { path: String, m: Arc<lm::Lm> },
    Py { path: String, m: Arc<py_module::M> },
    PyPost { path: String, m: Arc<py_module_post::M> },
}

struct SharedStateInner {
    config: Config,
}

type SharedState = Arc<SharedStateInner>;

fn lm_router(s: Arc<lm::Lm>, path: &str) -> axum::Router<()> {
    async fn lm_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<lm::Lm>,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket).await {
            tracing::error!(?err, "lm")
        }
    }

    async fn lm_streaming(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<Arc<lm::Lm>>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling lm-streaming query");
        let state = state.0.clone();
        let upg = ws.write_buffer_size(0).on_upgrade(move |v| lm_websocket(v, state, addr));
        Ok(upg)
    }

    axum::Router::new().route(path, axum::routing::get(lm_streaming)).with_state(s)
}

impl Module {
    fn new(module_cfg: &ModuleConfig, full_cfg: &Config, dev: &Device) -> Result<Self> {
        let m = match module_cfg {
            ModuleConfig::Lm { path, config } => {
                let m = lm::Lm::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                Self::Lm { m, path: path.to_string() }
            }
            ModuleConfig::Asr { path, config } => {
                let m = asr::Asr::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                tracing::info!("warming up the asr");
                m.warmup()?;
                tracing::info!("done warming up the asr, ready to roll!");
                Self::Asr { m, path: path.to_string() }
            }
            ModuleConfig::BatchedAsr { path, config, batch_size } => {
                let m = batched_asr::BatchedAsr::new(*batch_size, config, full_cfg, dev)?;
                let m = Arc::new(m);
                Self::BatchedAsr { m, path: path.to_string() }
            }
            ModuleConfig::Tts { path, config } => {
                let voice = config.voices.keys().next();
                let m = tts::Model::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                if let Some(voice) = voice {
                    tracing::info!(voice, "warming up the tts");
                    m.run(&TtsQuery {
                        text: vec!["hello".to_string()],
                        seed: 42,
                        temperature: 0.8,
                        top_k: 250,
                        voice: Some(voice.clone()),
                        voices: None,
                        max_seq_len: None,
                        return_timestamps: None,
                        cfg_alpha: None,
                    })?;
                    tracing::info!("done warming up the tts, ready to roll!");
                }
                Self::Tts { m, path: path.to_string() }
            }
            ModuleConfig::Mimi { send_path, recv_path, config } => {
                let m = mimi::Mimi::new(config, full_cfg, dev)?;
                let m = Arc::new(m);
                Self::Mimi { m, send_path: send_path.to_string(), recv_path: recv_path.to_string() }
            }
            ModuleConfig::Py { path, config } => {
                let m = py_module::M::new(config.clone())?;
                let m = Arc::new(m);
                Self::Py { m, path: path.to_string() }
            }
            ModuleConfig::PyPost { path, config } => {
                let m = py_module_post::M::new(config.clone())?;
                let m = Arc::new(m);
                Self::PyPost { m, path: path.to_string() }
            }
        };
        Ok(m)
    }

    fn router(&self, shared_state: &SharedState) -> Result<axum::Router<()>> {
        let router = match self {
            Self::Lm { path, m } => lm_router(m.clone(), path),
            Self::Asr { path, m } => asr_router(m.clone(), path, shared_state),
            Self::BatchedAsr { path, m } => batched_asr_router(m.clone(), path, shared_state),
            Self::Tts { path, m } => tts_router(m.clone(), path, shared_state),
            Self::Mimi { send_path, recv_path, m } => {
                mimi_router(m.clone(), send_path, recv_path, shared_state)
            }
            Self::Py { path, m } => py_router(m.clone(), path, shared_state),
            Self::PyPost { path, m } => py_router_post(m.clone(), path, shared_state),
        };
        Ok(router)
    }
}

struct AppStateInner {
    modules: Vec<Module>,
}

type AppState = Arc<AppStateInner>;

impl AppStateInner {
    fn new(args: &WorkerArgs, config: Config) -> Result<Self> {
        let device = device(args.cpu)?;

        // The following does not have a significant impact as soon as batch sizes are
        // large enough so we don't activate it for now.
        // #[cfg(feature = "cuda")]
        // if let candle::Device::Cuda(d) = &device {
        //     unsafe {
        //         d.disable_event_tracking();
        //     }
        // };

        let mut modules = Vec::with_capacity(config.modules.len());
        for (_, module_cfg) in config.modules.iter() {
            let m = Module::new(module_cfg, &config, &device)?;
            modules.push(m)
        }
        Ok(Self { modules })
    }
}

fn tracing_init(
    log_dir: &str,
    instance_name: &str,
    log_level: &str,
    silent: bool,
) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    use tracing_subscriber::prelude::*;

    let build_info = utils::BuildInfo::new();
    let file_appender = tracing_appender::rolling::daily(log_dir, format!("log.{}", instance_name));
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
    let filter = tracing_subscriber::filter::LevelFilter::from_str(log_level)?;
    let mut layers = vec![tracing_subscriber::fmt::layer()
        .event_format(tracing_subscriber::fmt::format().with_file(true).with_line_number(true))
        .with_writer(non_blocking)
        .with_filter(filter)
        .boxed()];
    if !silent {
        layers.push(Box::new(
            tracing_subscriber::fmt::layer()
                .event_format(
                    tracing_subscriber::fmt::format().with_file(true).with_line_number(true),
                )
                .with_writer(std::io::stdout)
                .with_filter(filter),
        ))
    };
    tracing_subscriber::registry().with(layers).init();
    tracing::info!(?build_info);
    Ok(guard)
}

async fn metrics(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    _state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    use prometheus::Encoder;

    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    if let Err(err) = encoder.encode(&metric_families, &mut buffer) {
        return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response();
    };
    axum::response::Response::builder()
        .status(200)
        .header(axum::http::header::CONTENT_TYPE, encoder.format_type())
        .body(axum::body::Body::from(buffer))
        .unwrap()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    // When an error bubbles up in the tokio main function, the whole program does not
    // seem to crash if some background tasks are still running.
    // This can lead to errors such as "port already in use" not being reported so we
    // exit the process explicitely here.
    if let Err(err) = main_().await {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

async fn main_() -> Result<()> {
    let args = <Args as clap::Parser>::parse();
    match args.command {
        Command::Validate { configs } => {
            tracing_subscriber::fmt().init();
            for config in configs.iter() {
                let _ = Config::load(config)?;
                tracing::info!(?config, "loaded succesfully")
            }
        }
        Command::Worker(args) => {
            use axum::routing::get;

            let config = Config::load(&args.config)?;
            if std::env::var("RUST_LOG").is_err() {
                std::env::set_var("RUST_LOG", format!("{},hyper=info,mio=info", args.log_level))
            }
            let _guard =
                tracing_init(&config.log_dir, &config.instance_name, &args.log_level, args.silent)?;
            let num_workers = tokio::runtime::Handle::current().metrics().num_workers();
            tracing::info!(num_workers, "starting worker");

            let static_dir = utils::resolve_or_download(&config.static_dir)?;
            let shared_state = Arc::new(SharedStateInner { config: config.clone() });
            let state = Arc::new(AppStateInner::new(&args, config)?);
            let mut app = axum::Router::new()
                .route("/api/build_info", get(build_info))
                .route("/api/modules_info", get(modules_info))
                .route("/metrics", axum::routing::get(metrics))
                .fallback_service(
                    tower_http::services::ServeDir::new(&static_dir)
                        .append_index_html_on_directories(true),
                )
                .layer(
                    tower::ServiceBuilder::new()
                        .layer(tower_http::trace::TraceLayer::new_for_http()),
                )
                .with_state(state.clone());
            for module in state.modules.iter() {
                app = app.merge(module.router(&shared_state)?)
            }

            let sock_addr = std::net::SocketAddr::from((
                std::net::IpAddr::from_str(args.addr.as_str())
                    .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
                args.port,
            ));
            tracing::info!("listening on http://{}", sock_addr);
            let listener = tokio::net::TcpListener::bind(sock_addr).await?;
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await?;
        }
    }
    Ok(())
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq)]
enum StreamingOutput {
    Pcm,
    PcmMessagePack,
    OggOpus,
    OggOpusMessagePack,
}
fn default_seed() -> u64 {
    42
}
fn default_temperature() -> f64 {
    0.8
}
fn default_top_k() -> usize {
    250
}
fn default_format() -> StreamingOutput {
    StreamingOutput::OggOpus
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct TtsStreamingQuery {
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default = "default_top_k")]
    top_k: usize,
    #[serde(default = "default_format")]
    format: StreamingOutput,
    voice: Option<String>,
    voices: Option<Vec<String>>,
    max_seq_len: Option<usize>,
    cfg_alpha: Option<f64>,
    auth_id: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct TtsQuery {
    text: Vec<String>,
    seed: u64,
    temperature: f64,
    top_k: usize,
    voice: Option<String>,
    voices: Option<Vec<String>>,
    max_seq_len: Option<usize>,
    return_timestamps: Option<bool>,
    cfg_alpha: Option<f64>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct TtsResponse {
    wav: String,
    transcript: Vec<crate::tts::WordWithTimestamps>,
}

fn tts_router(s: Arc<tts::Model>, path: &str, ss: &SharedState) -> axum::Router<()> {
    use base64::Engine;

    async fn tts_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<tts::Model>,
        query: TtsStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "tts")
        }
    }

    async fn t(
        state: axum::extract::State<(Arc<tts::Model>, SharedState)>,
        headers: axum::http::HeaderMap,
        req: axum::Json<TtsQuery>,
    ) -> utils::AxumResult<Response> {
        tracing::info!("handling tts query {req:?}");
        let valid_id = headers
            .get(ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|id| state.0 .1.config.authorized_ids.contains(id));
        if !valid_id {
            return Ok(StatusCode::UNAUTHORIZED.into_response());
        }
        let (wav, transcript) = {
            let _guard = state.0 .0.mutex.lock().await;
            state.0 .0.run(&req)?
        };
        tracing::info!("ok {}", wav.len());
        if req.return_timestamps.unwrap_or(false) {
            let data =
                TtsResponse { wav: base64::prelude::BASE64_STANDARD.encode(wav), transcript };
            Ok((
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                axum::Json(data),
            )
                .into_response())
        } else {
            Ok((StatusCode::OK, [(axum::http::header::CONTENT_TYPE, "audio/wav")], wav)
                .into_response())
        }
    }

    async fn streaming_t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<tts::Model>, SharedState)>,
        req: axum::extract::Query<TtsStreamingQuery>,
    ) -> utils::AxumResult<Response> {
        tracing::info!("handling tts streaming query {req:?}");
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        // It's tricky to set the headers of a websocket in javascript so we pass the token via the
        // query too.
        let auth_id = match headers.get(ID_HEADER) {
            Some(v) => v.to_str().ok(),
            None => req.auth_id.as_deref(),
        };
        let valid_id = auth_id.is_some_and(|id| state.1.config.authorized_ids.contains(id));
        if !valid_id {
            return Ok(StatusCode::UNAUTHORIZED.into_response());
        }
        let tts_query = req.0.clone();
        let tts = state.0 .0.clone();
        let upg =
            ws.write_buffer_size(0).on_upgrade(move |v| tts_websocket(v, tts, tts_query, addr));
        Ok(upg)
    }

    axum::Router::new()
        .route(path, axum::routing::post(t))
        .route(&format!("{path}_streaming"), axum::routing::get(streaming_t))
        .with_state((s, ss.clone()))
}

async fn build_info(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    _state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    let build_info = utils::BuildInfo::new();
    utils::WrapJson(Ok(build_info)).into_response()
}

async fn modules_info(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    let modules: Vec<_> = state
        .modules
        .iter()
        .filter_map(|m| match m {
            Module::BatchedAsr { path, m } => {
                let config = m.config();
                let mut info = std::collections::HashMap::new();
                info.insert("type", "batched_asr".to_string());
                info.insert("path", path.to_string());
                info.insert("lm", config.lm_model_file.clone());
                info.insert("audio_tokenizer", config.audio_tokenizer_file.clone());
                info.insert("used_slots", m.used_slots().to_string());
                info.insert("total_slots", m.total_slots().to_string());
                Some(info)
            }
            Module::Py { path, m } => {
                let config = m.config();
                let mut info = std::collections::HashMap::new();
                info.insert("type", "py".to_string());
                info.insert("path", path.to_string());
                info.insert("script", config.script.to_string());
                info.insert("used_slots", m.used_slots().to_string());
                info.insert("total_slots", m.total_slots().to_string());
                Some(info)
            }
            _ => None,
        })
        .collect();
    utils::WrapJson(Ok(modules)).into_response()
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct AsrStreamingQuery {
    auth_id: Option<String>,
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct PyStreamingQuery {
    auth_id: Option<String>,
    #[serde(default = "default_format")]
    format: StreamingOutput,
    #[serde(default)]
    voice: Option<String>,
}

fn asr_router(s: Arc<asr::Asr>, path: &str, ss: &SharedState) -> axum::Router<()> {
    async fn asr_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<asr::Asr>,
        query: AsrStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "asr")
        }
    }

    async fn t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<asr::Asr>, SharedState)>,
        req: axum::extract::Query<AsrStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling asr-streaming query");
        // It's tricky to set the headers of a websocket in javascript so we pass the token via the
        // query too.
        let auth_id = match headers.get(ID_HEADER) {
            Some(v) => v.to_str().ok(),
            None => req.auth_id.as_deref(),
        };
        let valid_id = auth_id.is_some_and(|id| state.1.config.authorized_ids.contains(id));
        if !valid_id {
            return Ok(StatusCode::UNAUTHORIZED.into_response());
        }
        let asr_query = req.0.clone();
        let asr = state.0 .0.clone();
        let upg =
            ws.write_buffer_size(0).on_upgrade(move |v| asr_websocket(v, asr, asr_query, addr));
        Ok(upg)
    }
    axum::Router::new().route(path, axum::routing::get(t)).with_state((s, ss.clone()))
}

fn batched_asr_router(
    s: Arc<batched_asr::BatchedAsr>,
    path: &str,
    ss: &SharedState,
) -> axum::Router<()> {
    async fn asr_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<batched_asr::BatchedAsr>,
        query: AsrStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "asr")
        }
    }

    async fn t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<batched_asr::BatchedAsr>, SharedState)>,
        req: axum::extract::Query<AsrStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling batched asr-streaming query");
        // It's tricky to set the headers of a websocket in javascript so we pass the token via the
        // query too.
        let auth_id = match headers.get(ID_HEADER) {
            Some(v) => v.to_str().ok(),
            None => req.auth_id.as_deref(),
        };
        let valid_id = auth_id.is_some_and(|id| state.1.config.authorized_ids.contains(id));
        if !valid_id {
            return Ok(StatusCode::UNAUTHORIZED.into_response());
        }
        let asr_query = req.0.clone();
        let asr = state.0 .0.clone();
        let upg =
            ws.write_buffer_size(0).on_upgrade(move |v| asr_websocket(v, asr, asr_query, addr));
        Ok(upg)
    }
    axum::Router::new().route(path, axum::routing::get(t)).with_state((s, ss.clone()))
}

fn py_router_post(s: Arc<py_module_post::M>, path: &str, ss: &SharedState) -> axum::Router<()> {
    async fn t(
        state: axum::extract::State<(Arc<py_module_post::M>, SharedState)>,
        _headers: axum::http::HeaderMap,
        req: axum::body::Bytes,
    ) -> utils::AxumResult<Response> {
        tracing::info!("handling py-post query");
        match state.0 .0.run_one(req).await {
            Ok(data) => Ok((StatusCode::OK, data).into_response()),
            Err(err) => {
                tracing::error!(?err, "py-post");
                Ok(StatusCode::INTERNAL_SERVER_ERROR.into_response())
            }
        }
    }

    axum::Router::new()
        .route(path, axum::routing::post(t))
        .with_state((s, ss.clone()))
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(16 * 1024 * 1024))
}

fn py_router(s: Arc<py_module::M>, path: &str, ss: &SharedState) -> axum::Router<()> {
    async fn py_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<py_module::M>,
        query: PyStreamingQuery,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.handle_socket(socket, query).await {
            tracing::error!(?err, "py")
        }
    }

    async fn t(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<py_module::M>, SharedState)>,
        req: axum::extract::Query<PyStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling py streaming query");
        // It's tricky to set the headers of a websocket in javascript so we pass the token via the
        // query too.
        let auth_id = match headers.get(ID_HEADER) {
            Some(v) => v.to_str().ok(),
            None => req.auth_id.as_deref(),
        };
        let valid_id = auth_id.is_some_and(|id| state.1.config.authorized_ids.contains(id));
        if !valid_id {
            return Ok(StatusCode::UNAUTHORIZED.into_response());
        }
        let py_query = req.0.clone();
        let py = state.0 .0.clone();
        let upg = ws.write_buffer_size(0).on_upgrade(move |v| py_websocket(v, py, py_query, addr));
        Ok(upg)
    }
    axum::Router::new().route(path, axum::routing::get(t)).with_state((s, ss.clone()))
}

#[derive(serde::Deserialize, serde::Serialize, Debug, Clone)]
struct MimiStreamingQuery {
    auth_id: Option<String>,
    room_id: Option<String>,
}

fn mimi_router(
    s: Arc<mimi::Mimi>,
    send_path: &str,
    recv_path: &str,
    ss: &SharedState,
) -> axum::Router<()> {
    async fn mimi_recv_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<mimi::Mimi>,
        room_id: Option<String>,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.recv_socket(socket, room_id).await {
            tracing::error!(?err, "mimi")
        }
    }

    async fn recv(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<mimi::Mimi>, SharedState)>,
        req: axum::extract::Query<MimiStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling mimi-streaming query");
        // It's tricky to set the headers of a websocket in javascript so we pass the token via the
        // query too.
        if state.0 .0.auth_recv() {
            let auth_id = match headers.get(ID_HEADER) {
                Some(v) => v.to_str().ok(),
                None => req.auth_id.as_deref(),
            };
            let valid_id = auth_id.is_some_and(|id| state.0 .1.config.authorized_ids.contains(id));
            if !valid_id {
                return Ok(StatusCode::UNAUTHORIZED.into_response());
            }
        }
        let room_id = match headers.get(ROOM_ID_HEADER) {
            Some(v) => v.to_str().ok().map(|v| v.to_string()),
            None => req.room_id.clone(),
        };
        let state = state.0 .0.clone();
        let upg = ws
            .write_buffer_size(0)
            .on_upgrade(move |v| mimi_recv_websocket(v, state, room_id, addr));
        Ok(upg)
    }

    async fn mimi_send_websocket(
        socket: axum::extract::ws::WebSocket,
        state: Arc<mimi::Mimi>,
        room_id: String,
        _addr: Option<String>,
    ) {
        if let Err(err) = state.send_socket(socket, room_id).await {
            tracing::error!(?err, "mimi")
        }
    }

    async fn send(
        ws: axum::extract::ws::WebSocketUpgrade,
        headers: axum::http::HeaderMap,
        state: axum::extract::State<(Arc<mimi::Mimi>, SharedState)>,
        req: axum::extract::Query<MimiStreamingQuery>,
    ) -> utils::AxumResult<axum::response::Response> {
        let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
        tracing::info!(addr, "handling mimi-streaming send query");
        let auth_id = match headers.get(ID_HEADER) {
            Some(v) => v.to_str().ok(),
            None => req.auth_id.as_deref(),
        };
        let valid_id = auth_id.is_some_and(|id| state.0 .1.config.authorized_ids.contains(id));
        if !valid_id {
            return Ok(StatusCode::UNAUTHORIZED.into_response());
        }
        let room_id = match headers.get(ROOM_ID_HEADER) {
            Some(v) => v.to_str().ok().map(|v| v.to_string()),
            None => req.room_id.clone(),
        };
        let room_id = match room_id {
            None => Err(anyhow::format_err!("no room_id"))?,
            Some(room_id) => room_id,
        };
        let state = state.0 .0;
        let upg = ws
            .write_buffer_size(0)
            .on_upgrade(move |v| mimi_send_websocket(v, state, room_id, addr));
        Ok(upg)
    }
    axum::Router::new()
        .route(send_path, axum::routing::get(send))
        .route(recv_path, axum::routing::get(recv))
        .with_state((s, ss.clone()))
}
