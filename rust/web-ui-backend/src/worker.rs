use anyhow::{Context, Result};
use axum::{extract::ws, response::IntoResponse};
use std::str::FromStr;
use std::sync::Arc;

use crate::{metrics::worker as metrics, redis_client, stream_both, utils, StandaloneArgs};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    redis_key: String,
    redis_server: String,
    addr: String,
    port: u16,
    shutdown_file: Option<std::path::PathBuf>,

    #[serde(flatten)]
    pub stream: stream_both::Config,
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = serde_json::from_str(&config)?;
        config.stream.log_dir = crate::utils::replace_env_vars(&config.stream.log_dir);
        config.stream.text_tokenizer_file =
            crate::utils::replace_env_vars(&config.stream.text_tokenizer_file);
        config.stream.encodec_model_file =
            crate::utils::replace_env_vars(&config.stream.encodec_model_file);
        config.stream.lm_model_file = crate::utils::replace_env_vars(&config.stream.lm_model_file);
        Ok(config)
    }
}

/// The state machine works as follows:
/// - A user query is accepted only if it provides an auth_id that matches the
///   current state.
/// - The state is moved to Running at the beginning of the query.
/// - The state is moved to Waiting(None) at the end of the query.
/// - When advertising, if the state is Waiting(None), a new auth-id is generated
///   and the state is updated.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Session {
    Waiting { advertised_auth_id: Option<u64> },
    Running,
}

pub struct AppState {
    stream_state: Arc<stream_both::AppStateInner>,
    redis_client: redis_client::RedisClient,
    instance_name: String,
    session: tokio::sync::Mutex<Session>,
    local_ip: std::net::IpAddr,
    port: u16,
}

impl AppState {
    fn adname(&self, auth_id: u64) -> String {
        // We convert the auth_id to a string as handling int greater than 2**53 in javascript
        // could yield unexpected results.
        crate::utils::AdName {
            instance_name: self.instance_name.to_string(),
            local_ip: self.local_ip,
            port: self.port,
            auth_id: auth_id.to_string(),
        }
        .to_string()
    }

    async fn advertised_name(&self) -> Option<String> {
        match *self.session.lock().await {
            Session::Waiting { advertised_auth_id: Some(auth_id) } => Some(self.adname(auth_id)),
            _ => None,
        }
    }
}

async fn handle_socket(
    mut socket: ws::WebSocket,
    state: Arc<AppState>,
    cfg: stream_both::SessionConfigReq,
    addr: Option<String>,
) {
    {
        let worker_auth_id = cfg.worker_auth_id.unwrap_or(0u64);
        let expected_state = Session::Waiting { advertised_auth_id: Some(worker_auth_id) };
        let mut session = state.session.lock().await;
        if *session != expected_state {
            tracing::error!(?session, worker_auth_id, "already in a concurrent session");
            let msg: Vec<u8> =
                [&[stream_both::MsgType::Error.to_u8()], b"already busy".as_slice()].concat();
            let msg = ws::Message::Binary(msg);
            let _ = socket.send(msg).await;
            metrics::CHAT_AUTH_ISSUE.inc();
            return;
        }
        *session = Session::Running;
    }
    metrics::CHAT.inc();
    let timer = metrics::CHAT_DURATION.start_timer();
    let sm = stream_both::StreamingModel::new(&state.stream_state, cfg);
    if let Err(err) = stream_both::handle_socket(socket, sm, addr).await {
        tracing::error!(err = err.to_string(), "handle_socket")
    }
    timer.stop_and_record();
    // TODO: Maybe we should use RAII to automatically revert from the Running state to
    // Waiting.
    let mut session = state.session.lock().await;
    *session = Session::Waiting { advertised_auth_id: None };
}

pub async fn stream_handler(
    ws: ws::WebSocketUpgrade,
    headers: http::HeaderMap,
    state: axum::extract::State<Arc<AppState>>,
    req: axum::extract::Query<stream_both::SessionConfigReq>,
) -> impl IntoResponse {
    let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok().map(|v| v.to_string()));
    tracing::info!(addr, "received connection");
    let state = state.0.clone();
    let session_cfg = req.0;
    ws.write_buffer_size(0).on_upgrade(move |v| handle_socket(v, state, session_cfg, addr))
}

pub async fn build_info() -> impl IntoResponse {
    let build_info = utils::BuildInfo::new();
    utils::WrapJson(Ok(build_info)).into_response()
}

pub async fn metrics() -> impl IntoResponse {
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

// The worker setup uses http rather than https as decoding the TLS layer is done in nginx.
pub async fn run(args: &StandaloneArgs, config: &Config) -> Result<()> {
    use rand::{RngCore, SeedableRng};

    let sock_addr = std::net::SocketAddr::from((
        std::net::IpAddr::from_str(config.addr.as_str())
            .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
        config.port,
    ));
    if let Some(shutdown_file) = config.shutdown_file.as_ref() {
        if shutdown_file.try_exists().unwrap_or(false) {
            match std::fs::remove_file(shutdown_file) {
                Ok(()) => tracing::info!(?shutdown_file, "removed shutdown-file"),
                Err(err) => tracing::error!(?err, ?shutdown_file, "cannot remove shutdown-file"),
            }
        }
    }
    let stream_state = Arc::new(stream_both::AppStateInner::new(args, &config.stream)?);
    let redis_client = redis_client::RedisClient::new(&config.redis_server, &config.redis_key)?;
    tracing::info!(cmap = ?redis_client.client_map(), client_ip = ?redis_client.client_ip(), "started redis client");
    let mut rng = rand_chacha::ChaChaRng::from_entropy();
    let auth_id = rng.next_u64();

    let local_ip = redis_client.client_ip().context("no client-ip extracted from redis")?;

    let state = AppState {
        stream_state,
        redis_client,
        instance_name: config.stream.instance_name.to_string(),
        session: tokio::sync::Mutex::new(Session::Waiting { advertised_auth_id: Some(auth_id) }),
        local_ip: std::net::IpAddr::V4(local_ip),
        port: config.port,
    };
    let state = Arc::new(state);

    let app = axum::Router::new()
        .route("/api/chat", axum::routing::get(stream_handler))
        .route("/api/build_info", axum::routing::get(build_info))
        .route("/metrics", axum::routing::get(metrics))
        .layer(tower::ServiceBuilder::new().layer(tower_http::trace::TraceLayer::new_for_http()))
        .with_state(state.clone());
    tracing::info!(?local_ip, "worker listening on http://{}", sock_addr);
    let handle = axum_server::Handle::new();
    let handle_loop = handle.clone();
    let shutdown_file = config.shutdown_file.clone();
    tokio::spawn(async move {
        if let Some(advertised_name) = state.advertised_name().await {
            if let Err(err) = state.redis_client.advertise_worker(&advertised_name, 2.0, false) {
                tracing::error!(?err, "error advertising worker")
            }
        }
        'outer: loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            let already_advertised = match state.advertised_name().await {
                None => false,
                Some(advertised_name) => {
                    let should_shutdown =
                        shutdown_file.as_ref().map_or(false, |v| v.try_exists().unwrap_or(false));
                    if should_shutdown {
                        if let Err(err) = state.redis_client.unadvertise(&advertised_name) {
                            tracing::error!(?err, "error unadvertising worker");
                            // Be conservative and assume we're still advertised on redis errors.
                            // ZREM should not fail but return 0 if the key or member don't exist.
                            true
                        } else {
                            false
                        }
                    } else {
                        match state.redis_client.advertise_worker(&advertised_name, 2.0, true) {
                            Ok(upd) => upd,
                            Err(err) => {
                                tracing::error!(?err, "error advertising worker");
                                false
                            }
                        }
                    }
                }
            };
            if !already_advertised {
                // The key has been removed by the dispatcher, so we wait for a client to
                // arrive with the proper token. If there is nothing after a couple seconds
                // we assume that the client won't join and update the required token.
                tokio::time::sleep(tokio::time::Duration::from_secs(4)).await;

                loop {
                    // Wait for no session to be active anymore.
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    let mut session = state.session.lock().await;
                    if *session == Session::Running {
                        continue;
                    }
                    let should_shutdown =
                        shutdown_file.as_ref().map_or(false, |v| v.try_exists().unwrap_or(false));
                    // At this point the worker has not been advertised again and no
                    // upcoming session is expected so it's a good time to shutdown.
                    if should_shutdown {
                        tracing::info!("shutdown after grace period");
                        if let Some(shutdown_file) = shutdown_file.as_ref() {
                            if let Err(err) = std::fs::remove_file(shutdown_file) {
                                tracing::error!(?err, "removing shutdown-file");
                            }
                        }
                        break 'outer;
                    }
                    let auth_id = rng.next_u64();
                    let advertised_name = state.adname(auth_id);
                    *session = Session::Waiting { advertised_auth_id: Some(auth_id) };
                    tracing::info!(?advertised_name, "advertise new token");
                    if let Err(err) =
                        state.redis_client.advertise_worker(&advertised_name, 2.0, false)
                    {
                        tracing::error!(?err, "error advertising worker")
                    }
                    break;
                }
            }
        }
        handle_loop.shutdown();
        tracing::info!("exiting the control loop");
    });
    axum_server::bind(sock_addr)
        .acceptor(crate::NoDelayAcceptor)
        .handle(handle)
        .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .await?;
    Ok(())
}
