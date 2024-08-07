// The dispatcher is responsible for allocating user to gpu workers.
// Redis is used to discover workers that are ready to process data.
use crate::waiting_queue::SessionId;
use crate::{metrics::dispatcher as metrics, utils, DispatcherArgs};
use anyhow::Result;
use axum::response::IntoResponse;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    static_dir: String,
    log_dir: String,
    instance_name: String,
    addr: String,
    port: u16,
    max_connections_per_queue_id: HashMap<String, usize>,
    redis_key: String,
    redis_server: String,
    max_queue_size: usize,
    max_recent_users: usize,
    secret: String,
    drop_stale_clients_after_s: f64,
    worker_addr: Option<String>,
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = serde_json::from_str(&config)?;
        config.static_dir = crate::utils::replace_env_vars(&config.static_dir);
        config.log_dir = crate::utils::replace_env_vars(&config.log_dir);
        Ok(config)
    }

    pub fn log_dir(&self) -> &str {
        self.log_dir.as_str()
    }

    pub fn instance_name(&self) -> &str {
        self.instance_name.as_str()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct AddUser {
    session_id: u64,
    session_auth_id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SecretReq {
    secret: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AddUserReq {
    queue_id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UserFeedbackReq {
    session_id: u64,
    session_auth_id: String,
    worker_auth_id: Option<String>,
    feedback: i32,
    timestamp: f64,
    email: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckUserReq {
    session_id: u64,
    session_auth_id: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckUserResponse {
    session_id: u64,
    status: &'static str,
    worker_auth_id: Option<String>,
    worker_addr: Option<String>,
    current_position: String,
}

struct Worker {
    last_match: std::time::SystemTime,
    last_session_id: SessionId,
}

pub struct AppStateInner {
    pub config: Config,
    redis_client: crate::redis_client::RedisClient,
    user_queue: Mutex<crate::waiting_queue::Queue>,
    feedbacks: Mutex<HashMap<u64, UserFeedbackReq>>,
    rnd_secret: u64,
    config_file: String,
    workers: Mutex<HashMap<String, Worker>>,
}

pub type AppState = Arc<AppStateInner>;

fn auth_token(id: u64, secret: u64) -> String {
    use base64ct::Encoding;
    use sha3::Digest;

    let mut hasher = sha3::Sha3_256::new();
    hasher.update(id.to_le_bytes());
    hasher.update(secret.to_le_bytes());
    let hash = hasher.finalize();
    base64ct::Base64::encode_string(&hash)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct WorkerDetails {
    instance_name: String,
    auth_id: Option<String>,
    local_ip: Option<std::net::IpAddr>,
    port: u16,
    available: bool,
    available_until_s: Option<f64>,
    last_match_s: Option<f64>,
    last_session_id: Option<SessionId>,
}

impl AppStateInner {
    pub fn new(_args: &DispatcherArgs, config: &Config, config_file: &str) -> Result<Self> {
        use rand::{RngCore, SeedableRng};

        let mut rng = rand_chacha::ChaChaRng::from_entropy();
        let redis_key = config.redis_key.to_string();
        let redis_server = config.redis_server.to_string();
        tracing::info!(redis_server, redis_key, "starting dispatcher");
        let redis_client = crate::redis_client::RedisClient::new(&redis_server, &redis_key)?;
        tracing::info!(cmap = ?redis_client.client_map(), client_ip = ?redis_client.client_ip(), "started redis client");
        Ok(Self {
            config: config.clone(),
            redis_client,
            user_queue: Mutex::new(crate::waiting_queue::Queue::new(
                config.max_queue_size,
                config.max_recent_users,
                config.max_connections_per_queue_id.clone(),
                config.drop_stale_clients_after_s,
            )),
            rnd_secret: rng.next_u64(),
            config_file: config_file.to_string(),
            workers: Mutex::new(HashMap::new()),
            feedbacks: Mutex::new(HashMap::new()),
        })
    }

    fn list_workers(&self) -> Result<Vec<WorkerDetails>> {
        use redis::Commands;
        use std::str::FromStr;

        let mut con = self.redis_client.get_connection()?;
        let workers_str: Vec<(String, f64)> =
            con.zrange_withscores(&self.config.redis_key, 0, -1)?;
        let now = crate::redis_client::time(&mut con)?;
        let mut workers = Vec::with_capacity(workers_str.len());
        let mut already_seen = std::collections::HashSet::new();
        // The score represents the time until which the worker can be used.
        let self_workers = self.workers.lock().unwrap();
        for (w, available_until_s) in workers_str {
            if available_until_s < now {
                continue;
            }
            match utils::AdName::from_str(&w) {
                Ok(w) => {
                    already_seen.insert(w.instance_name.to_string());
                    let (last_match_s, last_session_id) = match self_workers.get(&w.instance_name) {
                        None => (None, None),
                        Some(w) => {
                            (Some(crate::utils::duration_s(w.last_match)), Some(w.last_session_id))
                        }
                    };
                    workers.push(WorkerDetails {
                        instance_name: w.instance_name,
                        auth_id: Some(w.auth_id),
                        local_ip: Some(w.local_ip),
                        port: w.port,
                        available: true,
                        available_until_s: Some(available_until_s),
                        last_match_s,
                        last_session_id,
                    });
                }
                Err(err) => tracing::error!(?err, ?w, "cannot parse worker name"),
            }
        }
        for (worker, w) in self_workers.iter() {
            if already_seen.contains(worker.as_str()) {
                continue;
            }
            workers.push(WorkerDetails {
                instance_name: worker.to_string(),
                auth_id: None,
                local_ip: None,
                port: 0,
                available: false,
                available_until_s: None,
                last_match_s: Some(crate::utils::duration_s(w.last_match)),
                last_session_id: Some(w.last_session_id),
            })
        }
        Ok(workers)
    }

    fn list_users(&self) -> Vec<crate::waiting_queue::UserOut> {
        self.user_queue.lock().unwrap().users(true)
    }

    fn list_queue_ids(&self) -> Vec<(String, usize)> {
        self.user_queue.lock().unwrap().queue_ids()
    }

    fn reload_config(&self) -> Result<()> {
        let config = Config::load(&self.config_file)?;
        self.user_queue.lock().unwrap().set_config(
            config.max_queue_size,
            config.max_recent_users,
            config.max_connections_per_queue_id,
            config.drop_stale_clients_after_s,
        );
        Ok(())
    }

    fn clear_queue(&self) -> Result<()> {
        self.user_queue.lock().unwrap().clear();
        Ok(())
    }

    fn add_user(&self, queue_id: String, addr: Option<&str>) -> Result<AddUser> {
        metrics::ADD_USER.inc();
        let session_id = self.user_queue.lock().unwrap().add_user(queue_id, addr)?.to_u64();
        let session_auth_id = auth_token(session_id, self.rnd_secret);
        Ok(AddUser { session_id, session_auth_id })
    }

    fn user_feedback(&self, req: UserFeedbackReq) -> Result<()> {
        metrics::USER_FEEDBACK.inc();
        let expected_session_auth_id = auth_token(req.session_id, self.rnd_secret);
        if req.session_auth_id != expected_session_auth_id {
            anyhow::bail!(
                "incorrect session-auth-id {}\nsession-id: '{}'",
                req.session_auth_id,
                req.session_id
            )
        }
        let mut feedbacks = self.feedbacks.lock().unwrap();
        feedbacks.insert(req.session_id, req);
        Ok(())
    }

    fn flush_feedbacks(&self) -> Result<()> {
        let mut feedbacks = self.feedbacks.lock().unwrap();
        if !feedbacks.is_empty() {
            let since_epoch = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?;
            let (secs, us) = (since_epoch.as_secs(), since_epoch.subsec_micros());
            let log_dir = &self.config.log_dir;
            let json_filename = format!("{log_dir}/{}-{secs}-{us}.json", self.config.instance_name);
            let json_content = serde_json::to_string_pretty(&*feedbacks)?;
            std::fs::write(json_filename, json_content)?;
            feedbacks.clear();
        }
        Ok(())
    }

    fn check_user(&self, req: CheckUserReq) -> Result<CheckUserResponse> {
        use std::str::FromStr;

        metrics::CHECK_USER.inc();
        let expected_session_auth_id = auth_token(req.session_id, self.rnd_secret);
        if req.session_auth_id != expected_session_auth_id {
            anyhow::bail!(
                "incorrect session-auth-id {}\nsession-id: '{}'",
                req.session_auth_id,
                req.session_id
            )
        }
        let session_id = SessionId::from_u64(req.session_id);
        // The current position can only go down if there are other threads running. No
        // session can be inserted before this one.
        let current_position = self.user_queue.lock().unwrap().refresh_user(session_id)?;
        let worker = match current_position {
            // pos is 0 if the client is at the top of the queue.
            crate::waiting_queue::QueuePosition::Exact(pos) => {
                self.redis_client.pop_worker_if_at_least(pos + 1)?
            }
            crate::waiting_queue::QueuePosition::GreaterThan(_) => None,
        };
        let (status, worker_addr, worker_auth_id) = match worker {
            Some(worker) => match utils::AdName::from_str(&worker) {
                Ok(ad_name) => {
                    tracing::info!(?worker, ?session_id, ?ad_name, "match");
                    self.user_queue
                        .lock()
                        .unwrap()
                        .remove_on_match(session_id, &ad_name.instance_name);
                    let worker_addr = match ad_name.local_ip {
                        std::net::IpAddr::V4(addr) => {
                            let octets = addr.octets();
                            match self.config.worker_addr.as_ref() {
                                None => format!("ws-{}-{}.ws.moshi.chat", octets[2], octets[3]),
                                Some(pat) => pat
                                    .replace("$ip.0", &octets[0].to_string())
                                    .replace("$ip.1", &octets[1].to_string())
                                    .replace("$ip.2", &octets[2].to_string())
                                    .replace("$ip.3", &octets[3].to_string())
                                    .replace("$port", &ad_name.port.to_string()),
                            }
                        }
                        std::net::IpAddr::V6(_) => "ipv6.moshi.chat".to_string(),
                    };
                    self.workers.lock().unwrap().insert(
                        ad_name.instance_name,
                        Worker {
                            last_match: std::time::SystemTime::now(),
                            last_session_id: session_id,
                        },
                    );
                    ("ready", Some(worker_addr), Some(ad_name.auth_id))
                }
                Err(err) => {
                    tracing::error!(?err, ?worker, "error parsing ad-name");
                    ("wait", None, None)
                }
            },
            None => ("wait", None, None),
        };
        let current_position = match current_position {
            crate::waiting_queue::QueuePosition::Exact(pos) => pos.to_string(),
            crate::waiting_queue::QueuePosition::GreaterThan(pos) => format!("> {pos}"),
        };
        Ok(CheckUserResponse {
            session_id: req.session_id,
            worker_auth_id,
            status,
            worker_addr,
            current_position,
        })
    }
}

pub async fn list_workers(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    req: axum::extract::Query<SecretReq>,
) -> impl IntoResponse {
    if req.0.secret != state.config.secret {
        return (axum::http::StatusCode::FORBIDDEN, "auth failure").into_response();
    }
    utils::WrapBincode(state.list_workers()).into_response()
}

pub async fn list_users(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    req: axum::extract::Query<SecretReq>,
) -> impl IntoResponse {
    if req.0.secret != state.config.secret {
        return (axum::http::StatusCode::FORBIDDEN, "auth failure").into_response();
    }
    utils::WrapBincode(Ok(state.list_users())).into_response()
}

pub async fn list_queue_ids(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    req: axum::extract::Query<SecretReq>,
) -> impl IntoResponse {
    if req.0.secret != state.config.secret {
        return (axum::http::StatusCode::FORBIDDEN, "auth failure").into_response();
    }
    utils::WrapBincode(Ok(state.list_queue_ids())).into_response()
}

pub async fn clear_queue(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    req: axum::extract::Query<SecretReq>,
) -> impl IntoResponse {
    if req.0.secret != state.config.secret {
        return (axum::http::StatusCode::FORBIDDEN, "auth failure").into_response();
    }
    utils::WrapBincode(state.clear_queue()).into_response()
}

pub async fn reload_config(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    req: axum::extract::Query<SecretReq>,
) -> impl IntoResponse {
    if req.0.secret != state.config.secret {
        return (axum::http::StatusCode::FORBIDDEN, "auth failure").into_response();
    }
    utils::WrapBincode(state.reload_config()).into_response()
}

pub async fn add_user(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    headers: http::HeaderMap,
    req: axum::extract::Query<AddUserReq>,
) -> impl IntoResponse {
    let addr = headers.get("X-Real-IP").and_then(|v| v.to_str().ok());
    utils::WrapJson(state.add_user(req.0.queue_id, addr)).into_response()
}

pub async fn check_user(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    req: axum::extract::Query<CheckUserReq>,
) -> impl IntoResponse {
    utils::WrapJson(state.check_user(req.0)).into_response()
}

pub async fn user_feedback(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<AppState>,
    req: axum::extract::Query<UserFeedbackReq>,
) -> impl IntoResponse {
    utils::WrapJson(state.user_feedback(req.0)).into_response()
}

pub async fn build_info(
    axum::extract::ConnectInfo(_addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    _state: axum::extract::State<AppState>,
    _req: axum::extract::Query<()>,
) -> impl IntoResponse {
    let build_info = utils::BuildInfo::new();
    utils::WrapJson(Ok(build_info)).into_response()
}

pub async fn metrics(
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

pub async fn run(args: &DispatcherArgs, config: &Config, config_file: &str) -> Result<()> {
    use std::str::FromStr;

    let sock_addr = std::net::SocketAddr::from((
        std::net::IpAddr::from_str(config.addr.as_str())
            .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
        config.port,
    ));
    let state = Arc::new(AppStateInner::new(args, config, config_file)?);
    tokio::spawn({
        let state = state.clone();
        async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs_f32(1.414)).await;
                state.user_queue.lock().unwrap().remove_stale_users();
                match state.redis_client.remove_stale_workers() {
                    Err(err) => tracing::error!("redis err {err:?}"),
                    Ok(available_workers) => {
                        metrics::AVAILABLE_WORKERS.set(available_workers as f64)
                    }
                }
            }
        }
    });
    tokio::spawn({
        let state = state.clone();
        async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(300)).await;
                if let Err(err) = state.flush_feedbacks() {
                    tracing::error!("redis err {err:?}")
                }
            }
        }
    });
    let app = axum::Router::new()
        // The admin endpoints are protected by the secret from the config but we should also
        // configure nginx to only allow using them from the vpn/...
        .route("/api/admin/workers", axum::routing::get(list_workers))
        .route("/api/admin/users", axum::routing::get(list_users))
        .route("/api/admin/queue_ids", axum::routing::get(list_queue_ids))
        .route("/api/admin/clear_queue", axum::routing::get(clear_queue))
        .route("/api/admin/reload_config", axum::routing::get(reload_config))
        .route("/api/add_user", axum::routing::get(add_user))
        .route("/api/check_user", axum::routing::get(check_user))
        .route("/api/user_feedback", axum::routing::get(user_feedback))
        .route("/metrics", axum::routing::get(metrics))
        .route("/api/build_info", axum::routing::get(build_info))
        .fallback_service(
            tower_http::services::ServeDir::new(&state.config.static_dir)
                .append_index_html_on_directories(true),
        )
        .layer(tower::ServiceBuilder::new().layer(tower_http::trace::TraceLayer::new_for_http()))
        .with_state(state);
    tracing::info!("dispatcher listening on http://{}", sock_addr);
    axum_server::bind(sock_addr)
        .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .await?;
    Ok(())
}
