// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::{Context, Result};
use axum::extract::ws;
use std::sync::Arc;
use std::{path::Path, str::FromStr};

use crate::{stream_both, StandaloneArgs};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    cert_dir: String,
    pub static_dir: String,
    addr: String,
    port: u16,

    #[serde(flatten)]
    pub stream: stream_both::Config,
}

impl Config {
    pub fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let config = std::fs::read_to_string(p)?;
        let mut config: Self = serde_json::from_str(&config)?;
        config.static_dir = crate::utils::replace_env_vars(&config.static_dir);
        config.cert_dir = crate::utils::replace_env_vars(&config.cert_dir);
        config.stream.log_dir = crate::utils::replace_env_vars(&config.stream.log_dir);
        config.stream.text_tokenizer_file =
            crate::utils::replace_env_vars(&config.stream.text_tokenizer_file);
        config.stream.mimi_model_file =
            crate::utils::replace_env_vars(&config.stream.mimi_model_file);
        config.stream.lm_model_file = crate::utils::replace_env_vars(&config.stream.lm_model_file);
        Ok(config)
    }

    pub fn cert_file(&self, name: &str) -> std::path::PathBuf {
        let cert_dir = std::path::PathBuf::from(&self.cert_dir);
        cert_dir.join(name)
    }
}

pub(crate) fn device(cpu: bool) -> Result<candle::Device> {
    use candle::Device;
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

impl stream_both::AppStateInner {
    pub fn new(args: &StandaloneArgs, config: &stream_both::Config) -> Result<Self> {
        let device = device(args.cpu)?;
        let dtype = if device.is_cuda() { candle::DType::BF16 } else { candle::DType::F32 };
        let lm_model = moshi::lm::load_streaming(&config.lm_model_file, dtype, &device)?;
        let mimi_device = if config.use_cpu_for_mimi { &candle::Device::Cpu } else { &device };
        let mimi_model = moshi::mimi::load(
            &config.mimi_model_file,
            Some(config.mimi_num_codebooks),
            mimi_device,
        )?;
        let text_tokenizer =
            sentencepiece::SentencePieceProcessor::open(&config.text_tokenizer_file)?;
        // Warm-up code.
        {
            tracing::info!(?dtype, ?device, "warming up the model");
            let mut lm_model = lm_model.clone();
            let (_v, ys) =
                lm_model.forward(None, vec![None; config.mimi_num_codebooks], &().into())?;
            let mut lp = candle_transformers::generation::LogitsProcessor::new(123, None, None);
            let _ = lm_model.depformer_sample(&ys, None, &[], &mut lp)?;
            let mut mimi_model = mimi_model.clone();
            let config = mimi_model.config();
            let frame_length = (config.sample_rate / config.frame_rate).ceil() as usize;
            let fake_pcm =
                candle::Tensor::zeros((1, 1, frame_length), candle::DType::F32, mimi_device)?;
            let codes = mimi_model.encode_step(&fake_pcm.into(), &().into())?;
            let ys = mimi_model.decode_step(&codes, &().into())?;
            if ys.as_option().is_none() {
                anyhow::bail!("Expected Mimi to output some stuff, but nothing came out.");
            }
            device.synchronize()?;
            tracing::info!("model is ready to roll!");
        }
        Ok(Self { lm_model, mimi_model, device, config: config.clone(), text_tokenizer })
    }
}

async fn handle_socket(socket: ws::WebSocket, sm: stream_both::StreamingModel) {
    if let Err(err) = stream_both::handle_socket(socket, sm, None).await {
        tracing::error!(err = err.to_string(), "handle_socket")
    }
}

pub async fn stream_handler(
    ws: ws::WebSocketUpgrade,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    state: axum::extract::State<stream_both::AppState>,
    req: axum::extract::Query<stream_both::SessionConfigReq>,
) -> impl axum::response::IntoResponse {
    tracing::info!(?addr, "received connection");
    let sm = stream_both::StreamingModel::new(&state.0, req.0);
    ws.on_upgrade(move |v| handle_socket(v, sm))
}

pub async fn download_from_hub(config: &mut stream_both::Config) -> Result<()> {
    use hf_hub::api::tokio::Api;
    let api = Api::new()?;
    let repo = api.model(config.hf_repo.clone());
    let extract_filename = |path: &str| -> Result<String> {
        Path::new(path)
            .file_name()
            .and_then(|f| f.to_str())
            .map(String::from)
            .ok_or_else(|| anyhow::anyhow!("'{path}' has no file name"))
    };
    for file_path in
        [&mut config.lm_model_file, &mut config.mimi_model_file, &mut config.text_tokenizer_file]
            .iter_mut()
    {
        let filename = extract_filename(file_path)
            .with_context(|| format!("Failed to extract filename for '{file_path}'"))?;
        let downloaded_path = repo
            .get(&filename)
            .await
            .with_context(|| format!("Failed to download '{file_path}' file"))?;
        **file_path = downloaded_path
            .into_os_string()
            .into_string()
            .map_err(|_| anyhow::anyhow!("'{file_path}' path is not a valid string"))?;
    }
    Ok(())
}

pub async fn run(args: &StandaloneArgs, config: &Config) -> Result<()> {
    let cert_pem = config.cert_file("cert.pem");
    let key_pem = config.cert_file("key.pem");
    if !cert_pem.exists() || !key_pem.exists() {
        let rcgen::CertifiedKey { cert, key_pair } =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()])?;
        std::fs::write(&cert_pem, cert.pem())?;
        std::fs::write(&key_pem, key_pair.serialize_pem())?;
    }

    let tls_config =
        axum_server::tls_rustls::RustlsConfig::from_pem_file(cert_pem, key_pem).await?;
    let sock_addr = std::net::SocketAddr::from((
        std::net::IpAddr::from_str(config.addr.as_str())
            .unwrap_or(std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
        config.port,
    ));
    let state = Arc::new(stream_both::AppStateInner::new(args, &config.stream)?);
    tracing::info!("serving static dir {}", config.static_dir);
    let app = axum::Router::new()
        .route("/api/chat", axum::routing::get(stream_handler))
        .fallback_service(
            tower_http::services::ServeDir::new(&config.static_dir)
                .append_index_html_on_directories(true),
        )
        .layer(tower::ServiceBuilder::new().layer(tower_http::trace::TraceLayer::new_for_http()))
        .with_state(state);
    tracing::info!("standalone worker listening on https://{}", sock_addr);
    axum_server::bind_rustls(sock_addr, tls_config)
        .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
        .await?;
    Ok(())
}
