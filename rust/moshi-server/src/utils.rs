// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;

#[derive(Debug, PartialEq, Clone, serde::Deserialize, serde::Serialize)]
pub struct BuildInfo {
    build_timestamp: String,
    build_date: String,
    git_branch: String,
    git_timestamp: String,
    git_date: String,
    git_hash: String,
    git_describe: String,
    rustc_host_triple: String,
    rustc_version: String,
    cargo_target_triple: String,
}

impl BuildInfo {
    pub fn new() -> BuildInfo {
        BuildInfo {
            build_timestamp: String::from(env!("VERGEN_BUILD_TIMESTAMP")),
            build_date: String::from(env!("VERGEN_BUILD_DATE")),
            git_branch: String::from(env!("VERGEN_GIT_BRANCH")),
            git_timestamp: String::from(env!("VERGEN_GIT_COMMIT_TIMESTAMP")),
            git_date: String::from(env!("VERGEN_GIT_COMMIT_DATE")),
            git_hash: String::from(env!("VERGEN_GIT_SHA")),
            git_describe: String::from(env!("VERGEN_GIT_DESCRIBE")),
            rustc_host_triple: String::from(env!("VERGEN_RUSTC_HOST_TRIPLE")),
            rustc_version: String::from(env!("VERGEN_RUSTC_SEMVER")),
            cargo_target_triple: String::from(env!("VERGEN_CARGO_TARGET_TRIPLE")),
        }
    }
}

pub fn replace_env_vars(input: &str) -> String {
    let re = regex::Regex::new(r"\$([A-Za-z_][A-Za-z0-9_]*)").unwrap();
    re.replace_all(input, |caps: &regex::Captures| {
        let var_name = &caps[1];
        std::env::var(var_name).unwrap_or_else(|_| "".to_string())
    })
    .to_string()
}

pub fn resolve_or_download(input: &str) -> Result<String> {
    let path = match input.strip_prefix("hf://") {
        None => replace_env_vars(input),
        Some(path) => {
            let s: Vec<&str> = path.split('/').collect();
            if s.len() < 3 {
                anyhow::bail!("unexpected format for hf path {input}")
            }
            let repo = format!("{}/{}", s[0], s[1]);
            let file = s[2..].join("/");
            let api = hf_hub::api::sync::Api::new()?.model(repo);
            api.get(&file)?.to_string_lossy().to_string()
        }
    };
    Ok(path)
}

fn walk_toml(t: &mut toml::Value, f: &impl Fn(&mut String) -> Result<()>) -> Result<()> {
    match t {
        toml::Value::Table(t) => {
            for (_, t) in t.iter_mut() {
                walk_toml(t, f)?;
            }
        }
        toml::Value::Array(a) => {
            for t in a.iter_mut() {
                walk_toml(t, f)?
            }
        }
        toml::Value::String(s) => f(s)?,
        toml::Value::Integer(_)
        | toml::Value::Float(_)
        | toml::Value::Boolean(_)
        | toml::Value::Datetime(_) => {}
    }
    Ok(())
}

pub fn resolve_or_download_toml(t: &mut toml::Table) -> Result<()> {
    for (_, t) in t.iter_mut() {
        walk_toml(t, &|s: &mut String| -> Result<()> {
            *s = resolve_or_download(s)?;
            Ok(())
        })?;
    }
    Ok(())
}

pub struct WrapJson<T>(pub Result<T>);

impl<T: serde::Serialize> axum::response::IntoResponse for WrapJson<T> {
    fn into_response(self) -> axum::response::Response {
        match self.0 {
            Ok(v) => axum::Json(v).into_response(),
            Err(err) => {
                tracing::error!(?err, "returning internal server error 500");
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{err}")).into_response()
            }
        }
    }
}

pub struct AxumError(anyhow::Error);

impl axum::response::IntoResponse for AxumError {
    fn into_response(self) -> axum::response::Response {
        let err = self.0;
        tracing::error!(?err);
        (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{err:?}")).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AxumError {
    fn from(value: E) -> Self {
        Self(value.into())
    }
}

pub type AxumResult<R> = std::result::Result<R, AxumError>;

fn conv<T>(samples: &mut Vec<f32>, data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    use symphonia::core::audio::Signal;
    use symphonia::core::conv::FromSample;
    samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
}

pub fn pcm_decode(bytes: axum::body::Bytes) -> anyhow::Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::{AudioBufferRef, Signal};

    let source = std::io::Cursor::new(bytes);
    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(source), Default::default());
    let hint = symphonia::core::probe::Hint::new();
    let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
    let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .expect("no supported audio tracks");
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .expect("unsupported codec");
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut pcm_data = Vec::new();
    while let Ok(packet) = format.next_packet() {
        while !format.metadata().is_latest() {
            format.metadata().pop();
        }
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet)? {
            AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
            AudioBufferRef::U8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::U32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S8(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S16(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S24(data) => conv(&mut pcm_data, data),
            AudioBufferRef::S32(data) => conv(&mut pcm_data, data),
            AudioBufferRef::F64(data) => conv(&mut pcm_data, data),
        }
    }
    Ok((pcm_data, sample_rate))
}

pub fn spawn<F>(name: &'static str, future: F) -> tokio::task::JoinHandle<()>
where
    F: std::future::Future<Output = Result<()>> + Send + 'static,
{
    tokio::task::spawn(async move {
        match future.await {
            Ok(_) => tracing::info!(?name, "task completed successfully"),
            Err(err) => tracing::error!(?name, ?err, "task failed"),
        }
    })
}

pub fn spawn_blocking<F>(name: &'static str, f: F) -> tokio::task::JoinHandle<()>
where
    F: FnOnce() -> Result<()> + Send + 'static,
{
    tokio::task::spawn_blocking(move || match f() {
        Ok(_) => tracing::info!(?name, "task completed successfully"),
        Err(err) => tracing::error!(?name, ?err, "task failed"),
    })
}
