// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

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

pub struct WrapJson<T>(pub anyhow::Result<T>);

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

pub fn replace_env_vars(input: &str) -> String {
    let re = regex::Regex::new(r"\$([A-Za-z_][A-Za-z0-9_]*)").unwrap();
    re.replace_all(input, |caps: &regex::Captures| {
        let var_name = &caps[1];
        std::env::var(var_name).unwrap_or_else(|_| "".to_string())
    })
    .to_string()
}

pub struct WrapBincode<T>(pub anyhow::Result<T>);

impl<T: serde::Serialize> axum::response::IntoResponse for WrapBincode<T> {
    fn into_response(self) -> axum::response::Response {
        match self.0.and_then(|v| Ok(bincode::serialize(&v)?)) {
            Ok(v) => (axum::http::StatusCode::OK, v).into_response(),
            Err(err) => {
                tracing::error!(?err, "returning internal server error 500");
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("{err}")).into_response()
            }
        }
    }
}
