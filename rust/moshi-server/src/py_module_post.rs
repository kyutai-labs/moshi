// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::metrics::py_post as metrics;
use crate::py_module::{toml_to_py, VerbosePyErr};
use anyhow::{Context, Result};
use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3_ffi::c_str;
use tokio::task;

type Out = (Vec<f32>, Vec<usize>);
struct ModelQuery {
    pcm: Vec<f32>,
    out_tx: tokio::sync::oneshot::Sender<Out>,
}

pub struct Inner {
    app: PyObject,
    in_rx: std::sync::mpsc::Receiver<ModelQuery>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum OutMsg {
    Voice { embeddings: Vec<f32>, shape: Vec<usize> },
}

impl Inner {
    fn start_model_loop(self) -> Result<()> {
        // use numpy::{PyArrayMethods, ToPyArray};
        let model_loop: task::JoinHandle<Result<()>> = task::spawn_blocking(move || {
            while let Ok(req) = self.in_rx.recv() {
                if let Err(err) = self.handle_query(req) {
                    tracing::error!(?err, "failed to handle query");
                }
            }
            Ok(())
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

    fn handle_query(&self, req: ModelQuery) -> Result<()> {
        let start_time = std::time::Instant::now();
        let emb = Python::with_gil(|py| -> Result<_> {
            let pcm = numpy::PyArray1::from_vec(py, req.pcm);
            let emb = self.app.call_method1(py, "run_one", (pcm,)).map_err(VerbosePyErr::from)?;
            let emb = match emb.downcast_bound::<numpy::PyArrayDyn<f32>>(py) {
                Ok(emb) => emb,
                Err(_) => {
                    anyhow::bail!("failed to downcast to PyArrayDyn<f32>")
                }
            };
            let shape = emb.shape().to_vec();
            tracing::info!(?shape, "generated embeddings");
            Ok((emb.to_vec()?, shape))
        })?;
        let elapsed = start_time.elapsed().as_secs_f64();
        metrics::MODEL_DURATION.observe(elapsed);
        if let Err(err) = req.out_tx.send(emb) {
            anyhow::bail!("failed to send response: {err:?}");
        }
        Ok(())
    }
}

pub struct M {
    in_tx: std::sync::mpsc::Sender<ModelQuery>,
}

impl M {
    pub fn new(config: crate::PyPostConfig) -> Result<Self> {
        crate::py_module::init()?;
        let (script, script_name) = match &config.script {
            None => {
                let script_name = std::ffi::CString::new("voice.py")?;
                let script = std::ffi::CString::new(crate::VOICE_PY)?;
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
                    .call1((py_config,))
                    .map_err(VerbosePyErr::from)?;
            Ok(app.unbind())
        })?;
        let (in_tx, in_rx) = std::sync::mpsc::channel();
        let inner = Inner { app, in_rx };
        inner.start_model_loop()?;
        Ok(Self { in_tx })
    }

    pub async fn run_one(&self, data: axum::body::Bytes) -> Result<axum::body::Bytes> {
        use serde::Serialize;

        metrics::CONNECT.inc();
        let (out_tx, out_rx) = tokio::sync::oneshot::channel::<Out>();
        let pcm = task::spawn_blocking(move || -> Result<Vec<f32>> {
            let (pcm, sample_rate) = crate::utils::pcm_decode(data)?;
            let mut pcm = if sample_rate == 24000 {
                pcm
            } else {
                kaudio::resample(&pcm, sample_rate as usize, 24000)?
            };
            pcm.resize(240000, 0.0);
            Ok(pcm)
        });
        let pcm = pcm.await??;
        let query = ModelQuery { pcm, out_tx };
        self.in_tx.send(query)?;
        let (embeddings, shape) = out_rx.await?;
        let msg = OutMsg::Voice { embeddings, shape };
        let mut bytes = vec![];
        msg.serialize(
            &mut rmp_serde::Serializer::new(&mut bytes).with_human_readable().with_struct_map(),
        )?;

        Ok(bytes.into())
    }
}
