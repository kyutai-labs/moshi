// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use pyo3::prelude::*;

use ::moshi as mm;
use mm::{candle, candle_nn, conv, encodec, seanet, transformer};

trait PyRes<R> {
    #[allow(unused)]
    fn w(self) -> PyResult<R>;
    fn w_f<P: AsRef<std::path::Path>>(self, p: P) -> PyResult<R>;
}

impl<R, E: Into<anyhow::Error>> PyRes<R> for Result<R, E> {
    fn w(self) -> PyResult<R> {
        self.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.into().to_string()))
    }
    fn w_f<P: AsRef<std::path::Path>>(self, p: P) -> PyResult<R> {
        self.map_err(|e| {
            let e = e.into().to_string();
            let msg = format!("{:?}: {e}", p.as_ref());
            pyo3::exceptions::PyValueError::new_err(msg)
        })
    }
}

#[macro_export]
macro_rules! py_bail {
    ($msg:literal $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($msg)))
    };
    ($err:expr $(,)?) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($err)))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err(pyo3::exceptions::PyValueError::new_err(format!($fmt, $($arg)*)))
    };
}

fn encodec_cfg(max_seq_len: Option<usize>) -> encodec::Config {
    let seanet_cfg = seanet::Config {
        dimension: 512,
        channels: 1,
        causal: true,
        n_filters: 64,
        n_residual_layers: 1,
        activation: candle_nn::Activation::Elu(1.),
        compress: 2,
        dilation_base: 2,
        disable_norm_outer_blocks: 0,
        final_activation: None,
        kernel_size: 7,
        residual_kernel_size: 3,
        last_kernel_size: 3,
        lstm: 0,
        norm: conv::Norm::WeightNorm,
        pad_mode: conv::PadMode::Constant,
        ratios: vec![8, 6, 5, 4],
        true_skip: true,
    };
    let transformer_cfg = transformer::Config {
        d_model: seanet_cfg.dimension,
        num_heads: 8,
        num_layers: 8,
        causal: true,
        norm_first: true,
        bias_ff: false,
        bias_attn: false,
        layer_scale: Some(0.01),
        context: 250,
        conv_kernel_size: 5,
        use_conv_bias: true,
        use_conv_block: false,
        max_period: 10000,
        positional_embedding: transformer::PositionalEmbedding::Rope,
        gating: None,
        norm: mm::NormType::LayerNorm,

        dim_feedforward: 2048,
        kv_repeat: 1,
        conv_layout: true, // see builders.py
        cross_attention: None,
        max_seq_len: max_seq_len.unwrap_or(8192), // the transformer works at 25hz so this is ~5 mins.
    };
    encodec::Config {
        channels: 1,
        sample_rate: 24_000.,
        frame_rate: 12.5,
        renormalize: true,
        resample_method: encodec::ResampleMethod::Conv,
        seanet: seanet_cfg,
        transformer: transformer_cfg,
        quantizer_n_q: 8,
        quantizer_bins: 2048,
        quantizer_dim: 256,
    }
}

#[pyclass]
struct Tokenizer {
    encodec: encodec::Encodec,
    device: candle::Device,
    dtype: candle::DType,
}

#[pymethods]
impl Tokenizer {
    #[pyo3(signature = (path, *, dtype="f32", max_seq_len=None))]
    #[new]
    fn new(path: std::path::PathBuf, dtype: &str, max_seq_len: Option<usize>) -> PyResult<Self> {
        let device = candle::Device::Cpu;
        let dtype = match dtype {
            "f32" => candle::DType::F32,
            "f16" => candle::DType::F16,
            "bf16" => candle::DType::BF16,
            dtype => py_bail!("unsupported dtype '{dtype}'"),
        };
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[path], dtype, &device).w()? };
        let cfg = encodec_cfg(max_seq_len);
        let encodec = encodec::Encodec::new(cfg, vb).w()?;
        Ok(Self { encodec, device, dtype })
    }

    fn encode(&mut self, pcm_data: numpy::PyReadonlyArray3<f32>) -> PyResult<PyObject> {
        let py = pcm_data.py();
        let pcm_data = pcm_data.as_array();
        let pcm_shape = pcm_data.shape().to_vec();
        let pcm_data = match pcm_data.to_slice() {
            None => py_bail!("input data is not contiguous"),
            Some(data) => data,
        };
        let codes = py
            .allow_threads(|| {
                let pcm_data = candle::Tensor::from_slice(pcm_data, pcm_shape, &self.device)?
                    .to_dtype(self.dtype)?;
                let codes = self.encodec.encode(&pcm_data)?;
                codes.to_vec3::<u32>()
            })
            .w()?;
        let codes = numpy::PyArray3::from_vec3_bound(py, &codes)?;
        Ok(codes.into_py(py))
    }

    fn encode_step(&mut self, pcm_data: numpy::PyReadonlyArray3<f32>) -> PyResult<PyObject> {
        let py = pcm_data.py();
        let pcm_data = pcm_data.as_array();
        let pcm_shape = pcm_data.shape().to_vec();
        let pcm_data = match pcm_data.to_slice() {
            None => py_bail!("input data is not contiguous"),
            Some(data) => data,
        };
        let codes = py
            .allow_threads(|| {
                let pcm_data = candle::Tensor::from_slice(pcm_data, pcm_shape, &self.device)?
                    .to_dtype(self.dtype)?;
                let codes = self.encodec.encode_step(&pcm_data.into())?;
                match codes.as_option() {
                    Some(codes) => Ok::<_, candle::Error>(Some(codes.to_vec3::<u32>()?)),
                    None => Ok(None),
                }
            })
            .w()?;
        match codes {
            Some(codes) => {
                let codes = numpy::PyArray3::from_vec3_bound(py, &codes)?;
                Ok(codes.into_py(py))
            }
            None => Ok(py.None()),
        }
    }

    fn decode(&mut self, codes: numpy::PyReadonlyArray3<u32>, py: Python) -> PyResult<PyObject> {
        let codes = codes.as_array();
        let codes_shape = codes.shape().to_vec();
        let codes = match codes.to_slice() {
            None => py_bail!("input data is not contiguous"),
            Some(data) => data,
        };
        let pcm = py
            .allow_threads(|| {
                let codes = candle::Tensor::from_slice(codes, codes_shape, &self.device)?;
                let pcm = self.encodec.decode(&codes)?.to_dtype(candle::DType::F32)?;
                pcm.to_vec3::<f32>()
            })
            .w()?;
        let pcm = numpy::PyArray3::from_vec3_bound(py, &pcm)?;
        Ok(pcm.into_py(py))
    }

    fn decode_step(
        &mut self,
        codes: numpy::PyReadonlyArray3<u32>,
        py: Python,
    ) -> PyResult<PyObject> {
        let codes = codes.as_array();
        let codes_shape = codes.shape().to_vec();
        let codes = match codes.to_slice() {
            None => py_bail!("input data is not contiguous"),
            Some(data) => data,
        };
        let pcm = py
            .allow_threads(|| {
                let codes = candle::Tensor::from_slice(codes, codes_shape, &self.device)?;
                let pcm = self.encodec.decode_step(&codes.into())?;
                match pcm.as_option() {
                    Some(pcm) => {
                        let pcm = pcm.to_dtype(candle::DType::F32)?;
                        Ok::<_, candle::Error>(Some(pcm.to_vec3::<f32>()?))
                    }
                    None => Ok(None),
                }
            })
            .w()?;
        match pcm {
            Some(pcm) => {
                let pcm = numpy::PyArray3::from_vec3_bound(py, &pcm)?;
                Ok(pcm.into_py(py))
            }
            None => Ok(py.None()),
        }
    }

    fn reset(&mut self) {
        self.encodec.reset_state()
    }
}

#[pyclass]
struct StreamTokenizer {
    #[allow(unused)]
    dtype: candle::DType,
    encoder_rx: std::sync::mpsc::Receiver<Vec<Vec<u32>>>,
    encoder_tx: std::sync::mpsc::Sender<Vec<f32>>,
    decoder_rx: std::sync::mpsc::Receiver<Vec<f32>>,
    decoder_tx: std::sync::mpsc::Sender<Vec<Vec<u32>>>,
}

#[pymethods]
impl StreamTokenizer {
    #[pyo3(signature = (path, *, dtype="f32", max_seq_len=None))]
    #[new]
    fn new(path: std::path::PathBuf, dtype: &str, max_seq_len: Option<usize>) -> PyResult<Self> {
        let device = candle::Device::Cpu;
        let dtype = match dtype {
            "f32" => candle::DType::F32,
            "f16" => candle::DType::F16,
            "bf16" => candle::DType::BF16,
            dtype => py_bail!("unsupported dtype '{dtype}'"),
        };
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[path], dtype, &device).w()? };
        let cfg = encodec_cfg(max_seq_len);
        let mut e_encodec = encodec::Encodec::new(cfg, vb).w()?;
        let mut d_encodec = e_encodec.clone();
        let (encoder_tx, e_rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let (decoder_tx, d_rx) = std::sync::mpsc::channel::<Vec<Vec<u32>>>();
        let (d_tx, decoder_rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let (e_tx, encoder_rx) = std::sync::mpsc::channel::<Vec<Vec<u32>>>();
        std::thread::spawn(move || {
            while let Ok(pcm_data) = e_rx.recv() {
                // Can't wait for try blocks to be a thing
                if let Err(err) = (|| {
                    let l = pcm_data.len();
                    let pcm_data =
                        candle::Tensor::from_vec(pcm_data, (1, 1, l), &candle::Device::Cpu)?
                            .to_dtype(dtype)?;
                    let codes = e_encodec.encode_step(&pcm_data.into())?;
                    if let Some(codes) = codes.as_option() {
                        let mut codes = codes.to_vec3::<u32>()?;
                        e_tx.send(codes.remove(0))?;
                    }
                    Ok::<_, anyhow::Error>(())
                })() {
                    eprintln!("error in encoder thread {err:?}")
                }
            }
        });
        std::thread::spawn(move || {
            while let Ok(codes) = d_rx.recv() {
                if let Err(err) = (|| {
                    let codes = candle::Tensor::new(codes, &candle::Device::Cpu)?.unsqueeze(2)?;
                    let pcm_data = d_encodec.decode_step(&codes.into())?;
                    if let Some(pcm_data) = pcm_data.as_option() {
                        let mut pcm_data = pcm_data.to_vec3::<f32>()?;
                        d_tx.send(pcm_data.remove(0).remove(0))?;
                    }
                    Ok::<_, anyhow::Error>(())
                })() {
                    eprintln!("error in decoder thread {err:?}")
                }
            }
        });
        Ok(Self { dtype, encoder_rx, encoder_tx, decoder_rx, decoder_tx })
    }

    fn encode(&mut self, pcm_data: numpy::PyReadonlyArray1<f32>) -> PyResult<()> {
        self.encoder_tx.send(pcm_data.as_array().to_vec()).w()?;
        Ok(())
    }

    fn decode(&mut self, codes: numpy::PyReadonlyArray2<u32>) -> PyResult<()> {
        let codes = codes.as_array();
        let dims = codes.shape();
        let codes = match codes.to_slice() {
            None => py_bail!("input data is not contiguous"),
            Some(data) => data.to_vec(),
        };
        let codes = codes.chunks_exact(dims[1]).map(|v| v.to_vec()).collect::<Vec<_>>();
        self.decoder_tx.send(codes).w()?;
        Ok(())
    }

    fn get_encoded(&mut self, py: Python) -> PyResult<PyObject> {
        match self.encoder_rx.try_recv() {
            Ok(codes) => {
                let codes = numpy::PyArray2::from_vec2_bound(py, &codes)?;
                Ok(codes.into_py(py))
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                py_bail!("worker thread disconnected")
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(py.None()),
        }
    }

    fn get_decoded(&mut self, py: Python) -> PyResult<PyObject> {
        match self.decoder_rx.try_recv() {
            Ok(pcm) => {
                let pcm = numpy::PyArray1::from_vec_bound(py, pcm);
                Ok(pcm.into_py(py))
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                py_bail!("worker thread disconnected")
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(py.None()),
        }
    }
}

/// Writes an audio file using the wav format based on pcm data from a numpy array.
///
/// This only supports a single channel at the moment so the input array data is expected to have a
/// single dimension.
#[pyfunction]
#[pyo3(signature = (filename, data, sample_rate))]
fn write_wav(
    filename: std::path::PathBuf,
    data: numpy::PyReadonlyArray1<f32>,
    sample_rate: u32,
) -> PyResult<()> {
    let w = std::fs::File::create(&filename).w_f(&filename)?;
    let mut w = std::io::BufWriter::new(w);
    let data = data.as_array().to_vec();
    mm::wav::write_pcm_as_wav(&mut w, &data, sample_rate).w_f(&filename)?;
    Ok(())
}

#[pymodule]
fn rustymimi(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_class::<StreamTokenizer>()?;
    m.add_function(wrap_pyfunction!(write_wav, m)?)?;
    Ok(())
}
