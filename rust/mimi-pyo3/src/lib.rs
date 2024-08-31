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

#[pyclass]
struct Tokenizer {
    encodec: encodec::Encodec,
    device: candle::Device,
    dtype: candle::DType,
}

#[pymethods]
impl Tokenizer {
    #[pyo3(signature = (path, *, dtype="f32"))]
    #[new]
    fn new(path: &str, dtype: &str) -> PyResult<Self> {
        let device = candle::Device::Cpu;
        let dtype = match dtype {
            "f32" => candle::DType::F32,
            "f16" => candle::DType::F16,
            "bf16" => candle::DType::BF16,
            dtype => py_bail!("unsupported dtype '{dtype}'"),
        };
        let vb =
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[path], dtype, &device).w()? };
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
            cross_attention: false,
            max_seq_len: 4096,
        };
        let cfg = encodec::Config {
            channels: 1,
            sample_rate: 24_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: encodec::ResampleMethod::Conv,
            seanet: seanet_cfg,
            transformer: transformer_cfg,
            quantizer_n_q: 16,
            quantizer_bins: 2048,
            quantizer_dim: 256,
        };

        let encodec = encodec::Encodec::new(cfg, vb).w()?;
        Ok(Self { encodec, device, dtype })
    }

    fn encode(&mut self, pcm_data: numpy::PyReadonlyArray3<f32>, py: Python) -> PyResult<PyObject> {
        let pcm_data = pcm_data.as_array();
        // TODO(laurent): maybe this should be run in another thread?
        let codes = py
            .allow_threads(|| {
                let pcm_shape = pcm_data.shape().to_vec();
                let pcm_data = pcm_data.iter().copied().collect::<Vec<_>>();
                let pcm_data = candle::Tensor::from_vec(pcm_data, pcm_shape, &self.device)?
                    .to_dtype(self.dtype)?;
                let codes = self.encodec.encode(&pcm_data)?;
                codes.to_vec3::<u32>()
            })
            .w()?;
        let codes = numpy::PyArray3::from_vec3_bound(py, &codes)?;
        Ok(codes.into_py(py))
    }

    fn encode_step(
        &mut self,
        pcm_data: numpy::PyReadonlyArray3<f32>,
        py: Python,
    ) -> PyResult<PyObject> {
        let pcm_data = pcm_data.as_array();
        // TODO(laurent): maybe this should be run in another thread?
        let codes = py
            .allow_threads(|| {
                let pcm_shape = pcm_data.shape().to_vec();
                let pcm_data = pcm_data.iter().copied().collect::<Vec<_>>();
                let pcm_data = candle::Tensor::from_vec(pcm_data, pcm_shape, &self.device)?
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
        // TODO(laurent): maybe this should be run in another thread?
        let pcm = py
            .allow_threads(|| {
                let codes_shape = codes.shape().to_vec();
                let codes = codes.iter().copied().collect::<Vec<_>>();
                let codes = candle::Tensor::from_vec(codes, codes_shape, &self.device)?;
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
        // TODO(laurent): maybe this should be run in another thread?
        let pcm = py
            .allow_threads(|| {
                let codes_shape = codes.shape().to_vec();
                let codes = codes.iter().copied().collect::<Vec<_>>();
                let codes = candle::Tensor::from_vec(codes, codes_shape, &self.device)?;
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
    let data = data.as_array();
    match data.as_slice() {
        None => {
            let data = data.to_vec();
            mm::wav::write_pcm_as_wav(&mut w, data.as_ref(), sample_rate).w_f(&filename)?
        }
        Some(data) => mm::wav::write_pcm_as_wav(&mut w, data, sample_rate).w_f(&filename)?,
    }
    Ok(())
}

#[pymodule]
fn mimi(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_function(wrap_pyfunction!(write_wav, m)?)?;
    Ok(())
}
