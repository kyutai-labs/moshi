// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use lazy_static::lazy_static;
#[cfg(feature = "cuda")]
use nvml_wrapper::{enum_wrappers::device::TemperatureSensor, error::NvmlError, Nvml};
use prometheus::{
    histogram_opts, labels, opts, register_counter, register_gauge, register_histogram,
};
use prometheus::{Counter, Gauge, Histogram};
use sysinfo::System;

pub mod system {
    use super::*;
    #[cfg(feature = "cuda")]
    lazy_static! {
        pub static ref GPU_UTILIZATION: Gauge = register_gauge!(opts!(
            "system_gpu_utilization",
            "Utilization of the GPU. (percentage)",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref GPU_MEMORY_UTILIZATION: Gauge = register_gauge!(opts!(
            "system_gpu_memory_utilization",
            "Utilization of the GPU's memory. (percentage)",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref GPU_TEMPERATURE: Gauge = register_gauge!(opts!(
            "system_gpu_temperature",
            "Temperature of the GPU.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }

    lazy_static! {
        pub static ref CPU_UTILIZATION: Gauge = register_gauge!(opts!(
            "system_cpu_utilization",
            "Utilization of the CPU. (percentage)",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref MEMORY_UTILIZATION: Gauge = register_gauge!(opts!(
            "system_memory_utilization",
            "Utilization of the system memory. (percentage)",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }

    #[cfg(feature = "cuda")]
    struct GpuMetrics {
        utilization: f64,
        total_memory: f64,
        used_memory: f64,
        temperature: f64,
    }

    #[cfg(feature = "cuda")]
    fn get_gpu_metrics(nvml: &Nvml, device_idx: u32) -> Result<GpuMetrics, NvmlError> {
        let dev = nvml.device_by_index(device_idx)?;
        let memory_info = dev.memory_info()?;

        Ok(GpuMetrics {
            utilization: dev.utilization_rates()?.gpu as f64,
            total_memory: memory_info.total as f64,
            used_memory: memory_info.used as f64,
            temperature: dev.temperature(TemperatureSensor::Gpu).map_or(f64::NAN, |x| x as f64),
        })
    }

    #[cfg(feature = "cuda")]
    fn get_gpus_metrics() -> Result<Option<GpuMetrics>, NvmlError> {
        let nvml = Nvml::init()?;

        let mut global_metrics = GpuMetrics {
            utilization: 0.0,
            total_memory: 0.0,
            used_memory: 0.0,
            temperature: f64::NAN,
        };
        let mut gpu_count: usize = 0;

        for device_idx in 0..nvml.device_count()? {
            match get_gpu_metrics(&nvml, device_idx) {
                Ok(metrics) => {
                    global_metrics.utilization += metrics.utilization;
                    global_metrics.total_memory += metrics.total_memory;
                    global_metrics.used_memory += metrics.used_memory;
                    global_metrics.temperature = global_metrics.temperature.max(metrics.temperature);

                    gpu_count += 1;
                }
                Err(err) => {
                    tracing::debug!(?err, "couldn't get statistics on one gpu");
                }
            }
        }

        if gpu_count == 0 {
            tracing::debug!("no gpu was found while collecting metrics");

            Ok(None)
        } else {
            global_metrics.utilization /= (gpu_count as f64);
            Ok(Some(global_metrics))
        }
    }

    pub(crate) async fn update_system_metrics() {
        #[cfg(feature = "cuda")]
        {
            match get_gpus_metrics() {
                Ok(Some(metrics)) => {
                    GPU_UTILIZATION.set(metrics.utilization);
                    GPU_MEMORY_UTILIZATION.set(metrics.used_memory / metrics.total_memory);
                    GPU_TEMPERATURE.set(metrics.temperature);
                }
                Err(err) => {
                    tracing::debug!(?err, "error while collecting gpu statistics");
                }
                _ => ()
            };
        }

        let sys = System::new_all();
        CPU_UTILIZATION.set((sys.global_cpu_usage() as f64) / 100.0);
        MEMORY_UTILIZATION.set((sys.used_memory() as f64) / (sys.total_memory() as f64));
    }
}

pub mod asr {
    use super::*;
    lazy_static! {
        pub static ref CONNECT: Counter = register_counter!(opts!(
            "asr_connect",
            "Number of connections to the asr.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref MODEL_STEP_DURATION: Histogram = register_histogram!(histogram_opts!(
            "asr_model_step_duration",
            "ASR model step duration distribution.",
            vec![20e-3, 30e-3, 40e-3, 50e-3, 60e-3, 70e-3, 80e-3],
        ))
        .unwrap();
        pub static ref CONNECTION_NUM_STEPS: Histogram = register_histogram!(histogram_opts!(
            "asr_connection_num_steps",
            "ASR model, distribution of number of steps for a connection.",
            vec![2., 25., 125., 250., 500., 750., 1125., 1500., 2250., 3000., 4500.],
        ))
        .unwrap();
        pub static ref OPEN_CHANNELS: Gauge = register_gauge!(opts!(
            "asr_open_channels",
            "Number of open channels (users currently connected).",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }
}

pub mod py {
    use super::*;
    lazy_static! {
        pub static ref CONNECT: Counter = register_counter!(opts!(
            "py_connect",
            "Number of connections to the py-module.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref TOTAL_STEPS: Counter = register_counter!(opts!(
            "py_total_steps",
            "Total number of times the python callback was called.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref ACTIVE_STEPS: Counter = register_counter!(opts!(
            "py_active_steps",
            "Number of times the python callback was called with some active users.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref MISSING_WORDS_STEPS: Counter = register_counter!(opts!(
            "py_missing_words_steps",
            "Number of times the user failed to send words fast enough.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref COULD_HAVE_RUN_STEPS: Counter = register_counter!(opts!(
            "py_could_have_run_steps",
            "Number of times we ran the callback with enough words for a user.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref MODEL_STEP_DURATION: Histogram = register_histogram!(histogram_opts!(
            "py_model_step_duration",
            "py module step duration distribution.",
            vec![10e-3, 15e-3, 20e-3, 30e-3, 40e-3, 50e-3, 80e-3],
        ))
        .unwrap();
        pub static ref CONNECTION_NUM_STEPS: Histogram = register_histogram!(histogram_opts!(
            "py_model_connection_num_steps",
            "py module number of steps with data being generated.",
            vec![2., 25., 62.5, 125., 250., 500., 750.],
        ))
        .unwrap();
        pub static ref OPEN_CHANNELS: Gauge = register_gauge!(opts!(
            "py_open_channels",
            "Number of open channels (users currently connected).",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }
}

pub mod py_post {
    use super::*;
    lazy_static! {
        pub static ref CONNECT: Counter = register_counter!(opts!(
            "py_post_connect",
            "Number of connections to the py_post module.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref MODEL_DURATION: Histogram = register_histogram!(histogram_opts!(
            "py_post_model_duration",
            "py-post model duration distribution.",
            vec![20e-3, 30e-3, 40e-3, 50e-3, 60e-3, 70e-3, 80e-3],
        ))
        .unwrap();
    }
}
