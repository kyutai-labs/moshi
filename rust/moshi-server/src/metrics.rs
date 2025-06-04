// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use lazy_static::lazy_static;
use prometheus::{
    histogram_opts, labels, opts, register_counter, register_gauge, register_histogram,
};
use prometheus::{Counter, Gauge, Histogram};

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
