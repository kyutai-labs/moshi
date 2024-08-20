// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use lazy_static::lazy_static;
use prometheus::{
    histogram_opts, labels, opts, register_counter, register_gauge, register_histogram,
};
use prometheus::{Counter, Gauge, Histogram};

pub mod dispatcher {
    use super::*;
    lazy_static! {
        pub static ref ADD_USER: Counter = register_counter!(opts!(
            "dispatcher_add_user",
            "Number of add_user requests made.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref CHECK_USER: Counter = register_counter!(opts!(
            "dispatcher_check_user",
            "Number of check_user requests made.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref USER_FEEDBACK: Counter = register_counter!(opts!(
            "dispatcher_user_feedback",
            "Number of feedback requests made.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref USER_MATCHED: Counter = register_counter!(opts!(
            "dispatcher_user_matched",
            "Number of matched users.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref USER_TIMED_OUT: Counter = register_counter!(opts!(
            "dispatcher_user_timed_out",
            "Number of timed-out users.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref USER_IN_QUEUE: Gauge = register_gauge!(opts!(
            "dispatcher_user_in_queue",
            "Number of users currently in queue.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
        pub static ref MATCHED_WAIT_TIME: Histogram = register_histogram!(histogram_opts!(
            "dispatcher_matched_wait_time",
            "Waiting times for matched users.",
            vec![1.0, 2.0, 5.0, 15., 30., 60., 120., 300., 600.],
        ))
        .unwrap();
        pub static ref TIMED_OUT_WAIT_TIME: Histogram = register_histogram!(histogram_opts!(
            "dispatcher_timed_out_wait_time",
            "Waiting times for timed-out users.",
            vec![1.0, 2.0, 5.0, 15., 30., 60., 120., 300., 600.],
        ))
        .unwrap();
        pub static ref AVAILABLE_WORKERS: Gauge = register_gauge!(opts!(
            "dispatcher_available_workers",
            "Number of available workers.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }
}

pub mod worker {
    use super::*;
    lazy_static! {
        pub static ref CHAT: Counter = register_counter!(opts!(
            "worker_chat",
            "Number of worker chat requests made.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }

    lazy_static! {
        pub static ref CHAT_AUTH_ISSUE: Counter = register_counter!(opts!(
            "worker_chat_auth_issues",
            "Number of worker chat requests that resulted in an auth issue.",
            labels! {"handler" => "all",}
        ))
        .unwrap();
    }

    lazy_static! {
        pub static ref CHAT_DURATION: Histogram = register_histogram!(histogram_opts!(
            "worker_chat_duration",
            "Chat duration distribution.",
            vec![1.0, 5.0, 20., 60., 180.],
        ))
        .unwrap();
    }

    lazy_static! {
        pub static ref MODEL_STEP_DURATION: Histogram = register_histogram!(histogram_opts!(
            "worker_model_step_duration",
            "Model step duration distribution.",
            vec![40e-3, 50e-3, 60e-3, 75e-3, 80e-3, 0.1],
        ))
        .unwrap();
    }
}
