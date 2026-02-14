//! macOS integration test for Moshi on-device inference via the C FFI.
//!
//! This test exercises the full lifecycle: create -> load -> process -> verify -> destroy.
//!
//! Model paths are configured via environment variables:
//! - `MOSHI_TEST_LM_PATH` -- path to the q8 GGUF language model
//! - `MOSHI_TEST_MIMI_PATH` -- path to the Mimi codec safetensors
//! - `MOSHI_TEST_TOKENIZER_PATH` -- (optional) path to the SentencePiece tokenizer
//!
//! Tests are `#[ignore]` by default because they require large model files.
//! Run with:
//!   MOSHI_TEST_LM_PATH=... MOSHI_TEST_MIMI_PATH=... cargo test \
//!       --package moshi-ios --test inference_test -- --ignored --nocapture

use std::ffi::CString;
use std::os::raw::c_int;
use std::ptr;

// Import the FFI functions directly from the moshi_ios crate.
// These are `pub extern "C"` functions, callable as normal Rust functions.
use moshi_ios::{
    moshi_create, moshi_destroy, moshi_free_string, moshi_get_last_error, moshi_get_model_text,
    moshi_get_sample_rate, moshi_is_initialized, moshi_load_model_with_assets_and_tokenizer,
    moshi_metal_available, moshi_process_audio_ex,
};

/// Number of samples per frame at 24kHz (80ms).
const FRAME_SIZE: usize = 1920;

/// Minimum number of frames to process (spec requires >= 10).
const MIN_FRAMES: usize = 10;

/// Output buffer capacity -- generous to handle variable output sizes.
const OUTPUT_CAPACITY: usize = FRAME_SIZE * 4;

/// Helper: get and clear the last FFI error as a Rust String.
fn get_last_error() -> Option<String> {
    unsafe {
        let ptr = moshi_get_last_error();
        if ptr.is_null() {
            None
        } else {
            let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
            moshi_free_string(ptr);
            Some(s)
        }
    }
}

/// Generate a sine wave at the given frequency, at 24kHz sample rate.
fn generate_sine_wave(num_samples: usize, frequency_hz: f32) -> Vec<f32> {
    let sample_rate = 24_000.0_f32;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * frequency_hz * t).sin() * 0.5
        })
        .collect()
}

/// Read model paths from environment variables.
/// Returns None if required paths are not set (test will be skipped).
struct ModelPaths {
    lm_path: CString,
    mimi_path: CString,
    tokenizer_path: Option<CString>,
}

fn load_model_paths() -> Option<ModelPaths> {
    let lm_path = std::env::var("MOSHI_TEST_LM_PATH").ok()?;
    let mimi_path = std::env::var("MOSHI_TEST_MIMI_PATH").ok()?;
    let tokenizer_path = std::env::var("MOSHI_TEST_TOKENIZER_PATH").ok();

    // Verify files exist
    if !std::path::Path::new(&lm_path).exists() {
        eprintln!("WARNING: MOSHI_TEST_LM_PATH does not exist: {}", lm_path);
        return None;
    }
    if !std::path::Path::new(&mimi_path).exists() {
        eprintln!("WARNING: MOSHI_TEST_MIMI_PATH does not exist: {}", mimi_path);
        return None;
    }

    Some(ModelPaths {
        lm_path: CString::new(lm_path).ok()?,
        mimi_path: CString::new(mimi_path).ok()?,
        tokenizer_path: tokenizer_path.and_then(|p| {
            if std::path::Path::new(&p).exists() {
                CString::new(p).ok()
            } else {
                eprintln!(
                    "WARNING: MOSHI_TEST_TOKENIZER_PATH does not exist, skipping tokenizer"
                );
                None
            }
        }),
    })
}

#[test]
#[ignore]
fn test_full_inference_lifecycle() {
    // -- Step 0: Check model paths, skip if not set --
    let paths = match load_model_paths() {
        Some(p) => p,
        None => {
            eprintln!(
                "Skipping inference test: set MOSHI_TEST_LM_PATH and MOSHI_TEST_MIMI_PATH to run"
            );
            return;
        }
    };

    // -- Step 1: Verify static helpers --
    let sample_rate = moshi_get_sample_rate();
    assert_eq!(sample_rate, 24000, "Sample rate must be 24000 Hz");
    eprintln!("[OK] moshi_get_sample_rate() = {}", sample_rate);

    let metal = moshi_metal_available();
    assert_eq!(metal, 1, "Metal must be available on Apple Silicon macOS");
    eprintln!("[OK] moshi_metal_available() = {}", metal);

    // -- Step 2: Create session --
    let session = moshi_create();
    assert!(!session.is_null(), "moshi_create() returned null: {:?}", get_last_error());
    eprintln!("[OK] Session created");

    // -- Step 3: Verify not initialized before model load --
    let initialized = moshi_is_initialized(session);
    assert_eq!(initialized, 0, "Session should not be initialized before model load");
    eprintln!("[OK] moshi_is_initialized() = 0 (before load)");

    // -- Step 4: Load model --
    eprintln!("Loading models (this may take a while)...");
    let tokenizer_ptr = paths.tokenizer_path.as_ref().map(|t| t.as_ptr()).unwrap_or(ptr::null());
    let load_result = moshi_load_model_with_assets_and_tokenizer(
        session,
        paths.lm_path.as_ptr(),
        paths.mimi_path.as_ptr(),
        tokenizer_ptr,
    );
    assert_eq!(
        load_result, 0,
        "moshi_load_model_with_assets_and_tokenizer() failed: {:?}",
        get_last_error()
    );
    eprintln!("[OK] Models loaded successfully");

    // -- Step 5: Verify initialized after model load --
    let initialized = moshi_is_initialized(session);
    assert_eq!(initialized, 1, "Session must be initialized after model load");
    eprintln!("[OK] moshi_is_initialized() = 1 (after load)");

    // -- Step 6: Process audio frames --
    // Generate a 440Hz sine wave for MIN_FRAMES frames (0.8 seconds)
    let total_samples = FRAME_SIZE * MIN_FRAMES;
    let input_audio = generate_sine_wave(total_samples, 440.0);

    let mut total_output_samples = 0usize;
    let mut frames_with_output = 0usize;
    let mut all_succeeded = true;

    eprintln!("Processing {} frames ({} samples)...", MIN_FRAMES, total_samples);
    for frame_idx in 0..MIN_FRAMES {
        let frame_start = frame_idx * FRAME_SIZE;
        let frame_end = frame_start + FRAME_SIZE;
        let frame_input = &input_audio[frame_start..frame_end];

        let mut output_buf = vec![0.0_f32; OUTPUT_CAPACITY];
        let mut output_len: c_int = 0;

        let result = moshi_process_audio_ex(
            session,
            frame_input.as_ptr(),
            FRAME_SIZE as c_int,
            output_buf.as_mut_ptr(),
            OUTPUT_CAPACITY as c_int,
            &mut output_len,
        );

        if result != 0 {
            let err = get_last_error().unwrap_or_else(|| "unknown error".to_string());
            eprintln!("  Frame {}: FAILED - {}", frame_idx, err);
            all_succeeded = false;
        } else {
            let out_len = output_len as usize;
            total_output_samples += out_len;
            if out_len > 0 {
                frames_with_output += 1;
            }
            eprintln!("  Frame {}: OK, output_len = {}", frame_idx, out_len);
        }
    }

    assert!(all_succeeded, "All frames must process successfully (return 0)");
    eprintln!(
        "[OK] Processed {} frames: {} frames produced output, {} total output samples",
        MIN_FRAMES, frames_with_output, total_output_samples
    );

    // Verify that at least some frames produced output.
    // The Mimi codec has a look-ahead, so the first few frames may produce 0 output.
    // After enough frames, output must be non-empty.
    assert!(
        total_output_samples > 0,
        "Expected non-zero output samples after {} frames, got 0",
        MIN_FRAMES
    );
    eprintln!("[OK] Total output samples = {} (non-zero)", total_output_samples);

    // -- Step 7: Optionally check model text --
    unsafe {
        let text_ptr = moshi_get_model_text(session);
        if !text_ptr.is_null() {
            let text = std::ffi::CStr::from_ptr(text_ptr).to_string_lossy().into_owned();
            moshi_free_string(text_ptr);
            eprintln!("[INFO] Model text after {} frames: {:?}", MIN_FRAMES, text);
        } else {
            eprintln!(
                "[INFO] No model text produced after {} frames (this is normal for short input)",
                MIN_FRAMES
            );
        }
    }

    // -- Step 8: Destroy session --
    moshi_destroy(session);
    eprintln!("[OK] Session destroyed");

    eprintln!("=== All inference lifecycle checks passed ===");
}

#[test]
#[ignore]
fn test_static_functions() {
    // Verify static FFI functions that do not require model loading.
    let sample_rate = moshi_get_sample_rate();
    assert_eq!(sample_rate, 24000, "Sample rate must be 24000 Hz");

    let metal = moshi_metal_available();
    assert_eq!(metal, 1, "Metal must be available on Apple Silicon macOS");

    // Create and destroy session without loading models.
    let session = moshi_create();
    assert!(!session.is_null(), "moshi_create() returned null: {:?}", get_last_error());

    let initialized = moshi_is_initialized(session);
    assert_eq!(initialized, 0, "Session should not be initialized without model load");

    moshi_destroy(session);
}

#[test]
#[ignore]
fn test_process_audio_without_model_returns_error() {
    // Processing audio without loading a model should return an error.
    let session = moshi_create();
    assert!(!session.is_null());

    let input = vec![0.0_f32; FRAME_SIZE];
    let mut output = vec![0.0_f32; OUTPUT_CAPACITY];
    let mut output_len: c_int = 0;

    let result = moshi_process_audio_ex(
        session,
        input.as_ptr(),
        FRAME_SIZE as c_int,
        output.as_mut_ptr(),
        OUTPUT_CAPACITY as c_int,
        &mut output_len,
    );

    assert_eq!(result, -1, "Processing audio without a loaded model must fail");
    let err = get_last_error();
    assert!(err.is_some(), "Error message must be set on failure");
    eprintln!("[OK] Expected error: {:?}", err.unwrap());

    moshi_destroy(session);
}
