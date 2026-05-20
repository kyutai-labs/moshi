// moshi-ios - iOS FFI bindings for Moshi
// Exposes a C-compatible streaming inference API for Swift integration.

use std::ffi::{c_char, c_float, c_int, CStr, CString};
use std::ptr;
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle::{Device, IndexOp, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};

const SAMPLE_RATE_HZ: u32 = 24_000;
const DEFAULT_MAX_STEPS: usize = 4_500;
const DEFAULT_TOP_K: usize = 250;
const DEFAULT_TEMPERATURE: f64 = 0.8;
const DEFAULT_SEED: u64 = 299_792_458;

/// Opaque handle to a Moshi inference session.
pub struct MoshiSession {
    device: Device,
    sample_rate: u32,
    runtime: Option<InferenceRuntime>,
}

#[derive(Clone)]
struct RuntimeConfig {
    max_steps: usize,
    text_topk: usize,
    text_temperature: f64,
    text_seed: u64,
    audio_topk: usize,
    audio_temperature: f64,
    audio_seed: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            text_topk: DEFAULT_TOP_K,
            text_temperature: DEFAULT_TEMPERATURE,
            text_seed: DEFAULT_SEED,
            audio_topk: DEFAULT_TOP_K,
            audio_temperature: DEFAULT_TEMPERATURE,
            audio_seed: DEFAULT_SEED,
        }
    }
}

struct InferenceRuntime {
    base_lm_model: moshi::lm::LmModel,
    base_mimi_model: moshi::mimi::Mimi,
    mimi: moshi::mimi::Mimi,
    state: moshi::lm_generate_multistream::State,
    stream_config: moshi::lm_generate_multistream::Config,
    runtime_config: RuntimeConfig,
    generated_audio_codebooks: usize,
    prev_text_token: u32,
    tokenizer: Option<sentencepiece::SentencePieceProcessor>,
    pending_user_text: String,
    pending_model_text: String,
    device: Device,
}

impl InferenceRuntime {
    fn new(
        device: &Device,
        lm_model: moshi::lm::LmModel,
        mimi_model: moshi::mimi::Mimi,
        tokenizer: Option<sentencepiece::SentencePieceProcessor>,
    ) -> Result<Self> {
        let runtime_config = RuntimeConfig::default();
        let stream_config = build_stream_config(&lm_model);
        let generated_audio_codebooks = stream_config.generated_audio_codebooks;

        let state =
            build_generation_state(lm_model.clone(), runtime_config.clone(), stream_config.clone());

        let mut runtime = Self {
            base_lm_model: lm_model,
            base_mimi_model: mimi_model.clone(),
            mimi: mimi_model,
            state,
            stream_config,
            runtime_config,
            generated_audio_codebooks,
            prev_text_token: 0,
            tokenizer,
            pending_user_text: String::new(),
            pending_model_text: String::new(),
            device: device.clone(),
        };
        runtime.prev_text_token = runtime.stream_config.text_start_token;
        runtime.mimi.reset_state();
        warm_up(&runtime.device, &runtime.base_lm_model, &runtime.base_mimi_model)?;
        Ok(runtime)
    }

    fn reset(&mut self) {
        self.state = build_generation_state(
            self.base_lm_model.clone(),
            self.runtime_config.clone(),
            self.stream_config.clone(),
        );
        self.mimi = self.base_mimi_model.clone();
        self.mimi.reset_state();
        self.prev_text_token = self.stream_config.text_start_token;
        self.pending_user_text.clear();
        self.pending_model_text.clear();
    }

    fn take_model_text(&mut self) -> Option<String> {
        if self.pending_model_text.is_empty() {
            None
        } else {
            let text = self.pending_model_text.clone();
            self.pending_model_text.clear();
            Some(text)
        }
    }

    fn take_user_text(&mut self) -> Option<String> {
        if self.pending_user_text.is_empty() {
            None
        } else {
            let text = self.pending_user_text.clone();
            self.pending_user_text.clear();
            Some(text)
        }
    }

    fn process_audio(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let pcm_len = input.len();
        let in_pcm = Tensor::from_vec(input.to_vec(), (1, 1, pcm_len), &self.device)?;
        let codes = self.mimi.encode_step(&in_pcm.into(), &().into())?;
        let Some(audio_tokens) = codes.as_option() else {
            return Ok(Vec::new());
        };

        let (_batch, _codebooks, steps) = audio_tokens.dims3()?;
        let mut out_pcm = Vec::new();

        for step in 0..steps {
            let step_codes = audio_tokens.i((0, .., step))?.to_vec1::<u32>()?;
            let text_token = self
                .state
                .step(self.prev_text_token, &step_codes, None, None)
                .with_context(|| format!("generation step failed at frame step={step}"))?;

            if let Some(delta) = self.decode_text_delta(self.prev_text_token, text_token) {
                self.pending_model_text.push_str(&delta);
            }
            self.prev_text_token = text_token;

            if let Some(audio_tokens) = self.state.last_audio_tokens() {
                let audio_tokens = Tensor::from_slice(
                    &audio_tokens[..self.generated_audio_codebooks],
                    (1, self.generated_audio_codebooks, 1),
                    &self.device,
                )?;
                let decoded = self.mimi.decode_step(&audio_tokens.into(), &().into())?;
                if let Some(decoded) = decoded.as_option() {
                    let pcm = decoded.i((0, 0))?.to_vec1::<f32>()?;
                    out_pcm.extend_from_slice(&pcm);
                }
            }
        }

        Ok(out_pcm)
    }

    fn decode_text_delta(&self, prev_text_token: u32, text_token: u32) -> Option<String> {
        let tokenizer = self.tokenizer.as_ref()?;
        let cfg = &self.stream_config;
        if text_token == cfg.text_start_token
            || text_token == cfg.text_pad_token
            || text_token == cfg.text_eop_token
        {
            return None;
        }

        if prev_text_token == cfg.text_start_token {
            tokenizer.decode_piece_ids(&[text_token]).ok()
        } else {
            let prev = tokenizer.decode_piece_ids(&[prev_text_token]).ok()?;
            let next = tokenizer.decode_piece_ids(&[prev_text_token, text_token]).ok()?;
            if next.len() > prev.len() {
                Some(next[prev.len()..].to_string())
            } else {
                Some(String::new())
            }
        }
    }
}

static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

fn set_error(msg: String) {
    *LAST_ERROR.lock().unwrap() = Some(msg);
}

fn clear_error() {
    *LAST_ERROR.lock().unwrap() = None;
}

fn with_ffi_error<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    clear_error();
    f()
}

fn cstr_to_str<'a>(ptr: *const c_char, field: &str) -> Result<&'a str> {
    if ptr.is_null() {
        anyhow::bail!("{field} pointer is null")
    }
    let s = unsafe { CStr::from_ptr(ptr) };
    Ok(s.to_str().with_context(|| format!("invalid UTF-8 in {field}"))?)
}

fn build_stream_config(model: &moshi::lm::LmModel) -> moshi::lm_generate_multistream::Config {
    let mut cfg = moshi::lm_generate_multistream::Config::v0_1();
    let generated = model.generated_audio_codebooks();
    let total_in = model.in_audio_codebooks();

    cfg.generated_audio_codebooks = generated;
    cfg.input_audio_codebooks = total_in.saturating_sub(generated);
    cfg.audio_vocab_size = model.audio_pad_token() as usize + 1;
    cfg.text_start_token = model.text_start_token();
    cfg
}

fn build_generation_state(
    lm_model: moshi::lm::LmModel,
    runtime_config: RuntimeConfig,
    stream_config: moshi::lm_generate_multistream::Config,
) -> moshi::lm_generate_multistream::State {
    let audio_lp = LogitsProcessor::from_sampling(
        runtime_config.audio_seed,
        Sampling::TopK {
            k: runtime_config.audio_topk,
            temperature: runtime_config.audio_temperature,
        },
    );
    let text_lp = LogitsProcessor::from_sampling(
        runtime_config.text_seed,
        Sampling::TopK {
            k: runtime_config.text_topk,
            temperature: runtime_config.text_temperature,
        },
    );

    moshi::lm_generate_multistream::State::new(
        lm_model,
        runtime_config.max_steps,
        audio_lp,
        text_lp,
        None,
        None,
        None,
        stream_config,
    )
}

fn warm_up(
    device: &Device,
    lm_model: &moshi::lm::LmModel,
    mimi_model: &moshi::mimi::Mimi,
) -> Result<()> {
    let mut lm = lm_model.clone();
    let mut mimi = mimi_model.clone();
    let cb = mimi.config().quantizer_n_q;

    let (_logits, ys) = lm.forward(None, vec![None; cb], &().into())?;
    let mut lp = LogitsProcessor::from_sampling(123, Sampling::ArgMax);
    let _ = lm.depformer_sample(&ys, None, &[], &mut lp)?;

    let frame_len = (mimi.config().sample_rate / mimi.config().frame_rate).ceil() as usize;
    let fake_pcm = Tensor::zeros((1, 1, frame_len), candle::DType::F32, device)?;
    let codes = mimi.encode_step(&fake_pcm.into(), &().into())?;
    let decoded = mimi.decode_step(&codes, &().into())?;
    if decoded.as_option().is_none() {
        anyhow::bail!("Mimi warm-up produced no output")
    }

    device.synchronize()?;
    Ok(())
}

/// Get last error message. Caller must free with moshi_free_string.
#[no_mangle]
pub extern "C" fn moshi_get_last_error() -> *mut c_char {
    let error = LAST_ERROR.lock().unwrap().take();
    match error {
        Some(msg) => CString::new(msg).unwrap().into_raw(),
        None => ptr::null_mut(),
    }
}

/// Free a string returned by moshi functions.
#[no_mangle]
pub extern "C" fn moshi_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Create a new Moshi session. Returns null on failure.
#[no_mangle]
pub extern "C" fn moshi_create() -> *mut MoshiSession {
    match with_ffi_error(create_session) {
        Ok(session) => Box::into_raw(Box::new(session)),
        Err(e) => {
            set_error(e.to_string());
            ptr::null_mut()
        }
    }
}

fn create_session() -> Result<MoshiSession> {
    let device = Device::new_metal(0).context("failed to initialize Metal device")?;
    Ok(MoshiSession { device, sample_rate: SAMPLE_RATE_HZ, runtime: None })
}

/// Load LM model path and use MOSHI_MIMI_MODEL_PATH (+ optional MOSHI_TEXT_TOKENIZER_PATH).
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn moshi_load_model(
    session: *mut MoshiSession,
    lm_model_path: *const c_char,
) -> c_int {
    let mimi_path = std::env::var("MOSHI_MIMI_MODEL_PATH").ok();
    let tokenizer_path = std::env::var("MOSHI_TEXT_TOKENIZER_PATH").ok();

    let rc = with_ffi_error(|| {
        let lm_path = cstr_to_str(lm_model_path, "lm_model_path")?;
        if mimi_path.is_none() {
            anyhow::bail!(
                "MOSHI_MIMI_MODEL_PATH is not set. Use moshi_load_model_with_assets(...) to pass Mimi path explicitly"
            )
        }
        let session =
            unsafe { session.as_mut().ok_or_else(|| anyhow::anyhow!("session pointer is null"))? };
        load_model_impl(session, lm_path, mimi_path.as_deref(), tokenizer_path.as_deref())
    });

    match rc {
        Ok(()) => 0,
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

/// Load LM + Mimi model paths.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn moshi_load_model_with_assets(
    session: *mut MoshiSession,
    lm_model_path: *const c_char,
    mimi_model_path: *const c_char,
) -> c_int {
    moshi_load_model_with_assets_and_tokenizer(session, lm_model_path, mimi_model_path, ptr::null())
}

/// Load LM + Mimi + optional SentencePiece tokenizer.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn moshi_load_model_with_assets_and_tokenizer(
    session: *mut MoshiSession,
    lm_model_path: *const c_char,
    mimi_model_path: *const c_char,
    tokenizer_path: *const c_char,
) -> c_int {
    let rc = with_ffi_error(|| {
        let session =
            unsafe { session.as_mut().ok_or_else(|| anyhow::anyhow!("session pointer is null"))? };
        let lm_path = cstr_to_str(lm_model_path, "lm_model_path")?;
        let mimi_path = cstr_to_str(mimi_model_path, "mimi_model_path")?;
        let tokenizer_path = if tokenizer_path.is_null() {
            None
        } else {
            Some(cstr_to_str(tokenizer_path, "tokenizer_path")?)
        };
        load_model_impl(session, lm_path, Some(mimi_path), tokenizer_path)
    });

    match rc {
        Ok(()) => 0,
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

fn load_model_impl(
    session: &mut MoshiSession,
    lm_path: &str,
    mimi_path: Option<&str>,
    tokenizer_path: Option<&str>,
) -> Result<()> {
    let mimi_path = mimi_path.context("missing mimi model path")?;

    let dtype = if session.device.is_cuda() { candle::DType::BF16 } else { candle::DType::F32 };
    log::info!("Loading LM model from: {lm_path}");
    let lm_model = moshi::lm::load_streaming(lm_path, dtype, &session.device)
        .with_context(|| format!("failed to load LM model: {lm_path}"))?;

    log::info!("Loading Mimi model from: {mimi_path}");
    let mimi_model = moshi::mimi::load(mimi_path, Some(8), &session.device)
        .with_context(|| format!("failed to load Mimi model: {mimi_path}"))?;

    let tokenizer = match tokenizer_path {
        None => None,
        Some(path) => {
            log::info!("Loading tokenizer from: {path}");
            Some(
                sentencepiece::SentencePieceProcessor::open(path)
                    .with_context(|| format!("failed to load tokenizer: {path}"))?,
            )
        }
    };

    let runtime = InferenceRuntime::new(&session.device, lm_model, mimi_model, tokenizer)?;
    session.runtime = Some(runtime);
    Ok(())
}

/// Process streaming PCM input and produce streaming PCM output.
/// `input_samples`/`output_samples` are f32 mono PCM at 24kHz.
/// `output_capacity` is number of f32 elements allocated in `output_samples`.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn moshi_process_audio_ex(
    session: *mut MoshiSession,
    input_samples: *const c_float,
    input_len: c_int,
    output_samples: *mut c_float,
    output_capacity: c_int,
    output_len: *mut c_int,
) -> c_int {
    let rc = with_ffi_error(|| {
        if input_len < 0 || output_capacity < 0 {
            anyhow::bail!("input_len/output_capacity must be non-negative")
        }
        let session =
            unsafe { session.as_mut().ok_or_else(|| anyhow::anyhow!("session pointer is null"))? };
        let runtime =
            session.runtime.as_mut().ok_or_else(|| anyhow::anyhow!("model is not loaded"))?;

        let input = unsafe { std::slice::from_raw_parts(input_samples, input_len as usize) };
        let output = runtime.process_audio(input)?;
        if output.len() > output_capacity as usize {
            anyhow::bail!(
                "output buffer too small: need {}, have {}",
                output.len(),
                output_capacity
            )
        }

        unsafe {
            ptr::copy_nonoverlapping(output.as_ptr(), output_samples, output.len());
            *output_len = output.len() as c_int;
        }
        Ok(())
    });

    match rc {
        Ok(()) => 0,
        Err(e) => {
            set_error(e.to_string());
            -1
        }
    }
}

/// Backward-compatible wrapper where output capacity equals input length.
#[no_mangle]
pub extern "C" fn moshi_process_audio(
    session: *mut MoshiSession,
    input_samples: *const c_float,
    input_len: c_int,
    output_samples: *mut c_float,
    output_len: *mut c_int,
) -> c_int {
    moshi_process_audio_ex(session, input_samples, input_len, output_samples, input_len, output_len)
}

/// Get transcribed user text since the last poll.
/// Returns null if no text is available.
#[no_mangle]
pub extern "C" fn moshi_get_user_text(session: *mut MoshiSession) -> *mut c_char {
    let rc = with_ffi_error(|| {
        let session =
            unsafe { session.as_mut().ok_or_else(|| anyhow::anyhow!("session pointer is null"))? };
        Ok(session.runtime.as_mut().and_then(InferenceRuntime::take_user_text))
    });

    match rc {
        Ok(Some(text)) => CString::new(text).unwrap().into_raw(),
        Ok(None) => ptr::null_mut(),
        Err(e) => {
            set_error(e.to_string());
            ptr::null_mut()
        }
    }
}

/// Get model text since the last poll.
/// Returns null if no text is available.
#[no_mangle]
pub extern "C" fn moshi_get_model_text(session: *mut MoshiSession) -> *mut c_char {
    let rc = with_ffi_error(|| {
        let session =
            unsafe { session.as_mut().ok_or_else(|| anyhow::anyhow!("session pointer is null"))? };
        Ok(session.runtime.as_mut().and_then(InferenceRuntime::take_model_text))
    });

    match rc {
        Ok(Some(text)) => CString::new(text).unwrap().into_raw(),
        Ok(None) => ptr::null_mut(),
        Err(e) => {
            set_error(e.to_string());
            ptr::null_mut()
        }
    }
}

/// Reset streaming state without unloading model weights.
#[no_mangle]
pub extern "C" fn moshi_reset(session: *mut MoshiSession) {
    let rc = with_ffi_error(|| {
        let session =
            unsafe { session.as_mut().ok_or_else(|| anyhow::anyhow!("session pointer is null"))? };
        if let Some(runtime) = session.runtime.as_mut() {
            runtime.reset();
        }
        Ok(())
    });

    if let Err(e) = rc {
        set_error(e.to_string());
    }
}

/// Destroy session and free memory.
#[no_mangle]
pub extern "C" fn moshi_destroy(session: *mut MoshiSession) {
    if !session.is_null() {
        unsafe {
            let _ = Box::from_raw(session);
        }
    }
}

/// Get sample rate in Hz.
#[no_mangle]
pub extern "C" fn moshi_get_sample_rate() -> c_int {
    SAMPLE_RATE_HZ as c_int
}

/// Returns 1 if Metal is available, otherwise 0.
#[no_mangle]
pub extern "C" fn moshi_metal_available() -> c_int {
    if candle::utils::metal_is_available() {
        1
    } else {
        0
    }
}

/// Returns 1 if the session has loaded models, otherwise 0.
#[no_mangle]
pub extern "C" fn moshi_is_initialized(session: *mut MoshiSession) -> c_int {
    if session.is_null() {
        return 0;
    }
    let session = unsafe { &*session };
    if session.runtime.is_some() {
        1
    } else {
        0
    }
}

/// Returns the sample-rate configured in this session.
#[no_mangle]
pub extern "C" fn moshi_session_sample_rate(session: *mut MoshiSession) -> c_int {
    if session.is_null() {
        return SAMPLE_RATE_HZ as c_int;
    }
    let session = unsafe { &*session };
    session.sample_rate as c_int
}
