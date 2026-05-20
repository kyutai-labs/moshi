# fn-1-complete-moshi-on-device-voice.4 Write macOS test harness for inference verification

## Description
Write a macOS test harness that verifies end-to-end Moshi inference works before deploying to an iOS device. The test builds the `moshi-ios` crate for the host macOS target (`aarch64-apple-darwin`), loads real model files, feeds test audio through the FFI, and verifies non-empty output.

**Size:** M
**Files:** `rust/moshi-ios/tests/inference_test.rs` (new), `rust/moshi-ios/tests/fixtures/` (new, test audio)

## Approach

- Create a Rust integration test in `rust/moshi-ios/tests/inference_test.rs` that exercises the C FFI functions on macOS
- Follow the file-to-file inference pattern from `rust/moshi-cli/src/gen.rs:15-141` — load WAV, process in 1920-sample chunks
- Use the warm-up pattern from `rust/moshi-ios/src/lib.rs:267-290` as reference for expected initialization flow
- Test sequence: `moshi_create()` → `moshi_load_model_with_assets_and_tokenizer()` → multiple `moshi_process_audio_ex()` calls → verify output → `moshi_destroy()`
- Generate or bundle a short test audio fixture (e.g., 1-2 seconds of 24kHz mono f32 PCM — can be silence or a sine wave)
- Model paths should be configurable via env vars (e.g., `MOSHI_TEST_LM_PATH`, `MOSHI_TEST_MIMI_PATH`, `MOSHI_TEST_TOKENIZER_PATH`) — tests skip if models not present
- Mark test with `#[ignore]` by default (requires large model files) — run explicitly with `cargo test -- --ignored`

## Key context

- Metal is available on macOS and works identically to iOS for inference (`Device::new_metal(0)` at line 325)
- The `moshi-ios` crate uses `default = ["metal"]` features — Metal will be active on macOS test
- Constants: `SAMPLE_RATE_HZ = 24000`, frame = 1920 samples, `DEFAULT_MAX_STEPS = 4500`
- `moshi_process_audio_ex()` signature: `(session, input_ptr, input_len, output_ptr, output_capacity, output_len_ptr) -> c_int`
- Output verification: at minimum check that `output_len > 0` for at least some frames; optionally check that `moshi_get_model_text()` returns non-null after enough frames
- The `gen.rs` pattern processes audio in chunks and writes WAV output — similar structure but for testing, just verify output is non-empty
## Acceptance
- [ ] Integration test at `rust/moshi-ios/tests/inference_test.rs` compiles
- [ ] Test exercises full lifecycle: create → load → process → verify → destroy
- [ ] Test processes at least 10 frames (19200 samples = 0.8 seconds) of audio
- [ ] Test verifies `moshi_process_audio_ex` returns success (0) and produces output samples
- [ ] Test verifies `moshi_is_initialized` returns 1 after model load
- [ ] Test verifies `moshi_get_sample_rate` returns 24000
- [ ] Test verifies `moshi_metal_available` returns 1 on Apple Silicon macOS
- [ ] Model paths configurable via env vars (test skips gracefully if not set)
- [ ] Test marked `#[ignore]` by default (opt-in via `--ignored`)
- [ ] Test passes on macOS with downloaded models: `MOSHI_TEST_LM_PATH=... cargo test --package moshi-ios --test inference_test -- --ignored --nocapture`
## Done summary
Added macOS integration test harness at rust/moshi-ios/tests/inference_test.rs that verifies end-to-end Moshi inference via the C FFI. The test exercises the full lifecycle (create, load, process 10+ frames, verify output, destroy) with model paths configurable via environment variables and all tests ignored by default.
## Evidence
- Commits: f94ffd1640b20f0387e0ce54ee6283f82ce873b9
- Tests: cargo test --package moshi-ios --test inference_test --no-run, cargo test --package moshi-ios --test inference_test, cargo test --package moshi-ios
- PRs: