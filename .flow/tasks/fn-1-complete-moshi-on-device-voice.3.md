# fn-1-complete-moshi-on-device-voice.3 Create Swift FFI wrapper (MoshiFFI.swift)

## Description
Create an idiomatic Swift wrapper (`MoshiFFI.swift`) that wraps the C FFI functions exported by the `moshi-ios` crate. The wrapper should provide a safe, ergonomic Swift API with proper memory management (ARC via `deinit`), error handling (Swift `Error` types), and thread safety (serial dispatch queue).

**Size:** M
**Files:** `rust/moshi-ios/swift/MoshiFFI.swift` (new), `rust/moshi-ios/swift/MoshiError.swift` (new)

## Approach

- Import the `MoshiFFI` framework module (from XCFramework) to access C functions
- Create a `MoshiSession` class wrapping the opaque `OpaquePointer` returned by `moshi_create()`
- Use `deinit` to call `moshi_destroy()` — prevents memory leaks
- Enforce thread safety with a private serial `DispatchQueue` for all FFI calls (the Rust FFI is NOT thread-safe — `MoshiSession` has no Send/Sync)
- Bridge errors via `moshi_get_last_error()` → `MoshiError` enum with `.loadFailed(String)`, `.processingFailed(String)`, `.notInitialized` cases
- Use `withCString` for passing Swift strings to C (avoids manual CString management)
- Use `defer { moshi_free_string(ptr) }` pattern for strings returned by `moshi_get_model_text()` / `moshi_get_user_text()`
- Do NOT wrap `moshi_load_model()` (env-var based, inappropriate for iOS) — wrap only the path-based load functions
- Follow the opaque pointer + paired create/destroy pattern from `rust/moshi-ios/src/lib.rs:314-554`

## Key context

- C type mapping: `moshi_session_t*` → `OpaquePointer`, `const char*` → `UnsafePointer<CChar>?`, `float*` → `UnsafeMutablePointer<Float>`, `int` → `Int32`
- `moshi_process_audio_ex()` (line 440) is preferred over `moshi_process_audio()` (line 485) because it takes separate `capacity` and `output_len` params — safer for variable-length output
- Empty audio input returns `Ok(Vec::new())` (line 133-135) — Swift wrapper must handle zero-length output without treating as error
- `moshi_get_user_text()` may always return null (appears to be a stub — `pending_user_text` is declared but never populated at line 60)
- Frame size = 1920 samples at 24kHz. Document this constant in Swift wrapper.
- All C functions follow convention: return `0` success, `-1` failure, with error retrievable via `moshi_get_last_error()`
## Acceptance
- [ ] `MoshiSession` class wraps opaque pointer with `deinit` calling `moshi_destroy()`
- [ ] All relevant C functions wrapped (14 of 16 — excludes `moshi_load_model` and `moshi_free_string` which is internal)
- [ ] Error handling bridges `moshi_get_last_error()` to Swift `MoshiError` type
- [ ] Thread safety enforced via serial `DispatchQueue`
- [ ] String memory correctly managed with `defer { moshi_free_string() }` pattern
- [ ] `processAudio` method uses `moshi_process_audio_ex` (not the simpler variant)
- [ ] Sample rate and frame size constants exposed as static properties
- [ ] Swift file compiles when imported alongside MoshiFFI.xcframework
- [ ] No force-unwraps (`!`) — all optionals handled safely
## Done summary
TBD

## Evidence
- Commits:
- Tests:
- PRs:
