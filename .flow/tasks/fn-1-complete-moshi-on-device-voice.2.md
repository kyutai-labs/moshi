# fn-1-complete-moshi-on-device-voice.2 Rebuild XCFramework with updated headers and optimized size

## Description
Rebuild the MoshiFFI XCFramework with the current source header (all 16 C functions) and optimize binary size by stripping debug symbols. The pre-built XCFramework at `rust/build/MoshiFFI.xcframework/` is **stale** — its header (`ios-arm64/MoshiFFI.framework/Headers/moshi_ios.h`, 84 lines) is missing 5 functions that exist in the source header (`rust/moshi-ios/include/moshi_ios.h`, 119 lines): `moshi_process_audio_ex`, `moshi_load_model_with_assets`, `moshi_load_model_with_assets_and_tokenizer`, `moshi_is_initialized`, `moshi_session_sample_rate`.

**Size:** M
**Files:** `rust/scripts/build-xcframework.sh` (modify), `rust/moshi-ios/Cargo.toml` (modify), `rust/build/MoshiFFI.xcframework/` (rebuilt output)

## Approach

- Update `rust/scripts/build-xcframework.sh` to use a release profile that strips debug symbols (currently `--release` includes `debug = true` from `rust/Cargo.toml:84`, producing ~128MB .a files)
- Options: use `--profile release-no-debug` (already defined at `rust/Cargo.toml:87` with `debug = false`), or add `strip = "symbols"` to the release profile
- Consider removing `cdylib` from `crate-type` in `rust/moshi-ios/Cargo.toml:8` — only `staticlib` is needed for XCFramework, and `cdylib` may slow builds
- Run `cbindgen` (via `build.rs`) to regenerate `rust/moshi-ios/include/moshi_ios.h` with all functions
- Build for both `aarch64-apple-ios` and `aarch64-apple-ios-sim` targets
- Package into XCFramework with proper `module.modulemap` and headers
- Verify resulting header has all 16 C functions using `grep -c` on the output header

## Key context

- The build script at `rust/scripts/build-xcframework.sh` already handles the full pipeline (cross-compile → framework bundle → xcodebuild -create-xcframework)
- Installed Rust targets confirmed: `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `aarch64-apple-darwin`
- cbindgen config at `rust/moshi-ios/cbindgen.toml:8-11` renames `MoshiSession` to `moshi_session_t`
- The module map uses `framework module MoshiFFI` syntax
- Bundle identifier `ai.openclaw.MoshiFFI` in build script line 63 — leave as-is unless user requests change
## Acceptance
- [ ] XCFramework rebuilt at `rust/build/MoshiFFI.xcframework/`
- [ ] Header in XCFramework contains all 16 C functions (matches source header)
- [ ] Both `ios-arm64` and `ios-arm64-simulator` slices present
- [ ] Static library size < 50MB per slice (debug symbols stripped)
- [ ] `module.modulemap` present in each framework's Modules/ directory
- [ ] `lipo -info` confirms correct architecture for each slice
- [ ] `cdylib` removed from crate-type (only `staticlib` remains)
- [ ] Build script updated to use stripped release profile
- [ ] Existing `cargo test` in workspace still passes
## Done summary
Rebuilt MoshiFFI XCFramework with all 17 C functions in header (was missing 5), optimized binary size from ~128MB to ~46MB per slice by using release-no-debug profile with strip="symbols", and fixed sentencepiece cross-compilation for CMake 4.x via iOS toolchain shim.
## Evidence
- Commits: 721126a6ece690d2802b97c8211e0fcfe26524e3
- Tests: cargo test --package moshi-ios, cargo test --package moshi, ./rust/scripts/build-xcframework.sh
- PRs: