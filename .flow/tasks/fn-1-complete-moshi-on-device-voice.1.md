# fn-1-complete-moshi-on-device-voice.1 Create model download script for HuggingFace models

## Description
Create a shell script that downloads all required model files from HuggingFace for Moshi iOS inference. The `hf-hub` crate is NOT a dependency of `moshi-ios`, so model download must happen externally before calling the FFI.

**Size:** M
**Files:** `rust/scripts/download-models.sh` (new)

## Approach

- Follow the existing HuggingFace download pattern at `rust/moshi-backend/src/standalone.rs:112-134` and `rust/moshi-server/src/utils.rs:58` for repo/filename conventions
- Use `huggingface-cli download` (Python CLI) or `curl` with HF API for downloading — avoids adding Rust deps
- Download three files from two repos:
  1. `kyutai/moshika-candle-q8` → `model.q8.gguf` (~8.5GB LM weights)
  2. `kyutai/moshika-candle-q8` → `tokenizer_spm_32k_3.model` (text tokenizer)
  3. `kyutai/mimi` → `model.safetensors` (~769MB Mimi codec)
- Store files in a configurable directory (default: `rust/models/`)
- Skip download if files already exist (idempotent)
- Reference canonical filenames from `moshi/moshi/models/loaders.py:31-35`

## Key context

- The q8 GGUF variant is chosen over bf16 safetensors because bf16 gets loaded as F32 on Metal, doubling effective memory (~31GB vs ~8.5GB)
- `moshi_load_model_with_assets_and_tokenizer()` at `rust/moshi-ios/src/lib.rs:374` takes three separate file paths for LM, Mimi, and tokenizer
- HuggingFace repos may require `HF_TOKEN` env var for gated models — script should check and warn
## Acceptance
- [ ] Script downloads `model.q8.gguf` from `kyutai/moshika-candle-q8`
- [ ] Script downloads `tokenizer_spm_32k_3.model` from `kyutai/moshika-candle-q8`
- [ ] Script downloads `model.safetensors` from `kyutai/mimi`
- [ ] Script is idempotent — skips existing files
- [ ] Script prints clear progress and final file paths
- [ ] Script is executable (`chmod +x`)
- [ ] Download directory is configurable via argument or env var
- [ ] Script warns if `HF_TOKEN` is not set (in case repos are gated)
## Done summary
Created download-models.sh script that fetches all three required model files (LM q8 GGUF, text tokenizer, Mimi codec) from HuggingFace using curl. Script is idempotent, supports configurable output directory via CLI arg or MOSHI_MODEL_DIR env var, and warns when HF_TOKEN is not set.
## Evidence
- Commits: aedfaf36e7004b185e3933b63a6c151224aaef71
- Tests: bash -n rust/scripts/download-models.sh
- PRs: