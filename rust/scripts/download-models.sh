#!/usr/bin/env bash
# Download required model files from HuggingFace for Moshi iOS inference.
#
# Usage:
#   ./rust/scripts/download-models.sh [OUTPUT_DIR]
#
# The output directory can also be set via the MOSHI_MODEL_DIR env var.
# Default: rust/models/ (relative to repo root)
#
# Environment:
#   HF_TOKEN          - HuggingFace token (optional, needed for gated repos)
#   MOSHI_MODEL_DIR   - Override default download directory
#
# Models downloaded:
#   kyutai/moshika-candle-q8  -> model.q8.gguf             (~8.5GB LM weights)
#   kyutai/moshika-candle-q8  -> tokenizer_spm_32k_3.model (text tokenizer)
#   kyutai/mimi               -> model.safetensors          (~769MB Mimi codec)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HuggingFace repos and filenames
LM_REPO="kyutai/moshika-candle-q8"
LM_FILE="model.q8.gguf"

TOKENIZER_REPO="kyutai/moshika-candle-q8"
TOKENIZER_FILE="tokenizer_spm_32k_3.model"

MIMI_REPO="kyutai/mimi"
MIMI_FILE="model.safetensors"

# Resolve output directory: CLI arg > env var > default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_MODEL_DIR="$REPO_ROOT/rust/models"

MODEL_DIR="${1:-${MOSHI_MODEL_DIR:-$DEFAULT_MODEL_DIR}}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "==> $*"; }
warn()  { echo "WARNING: $*" >&2; }
error() { echo "ERROR: $*" >&2; exit 1; }

# Download a single file from a HuggingFace repo.
# Args: repo_id filename local_path
download_hf_file() {
    local repo="$1"
    local filename="$2"
    local dest="$3"

    if [[ -f "$dest" ]]; then
        info "Already exists, skipping: $dest"
        return 0
    fi

    local url="https://huggingface.co/${repo}/resolve/main/${filename}"
    info "Downloading ${repo}/${filename} ..."
    info "  URL:  $url"
    info "  Dest: $dest"

    local curl_args=(
        --location          # follow redirects
        --fail              # fail on HTTP errors
        --progress-bar      # show progress
        --output "$dest"
    )

    # Include auth header if HF_TOKEN is available
    if [[ -n "${HF_TOKEN:-}" ]]; then
        curl_args+=(--header "Authorization: Bearer ${HF_TOKEN}")
    fi

    if ! curl "${curl_args[@]}" "$url"; then
        # Clean up partial download
        rm -f "$dest"
        error "Failed to download ${repo}/${filename}. If the repo is gated, set HF_TOKEN."
    fi

    info "Downloaded: $dest ($(du -h "$dest" | cut -f1))"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if ! command -v curl &>/dev/null; then
    error "curl is required but not found in PATH."
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    warn "HF_TOKEN is not set. Downloads may fail if the model repos are gated."
    warn "Set it with: export HF_TOKEN=hf_..."
fi

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

info "Moshi model download script"
info "Output directory: $MODEL_DIR"
echo

# Create output directory
mkdir -p "$MODEL_DIR"

# Download all three model files
download_hf_file "$LM_REPO"        "$LM_FILE"        "$MODEL_DIR/$LM_FILE"
echo
download_hf_file "$TOKENIZER_REPO" "$TOKENIZER_FILE"  "$MODEL_DIR/$TOKENIZER_FILE"
echo
download_hf_file "$MIMI_REPO"      "$MIMI_FILE"       "$MODEL_DIR/$MIMI_FILE"
echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

info "All downloads complete. Model files:"
echo
echo "  LM weights:     $MODEL_DIR/$LM_FILE"
echo "  Text tokenizer: $MODEL_DIR/$TOKENIZER_FILE"
echo "  Mimi codec:     $MODEL_DIR/$MIMI_FILE"
echo
info "Pass these paths to moshi_load_model_with_assets_and_tokenizer()."
