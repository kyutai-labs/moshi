"""
Example implementation of the streaming STT example. Here we group
test utterances in batches (pre- and post-padded with silence) and
and then feed these batches into the streaming STT model frame-by-frame.

Example command:
```
uv run scripts/streaming_stt.py \
    --dataset meanwhile \
    --hf-repo kyutai/<REPO> \
    --hf-cache-dir /home/user/huggingface_cache
```

"""

# The outputs I get on my H100 using this code with the 2.6B model,
# bsz 32:

# LibriVox === cer: 4.09% wer: 7.33% corpus_wer: 6.78% RTF = 52.72
# Ami === cer: 15.99% wer: 18.78% corpus_wer: 12.20% RTF = 28.37
# LibriSpeech other === cer: 2.31% wer: 5.24% corpus_wer: 4.33% RTF = 44.76
# LibriSpeech clean === cer: 0.67% wer: 1.95% corpus_wer: 1.69% RTF = 68.19
# Tedlium (short) === cer: 2.15% wer: 3.65% corpus_wer: 3.33% RTF = 67.44
# spgispeech === cer: 0.99% wer: 2.00% corpus_wer: 2.03% RTF = 78.64
# gigaspeech === cer: 6.80% wer: 11.31% corpus_wer: 9.81% RTF = 64.04
# earnings22 (short) === cer: 12.63% wer: 15.70% corpus_wer: 11.02% RTF = 50.13

# Meanwhile === cer: 2.02% wer: 5.50% corpus_wer: 5.60% RTF = 69.19
# Tedlium (long) == cer: 1.53% wer: 2.56% corpus_wer: 2.97% RTF = 33.92
# Rev16 === cer: 6.57% wer: 10.08% corpus_wer: 11.43% RTF = 40.34
# Earnings21 === cer: 5.73% wer: 9.84% corpus_wer: 10.38% RTF = 73.15

import dataclasses
import julius
import jiwer
from datasets import load_dataset, Dataset
from whisper.normalizers import EnglishTextNormalizer
import argparse

import torch
import moshi.models
import tqdm
import time


_NORMALIZER = EnglishTextNormalizer()


def get_text(sample):
    possible_keys = [
        "text",
        "sentence",
        "normalized_text",
        "transcript",
        "transcription",
    ]
    for key in possible_keys:
        if key in sample:
            return sample[key]
    raise ValueError(
        f"Expected transcript column of either {possible_keys}."
        f"Got sample with keys: {', '.join(sample.keys())}. Ensure a text column name is present in the dataset."
    )


# The two functions below are adapted from https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/data_utils.py


def normalize(batch):
    batch["original_text"] = get_text(batch)
    batch["norm_text"] = _NORMALIZER(batch["original_text"])
    return batch


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


# End of the adapted part


class AsrMetrics:
    def __init__(self):
        self.cer_sum = 0.0
        self.wer_sum = 0.0
        self.errors_sum = 0.0
        self.total_words_sum = 0.0
        self.num_sequences = 0.0

    def update(self, hyp: str, ref: str) -> None:
        normalized_ref = _NORMALIZER(ref)
        normalized_hyp = _NORMALIZER(hyp)

        this_wer = jiwer.wer(normalized_ref, normalized_hyp)
        this_cer = jiwer.cer(normalized_ref, normalized_hyp)
        measures = jiwer.compute_measures(normalized_ref, normalized_hyp)

        self.wer_sum += this_wer
        self.cer_sum += this_cer
        self.errors_sum += (
            measures["substitutions"] + measures["deletions"] + measures["insertions"]
        )
        self.total_words_sum += (
            measures["substitutions"] + measures["deletions"] + measures["hits"]
        )
        self.num_sequences += 1

    def compute(self) -> dict:
        assert (
            self.num_sequences > 0
        ), "Unable to compute with total number of comparisons <= 0"  # type: ignore
        return {
            "cer": (self.cer_sum / self.num_sequences),
            "wer": (self.wer_sum / self.num_sequences),
            "corpus_wer": (self.errors_sum / self.total_words_sum),
        }

    def __str__(self) -> str:
        result = self.compute()
        return " ".join(f"{k}: {100 * v:.2f}%" for k, v in result.items())


class Timer:
    def __init__(self):
        self.total = 0
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.total += time.perf_counter() - self._start_time
        self._start_time = None


@dataclasses.dataclass
class _DatasetInfo:
    alias: str

    name: str
    config: str
    split: str = "test"


_DATASETS = [
    # Long-form datasets from distil-whisper
    _DatasetInfo("rev16", "distil-whisper/rev16", "whisper_subset"),
    _DatasetInfo("earnings21", "distil-whisper/earnings21", "full"),
    _DatasetInfo("earnings22", "distil-whisper/earnings22", "full"),
    _DatasetInfo("tedlium", "distil-whisper/tedlium-long-form", None),
    _DatasetInfo("meanwhile", "distil-whisper/meanwhile", None),
    # Short-form datasets from OpenASR leaderboard
    _DatasetInfo("ami", "hf-audio/esb-datasets-test-only-sorted", "ami"),
    _DatasetInfo(
        "librispeech.clean",
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        split="test.clean",
    ),
    _DatasetInfo(
        "librispeech.other",
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        split="test.other",
    ),
    _DatasetInfo("voxpopuli", "hf-audio/esb-datasets-test-only-sorted", "voxpopuli"),
    _DatasetInfo("spgispeech", "hf-audio/esb-datasets-test-only-sorted", "spgispeech"),
    _DatasetInfo("gigaspeech", "hf-audio/esb-datasets-test-only-sorted", "gigaspeech"),
    _DatasetInfo("tedlium-short", "hf-audio/esb-datasets-test-only-sorted", "tedlium"),
    _DatasetInfo(
        "earnings22-short", "hf-audio/esb-datasets-test-only-sorted", "earnings22"
    ),
]
DATASET_MAP = {dataset.alias: dataset for dataset in _DATASETS}


def get_dataset(args) -> Dataset:
    if args.dataset not in DATASET_MAP:
        raise RuntimeError("Unknown dataset")

    info = DATASET_MAP[args.dataset]

    dataset = load_dataset(
        info.name,
        info.config,
        split=info.split,
        cache_dir=args.hf_cache_dir,
        streaming=False,
        token=True,
    )
    dataset = dataset.map(normalize)
    dataset = dataset.filter(is_target_text_in_range, input_columns=["norm_text"])

    return dataset


@torch.no_grad
def get_padded_batch(
    audios: list[tuple[torch.Tensor, int]],
    before_padding: float,
    after_padding: float,
    audio_encoder,
):
    sample_rate = audio_encoder.sample_rate

    max_len = 0
    batch = []
    durations = []
    for audio, sr in audios:
        durations.append(audio.shape[-1] / sr)
        audio = julius.resample_frac(audio, int(sr), int(sample_rate))
        audio = torch.nn.functional.pad(
            audio, (int(before_padding * sample_rate), int(after_padding * sample_rate))
        )
        max_len = max(max_len, audio.shape[-1])
        batch.append(audio)

    target = max_len
    if target % audio_encoder.frame_size != 0:
        target = target + (
            audio_encoder.frame_size - max_len % audio_encoder.frame_size
        )
    padded_batch = torch.stack(
        [
            torch.nn.functional.pad(audio, (0, target - audio.shape[-1]))
            for audio in batch
        ]
    )
    return padded_batch


@torch.no_grad
def streaming_transcribe(
    padded_batch: torch.Tensor,
    mimi,
    lm_gen,
):
    bsz = padded_batch.shape[0]

    text_tokens_acc = []

    with mimi.streaming(bsz), lm_gen.streaming(bsz):
        for offset in range(0, padded_batch.shape[-1], mimi.frame_size):
            audio_chunk = padded_batch[:, offset : offset + mimi.frame_size]
            audio_chunk = audio_chunk[:, None, :]

            audio_tokens = mimi.encode(audio_chunk)
            text_tokens = lm_gen.step(audio_tokens)
            if text_tokens is not None:
                text_tokens_acc.append(text_tokens)

    return torch.concat(text_tokens_acc, axis=-1)


def run_inference(
    dataset,
    mimi,
    lm_gen,
    tokenizer,
    padding_token_id,
    before_padding_sec,
    after_padding_sec,
):
    metrics = AsrMetrics()
    audio_time = 0.0
    inference_timer = Timer()

    for batch in tqdm.tqdm(dataset.iter(args.batch_size)):
        audio_data = list(
            zip(
                [torch.tensor(x["array"]).float() for x in batch["audio"]],
                [x["sampling_rate"] for x in batch["audio"]],
            )
        )

        audio_time += sum(audio.shape[-1] / sr for (audio, sr) in audio_data)

        gt_transcripts = batch["original_text"]

        padded_batch = get_padded_batch(
            audio_data,
            before_padding=before_padding_sec,
            after_padding=after_padding_sec,
            audio_encoder=mimi,
        )
        padded_batch = padded_batch.cuda()

        with inference_timer:
            text_tokens = streaming_transcribe(
                padded_batch,
                mimi=mimi,
                lm_gen=lm_gen,
            )

        for batch_index in range(text_tokens.shape[0]):
            utterance_tokens = text_tokens[batch_index, ...]
            utterance_tokens = utterance_tokens[utterance_tokens > padding_token_id]
            text = tokenizer.decode(utterance_tokens.cpu().numpy().tolist())
            metrics.update(hyp=text, ref=gt_transcripts[batch_index])

    return metrics, inference_timer.total, audio_time


def main(args):
    torch.set_float32_matmul_precision("high")

    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.moshi_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )

    mimi = info.get_mimi(device=args.device)
    tokenizer = info.get_text_tokenizer()
    lm = info.get_moshi(
        device=args.device,
        dtype=torch.bfloat16,
    )
    lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
    dataset = get_dataset(args)

    padding_token_id = info.raw_config.get("text_padding_token_id", 3)
    # Putting in some conservative defaults
    audio_silence_prefix_seconds = info.stt_config.get(
        "audio_silence_prefix_seconds", 1.0
    )
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 5.0)

    wer_metric, inference_time, audio_time = run_inference(
        dataset,
        mimi,
        lm_gen,
        tokenizer,
        padding_token_id,
        audio_silence_prefix_seconds,
        audio_delay_seconds + 0.5,
    )

    print(wer_metric, f"RTF = {audio_time / inference_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example streaming STT inference.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASET_MAP.keys(),
        help="Dataset to run inference on.",
    )

    parser.add_argument(
        "--hf-repo", type=str, help="HF repo to load the STT model from. "
    )
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to a local checkpoint file."
    )
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to a local config file.", default=None
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
        default=32,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument("--hf-cache-dir", type=str, help="HuggingFace cache folder.")
    args = parser.parse_args()

    main(args)
