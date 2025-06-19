import dataclasses
import julius
import sphn
import argparse

import torch
import moshi.models
import tqdm
import time


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


def run_inference(dataset, mimi, lm_gen, tokenizer, padding_token_id):
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
            before_padding=BEFORE_PADDING_SEC,
            after_padding=AFTER_PADDING_SEC,
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



@dataclasses.dataclass
class TimestampedText:
    text: str
    timestamp: tuple[float, float]

def tokens_to_timestamped_text(
    text_tokens,
    tokenizer,
    frame_rate,
    end_of_padding_id,
    padding_token_id
):
    assert text_tokens.ndim == 1, 'Only 1D tensors supported'
    text_tokens = text_tokens.clone()

    # Normally `end_of_padding` tokens indicate word boundaries.
    # Everything between them should be a single word;
    # the time offset of the those tokens correspond to word start and
    # end timestamps.
    # 
    # However, in rare cases some complexities could arise. Firstly,
    # for words that are said quickly but are represented by 
    # multiple tokens, the boundary might be omitted. Secondly,
    # for the very last word the end boundary might not happen.
    # Hence here we have a bit more involved code to handle those
    # situations.

        # eot
        # eos
        # min(duration of audio, duration on 3s)

    sequence_timestamps = []

    def _decode(t):
        t = t[t > padding_token_id]
        return tokenizer.decode(t.cpu().numpy().tolist())
    
    def _decode_segment(start, end):
        nonlocal text_tokens
        nonlocal sequence_timestamps

        text = _decode(text_tokens[start:end])
        words_inside = text.split()

        if len(words_inside) == 0:
            return
        if len(words_inside) == 1:
            # Single word within the boundaries, the general case
            sequence_timestamps.append(TimestampedText(text=text, timestamp=(start / frame_rate, end / frame_rate)))
        else:
            # We're in a rare situation where multiple words are so close they are not separated by `end_of_padding`.
            # We tokenize words one-by-one and assign sequential frames to their tokens.
            for adjacent_word in words_inside[:-1]:
                n_tokens = len(tokenizer.encode(adjacent_word))
                sequence_timestamps.append(
                    TimestampedText(text=adjacent_word,
                                    timestamp=(start / frame_rate, (start + n_tokens) / frame_rate)))

                start += n_tokens

            # The last word takes everything until the boundary
            adjacent_word = words_inside[-1]
            sequence_timestamps.append(
                TimestampedText(text=adjacent_word, timestamp=(start / frame_rate, end / frame_rate)))

    segment_boundaries, = torch.where(text_tokens == end_of_padding_id)

    if not segment_boundaries.numel():
        return []

    for i in range(len(segment_boundaries) - 1):
        segment_start = int(segment_boundaries[i]) + 1
        segment_end = int(segment_boundaries[i + 1])

        segment = text_tokens[segment_start: segment_end]
        _decode_segment(segment, segment_start, segment_end)

    last_segment_start = segment_boundaries[-1] + 1
    # at most, the last segment should go to the end of the stream
    text_tokens[-1] = end_of_padding_id
    boundary_tokens = torch.tensor([end_of_padding_id, tokenizer.eos_id()])
    end_of_last_segment, = torch.where(torch.isin(text_tokens[last_segment_start:],
                                                                 boundary_tokens))

    last_segment_end = last_segment_start + end_of_last_segment[0]
    _decode_segment(last_segment_start, last_segment_end)

    return sequence_timestamps



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

    audio, sample_rate = sphn.read(args.file)

    padding_token_id = info.raw_config.get("text_padding_token_id", 3)
    audio_silence_prefix_seconds = info.stt_config.get(
        "audio_silence_prefix_seconds", 1.0
    )
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 5.0)


    wer_metric, inference_time, audio_time = run_inference(
        dataset, mimi, lm_gen, tokenizer, padding_token_id
    )

    print(wer_metric, f"RTF = {audio_time / inference_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example streaming STT w/ timestamps.")
    parser.add_argument(
        "--file",
        required=True,
        help="File to transcribe.",
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
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    parser.add_argument("--hf-cache-dir", type=str, help="HuggingFace cache folder.")
    args = parser.parse_args()

    main(args)
