"""Wrapper around a LMModel to make it work as a TTS Engine when the text has sufficient
delay ahead of the audio."""
from dataclasses import dataclass
import logging
import math
import random
import re
import typing as tp

import omegaconf
import sentencepiece
import torch

from ..data.audio_utils import convert_audio
from .encodec import CompressionModel
from .lm import LMModel, _undelay_sequence, ConditionType
from ..modules.transformer import set_attention_context
from ..environment import AudioCraftEnvironment
from ..utils.autocast import TorchAutocast

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """
    In the following `Ka` is the number of audio codebooks, `K = 1 + Ka` is the total number of codebooks.
    Args:
        raw_tokens: all tokens generated, of shape [B, K, T], up to the max gen len used for this batch.
            used for debugging, you should rather use the `tokens` attribute that provide
            per item, properly trimmed values.
        audio_tokens: list of tensors of shape [Ka, T_k] containing the generated audio tokens for each input.
            Those are properly trimmed to exclude the first `delay` seconds, and stops at the end of the
            generation for that input.
        transcripts: for each item in the batch, list of tuples (word, timestamp).
        segmentations: same but for indicating the start of a new speaker with a list of tuples (speaker, timestamp).
    """
    raw_tokens: torch.Tensor
    audio_tokens: list[torch.Tensor]
    transcripts: list[list[tuple[str, float]]]
    segmentations: list[list[tuple[str, float]]]
    consumption_times: list[list[int]]

    @staticmethod
    def from_raw_tokens(tts_model: 'TTSModel', raw_tokens: torch.Tensor,
                        generation_lengths: list[int | None],
                        consumption_times: list[list[int]]) -> 'TTSResult':
        BOS = tts_model.tokenizer.bos_id()
        EOS = tts_model.tokenizer.eos_id()
        PAD = tts_model.model.text_padding_token_id
        EPAD = tts_model.model.end_of_text_padding_id
        frame_rate = tts_model.compression.frame_rate
        out = TTSResult(raw_tokens, [], [], [], consumption_times)

        for b in range(len(raw_tokens)):
            generation_length = generation_lengths[b]
            if generation_length is None:
                generation_length = raw_tokens.shape[-1]
            tokens = raw_tokens[b, :, :generation_length + tts_model.total_extra_steps]
            audio_tokens = tokens[1:, tts_model.delay_steps:]
            transcript: list[tuple[str, float]] = []
            segmentation: list[tuple[str, float]] = []
            out.audio_tokens.append(audio_tokens)
            out.transcripts.append(transcript)
            out.segmentations.append(segmentation)
            speaker = 'SPEAKER_OTHER'
            segment_start = 0.
            word_start = 0.
            current_tokens = []
            for tok_idx, tok in enumerate(tokens[0].tolist()):
                timestamp = tok_idx / frame_rate
                if tok == BOS:
                    if tok_idx > 0:
                        segmentation.append((speaker, segment_start))
                    speaker = 'SPEAKER_MAIN'
                    segment_start = timestamp
                elif tok == EOS:
                    assert speaker == 'SPEAKER_MAIN'
                    segmentation.append((speaker, segment_start))
                    speaker = 'SPEAKER_OTHER'
                    segment_start = timestamp
                elif tok in [PAD, EPAD]:
                    continue
                current_tokens.append(tok)
                word = tts_model.tokenizer.decode(current_tokens)
                if ' ' in word:
                    word = tts_model.tokenizer.decode(current_tokens[:-1])
                    assert ' ' not in word, (word, current_tokens)
                    transcript.append((word, word_start))
                    word_start = timestamp
                    current_tokens = [tok]
            if current_tokens:
                word = tts_model.tokenizer.decode(current_tokens)
                transcript.append((word, word_start))
            segmentation.append((speaker, segment_start))
        return out


@dataclass
class TTSModel:
    """TTS Model wrapping a compression model, text tokenizer, and a LM.

    Args:
        compression: Compression model used to convert to/from audio tokens.
        model: the language model itself, should be trained with interleaved data
            and a positive delay.
        tokenizer: text tokenizer.
        delay: delay in seconds of the audio with respect to the text.
        device: device to use.
        padding_bonus: bonus added to the padding tokens when they go under
            the target ratio.
        padding_target_ratio: target ratio of padding tokens.
        padding_bonus_min_ema_fraction: ensure we have enough steps to compute the average.
        padding_ema_decay: decay of the exponential moving average of the number of paddings.
        temp: temperature for sampling.
        top_k: top k for sampling.
        extra_steps: extra steps to generate after the last text token, on top of the delay.
        max_paddings: maximum number of padding tokens before forcing going to the next word.

    """
    compression: CompressionModel
    model: LMModel
    tokenizer: sentencepiece.SentencePieceProcessor
    delay: float
    device: torch.device

    padding_bonus: float = 4.
    padding_target_ratio: float = 0.65
    padding_bonus_min_ema_fraction: float = 0.25
    padding_ema_decay: float = 0.96

    temp: float = 0.8
    top_k: int = 250
    extra_steps: int = 0
    max_paddings: int = 16

    _whisper: object | None = None

    def preprocess_audio(self, wav: torch.Tensor, sample_rate: int,
                         is_main: bool = True, language: str = 'en',
                         keep_final_blank: bool = False, trim_initial_text: bool = True,
                         open_main: bool = True, close_main: bool = True) -> torch.Tensor:
        import whisper_timestamped as whisper

        BOS = self.tokenizer.bos_id()
        EOS = self.tokenizer.eos_id()
        PAD = self.model.text_padding_token_id
        EPAD = self.model.end_of_text_padding_id

        wav = wav.to(self.device)
        wav_for_tr = convert_audio(wav, sample_rate, 16000, 1)
        if self._whisper is None:
            self._whisper = whisper.load_model('large-v3', device=self.device)
        assert isinstance(self._whisper, whisper.model.Whisper)
        wav_for_comp = convert_audio(wav, sample_rate, self.compression.sample_rate, self.compression.channels)
        with torch.no_grad():
            audio_tokens, _ = self.compression.encode(wav_for_comp[None])
            audio_tokens = audio_tokens[0]
        transcript = whisper.transcribe(self._whisper, wav_for_tr[0], language=language, vad=True, verbose=None)
        if trim_initial_text:
            text_delay = self.delay
        else:
            text_delay = 0
            pad_audio = torch.full((audio_tokens.shape[0], self.delay_steps),
                                   self.model.ungenerated_token_id,
                                   device=self.device, dtype=torch.long)
            audio_tokens = torch.cat([pad_audio, audio_tokens], dim=-1)
        text_tokens = [PAD] * (audio_tokens.shape[-1])
        first = True
        eos_location = None
        last_write_location = 0
        for segment in transcript['segments']:
            for word in segment['words']:
                text = word['text'].strip()
                start = word['start'] - text_delay
                end = word['end'] - text_delay
                if start < 0:
                    continue
                word_tokens = self.tokenizer.encode(text)
                start_step = int(start * self.compression.frame_rate)
                end_step = int(end * self.compression.frame_rate)
                if start_step >= len(text_tokens):
                    continue
                if is_main and first:
                    first = False
                    if open_main:
                        if start_step == 0:
                            word_tokens.insert(0, self.tokenizer.bos_id())
                        else:
                            text_tokens[start_step - 1] = BOS
                to_write = min(len(word_tokens), len(text_tokens) - start_step)
                text_tokens[start_step:start_step + to_write] = word_tokens[:to_write]
                eos_location = max(end_step, start_step + to_write)
                eos_location = min(eos_location, len(text_tokens) - 1)
                if start_step > 0 and text_tokens[start_step - 1] == PAD:
                    text_tokens[start_step - 1] = EPAD
                last_write_location = start_step + to_write
        if is_main and eos_location is not None and close_main:
            text_tokens[eos_location] = EOS
            last_write_location = eos_location
        text_tokens_tensor = torch.tensor(text_tokens, device=self.device, dtype=torch.long)[None]
        if not keep_final_blank:
            blank_starts = max(len(text_tokens) - self.delay_steps, last_write_location)
            text_tokens_tensor[:, blank_starts:] = self.model.ungenerated_token_id
        tokens = torch.cat([text_tokens_tensor, audio_tokens], dim=0)
        return tokens

    def preprocess(self, turns: list[str], start_with_main: bool = False,
                   open_main: bool = True, close_main: bool = True) -> list[list[int]]:
        """Preprocess a list of turns into a TTSInput.
        Two speakers are supported: `SPEAKER_OTHER` and `SPEAKER_MAIN`.
        `SPEAKER_MAIN` will be surrounded with BOS/EOS. `turns` starts with
        `SPEAKER_OTHER`, by default, unless `start_with_main` is true.

        Each turn will be splitted into words. This method will not cleanup the text,
        you should properly call `cleanup_text` or something else before hand.

        To include padding manually, insert a special words containing  e.g. `'\x03\x00'`,
        with `\x03` used for padding and `\x00` for end of text padding.
        Those will be only processed if they are in their own word. No extra ' ' will be inserted.
        This can be useful to force the model to go slowly, especially for input slightly out of domain.
        For instance `'Hello \x03\x03 Bob! \x03\x00'`.
        """
        BOS = self.tokenizer.bos_id()
        EOS = self.tokenizer.eos_id()
        PAD = self.model.text_padding_token_id
        EPAD = self.model.end_of_text_padding_id
        pad_map = {'\x00': EPAD, '\x03': PAD}
        chunks: list[list[int]] = []
        re_pad = re.compile('^([\x03\x00]*)([^\x03\x00]*)([\x03\x00]*)$')
        for turn_idx, turn in enumerate(turns):
            if not turn:
                continue
            is_main_speaker = (turn_idx % 2) == 1 - int(bool(start_with_main))
            any_word_chunk = False
            for word in turn.split(' '):
                if not word:
                    continue
                match = re_pad.match(word)
                assert match is not None
                pre = [pad_map[char] for char in match.group(1)]
                mid = []
                if match.group(2):
                    mid = self.tokenizer.encode(match.group(2))

                if mid:
                    if is_main_speaker and not any_word_chunk and open_main:
                        mid.insert(0, BOS)
                    any_word_chunk = True
                post = [pad_map[char] for char in match.group(3)]
                chunks.append(pre + mid + post)
            if is_main_speaker and any_word_chunk and close_main:
                chunks.append([EOS])
        return chunks

    @staticmethod
    def from_solver(solver, **kwargs) -> 'TTSModel':
        # Ugly hack to avoid too many circular imports in this file.
        from ..solvers.musicgen import MusicGenSolver
        assert isinstance(solver, MusicGenSolver)
        typed_solver: MusicGenSolver = solver
        compression = typed_solver.compression_model
        model = solver.model
        assert solver.moshi_text_lm is not None
        tokenizer = solver.moshi_text_lm.tokenizer
        device = torch.device(solver.device)
        delay = solver.xp.cfg.multimodal.interleaver.delay
        return TTSModel(compression, model, tokenizer, delay, device, **kwargs)

    @staticmethod
    def from_sig(sig: str, epoch: int | None = None, compression: str | None = None,
                 device: torch.device = torch.device('cuda'),
                 **kwargs) -> 'TTSModel':
        """Builds a TTSModel from a given XP signature and optional epoch number.
        Allows to override the compression model if required."""
        from . import builders
        from .. import train
        from ..solvers.compression import CompressionSolver
        xp = train.main.get_xp_from_sig(sig)
        xp.cfg.device = str(device)
        if xp.cfg.fsdp.use:
            xp.cfg.dtype = xp.cfg.fsdp.param_dtype
        elif xp.cfg.autocast:
            xp.cfg.dtype = xp.autocast_dtype
        compression_checkpoint = compression or xp.cfg.compression_model_checkpoint
        compression_model = CompressionSolver.wrapped_model_from_checkpoint(xp.cfg, compression_checkpoint).to(device)
        if xp.cfg.compression_model_n_q:
            compression_model.set_num_codebooks(xp.cfg.compression_model_n_q)
        tokenizer = sentencepiece.SentencePieceProcessor(
            str(AudioCraftEnvironment.resolve_reference_path(xp.cfg.dataset.text.tokenizer_path)))
        model = builders.get_lm_model(xp.cfg, n_q=compression_model.num_codebooks,
                                      cardinality=compression_model.cardinality)
        model.eval()
        if epoch:
            ckpt = xp.folder / f'checkpoint_{epoch}.th'
        else:
            ckpt = xp.folder / 'checkpoint.th'
        pkg = torch.load(ckpt, 'cpu')
        if xp.cfg.fsdp.use:
            model.load_state_dict(pkg['fsdp_best_state']['model'])
        else:
            model.load_state_dict(pkg['best_state']['model'])
        dtype = getattr(torch, xp.cfg.dtype)
        model.autocast = TorchAutocast(enabled=True, dtype=dtype, device_type=device.type)

        delay = xp.cfg.multimodal.interleaver.delay

        return TTSModel(compression_model, model, tokenizer, delay, torch.device(device), **kwargs)

    @property
    def delay_steps(self) -> int:
        """Delay expressed in timesteps."""
        return int(math.ceil(self.delay * self.compression.frame_rate))

    @property
    def total_extra_steps(self) -> int:
        """Number of extra generation steps to perform after the last text token."""
        return self.delay_steps + self.extra_steps

    def _logits_hook(self, module, input, output: tuple[torch.Tensor | None, torch.Tensor | None]
                     ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Add the padding bonus.
        assert self.model._is_streaming
        logits: torch.Tensor
        text_logits: torch.Tensor
        logits, text_logits = output  # type: ignore
        if text_logits is not None:
            # retrieving the moving average of the number of paddings.
            count = self.model._streaming_state.get('pad_count')
            if count is not None:
                text_logits = text_logits.float()
                # Total is the normalization factor. In the permanent regime, it should be
                # really close to 1. In particular we will check it is >= padding_bonus_min_ratio
                # to ensure we are computed the average on enough steps.
                total = self.model._streaming_state['pad_total']
                ratio = count / total
                one = torch.ones(1, device=text_logits.device, dtype=text_logits.dtype)
                bonus = torch.where(
                    (total > self.padding_bonus_min_ema_fraction) & (ratio < self.padding_target_ratio),
                    self.padding_bonus * one, 0 * one)
                bonus = bonus.view(-1, 1, 1)
                text_logits[:] = torch.log_softmax(text_logits, dim=-1)
                text_logits[..., self.model.text_padding_token_id] += bonus
                text_logits[..., self.model.end_of_text_padding_id] += bonus * 0.5
        return logits, text_logits

    def generate(self,
                 tts_inputs: list[list[torch.Tensor | list[int]]],
                 max_gen_len: int = 5000,
                 callback=None) -> TTSResult:
        """
        Args:
            tts_inputs: one list per item to generate. Each item should be a list whoses elements
                are either tensors obtained from `preprocess_audio` or lists of integers obtained
                from `prepocess`. The list will be modified during the call,
                e.g. it will be empty at the end of this call.
            max_gen_len: maximum number of tokens to generate.
            callback: function with signature `f(current_step, total_steps)`.

        Returns:
            TTSResult
        """
        handle = self.model.register_forward_hook(self._logits_hook)
        try:
            return self._generate(tts_inputs, max_gen_len, callback)
        finally:
            handle.remove()

    @torch.no_grad()
    def _generate(self,
                  tts_inputs: list[list[torch.Tensor | list[int]]],
                  max_gen_len: int = 5000,
                  callback=None) -> TTSResult:
        model = self.model
        BOS = self.tokenizer.bos_id()
        EOS = self.tokenizer.eos_id()
        PAD = model.text_padding_token_id
        EPAD = model.end_of_text_padding_id
        num_samples = len(tts_inputs)

        initial = model._get_initial_token('both').expand(num_samples, -1, -1)
        max_delay = max(model.delays)  # with delays, we need to generate a few more time steps.
        ungenerated = model.ungenerated_token_id  # special value to indicate tokens to generate
        gen_sequence = torch.full((num_samples, model.num_codebooks, max_gen_len + max_delay + 1),
                                  ungenerated, device=self.device, dtype=torch.long)

        # We do not support conditioning here, setting conditions to {}.
        cfg_conditions: dict[str, ConditionType] = {}
        # special token for the beginning of the sequence.
        gen_sequence[:, :, :1] = initial
        start_offset = 0
        last_offset_written = 0

        next_generated_indexes: list[int] = [1] * num_samples
        remaining_paddings: list[int] = [self.max_paddings] * num_samples
        generation_lengths: list[int | None] = [None] * num_samples
        consumption_times: list[list[int]] = [[] for _ in range(num_samples)]

        set_attention_context(model.transformer, model.context)
        with model.streaming(), model.autocast:
            model._streaming_state['pad_count'] = torch.zeros(
                num_samples, dtype=torch.float, device=self.device)
            model._streaming_state['pad_total'] = torch.zeros(
                num_samples, dtype=torch.float, device=self.device)
            unconditional_state = dict(model.get_streaming_state())  # useless here but required.
            for offset in range(start_offset, max_gen_len + max_delay):
                if callback is not None:
                    callback(offset, max_gen_len + max_delay)
                if all([x is not None for x in generation_lengths]):
                    mx: int = max(generation_lengths)  # type: ignore
                    if offset >= mx + max_delay + self.total_extra_steps:
                        break
                # `offset` measures position in the output tensor with no delays.
                # In particular, there is a shift of 1 with the `gen_sequence` that includes
                # the initial empty token.
                if offset == start_offset:
                    input_ = gen_sequence[:, :, :offset + 1]
                else:
                    input_ = gen_sequence[:, :, offset:offset + 1]

                if offset > start_offset:
                    is_pad = (input_[:, 0, -1] == PAD) | (input_[:, 0, -1] == EPAD)
                    model._streaming_state['pad_count'] *= self.padding_ema_decay
                    model._streaming_state['pad_count'] += (1 - self.padding_ema_decay) * is_pad
                    model._streaming_state['pad_total'] *= self.padding_ema_decay
                    model._streaming_state['pad_total'] += (1 - self.padding_ema_decay)

                next_token, _ = model._sample_next_token(
                    input_, cfg_conditions, unconditional_state,
                    use_sampling=True, temp=self.temp, top_k=self.top_k, top_p=0,
                    text_or_audio='both')
                assert next_token.shape[-1] == 1
                next_token = next_token[:, 0, 0]   # shape is [B]

                for b in range(len(tts_inputs)):
                    if generation_lengths[b] is not None:
                        continue
                    if offset + 1 < next_generated_indexes[b]:
                        continue
                    next_is = None
                    if tts_inputs[b]:
                        tokens = tts_inputs[b][0]
                        if isinstance(tokens, list):
                            next_is = tokens[0]
                        else:
                            next_is = int(tokens[0, 0].item())

                    next_is_bos = next_is == BOS
                    next_is_eos = next_is == EOS
                    next_is_pad = next_is == PAD
                    next_is_epad = next_is == EPAD
                    acceptable: set[int]
                    if next_is_bos or next_is_eos:
                        # BOS will replace the EPAD and is already in the tokens we will write.
                        # Similarly, EOS will always come before the EPAD and might replace it
                        acceptable = set([PAD])
                    elif not tts_inputs[b]:
                        # We no longer have words to feed, the only acceptable output is padding.
                        acceptable = set([PAD])
                    elif input_[b, 0, -1] == EPAD:
                        acceptable = set([])
                    elif next_is_pad:
                        acceptable = set([PAD])
                    elif next_is_epad:
                        acceptable = set([EPAD])
                    else:
                        acceptable = set([PAD, EPAD])
                    # We take over if:
                    # (i) this is the very first step.
                    # (ii) the model outputs too many padding tokens in a row.
                    # (iii) the model is trying to ouput something not acceptable e.g. a word.
                    if offset == 0 or remaining_paddings[b] < 0 or next_token[b].item() not in acceptable:
                        # Note in the following that `next_token` and `gen_sequence` point to the same memory
                        # so that modifying one will change the other too.
                        if tts_inputs[b]:
                            if input_[b, 0, -1] == PAD and not (next_is_bos or next_is_eos):
                                # It can happen that the model forgets to output an EPAD, which we fix for it.
                                logger.debug("[%d] Required manual EPAD at %d", b, offset + 1)
                                next_token[b] = EPAD
                            else:
                                tokens = tts_inputs[b].pop(0)
                                consumption_times[b].append(offset)
                                if isinstance(tokens, list):
                                    tokens = torch.tensor(tokens, device=gen_sequence.device,
                                                          dtype=gen_sequence.dtype).view(1, -1)
                                    next_ungen = tokens.shape[-1]
                                else:
                                    tokens = tokens.to(gen_sequence)
                                    ungens = (tokens[0] == ungenerated).nonzero()[:, 0]
                                    if ungens.numel():
                                        next_ungen = int(ungens[0].item())
                                    else:
                                        next_ungen = tokens.shape[-1]

                                K_tokens, T_tokens = tokens.shape
                                for k in range(K_tokens):
                                    delay = model.delays[k]
                                    start_from = offset + delay + 1
                                    to_write = min(T_tokens, gen_sequence.shape[-1] - start_from)
                                    gen_sequence[b, k, start_from:start_from + to_write] = tokens[k, :to_write]
                                # We store up to where we have teacher forced, and we won't interfere
                                # until we reach that offset.
                                next_generated_indexes[b] = offset + 1 + min(to_write, next_ungen)
                                remaining_paddings[b] = self.max_paddings
                        elif generation_lengths[b] is None:
                            generation_lengths[b] = offset + 1
                            remaining = sum(length is None for length in generation_lengths)
                            logger.debug('[%d] End of transcription, %d remaining', b, remaining)
                            gen_sequence[b, 0, offset + 1:] = PAD
                    else:
                        remaining_paddings[b] -= 1

                this_gen_step = gen_sequence[:, :, offset + 1]
                depformer_tokens: list[torch.Tensor] = []
                for cb_index in range(model.num_codebooks):
                    if cb_index == 0:
                        # No need to generate, `next_token` is actually the next text token.
                        # We just need to only keep the new token if the value wasn't provided
                        # in the prompt.
                        next_token = torch.where(this_gen_step[:, 0] == ungenerated,
                                                 next_token, this_gen_step[:, 0])
                    else:
                        input_ = next_token[:, None, None]
                        next_token, _ = model._sample_next_token(
                            input_, cfg_conditions, unconditional_state,
                            use_sampling=True, temp=self.temp, top_k=self.top_k, top_p=0,
                            text_or_audio='both')
                        assert next_token.shape[-1] == 1
                        next_token = next_token[:, 0, 0]   # shape is [B, K]
                        next_token = torch.where(this_gen_step[:, cb_index] == ungenerated,
                                                 next_token, this_gen_step[:, cb_index])

                        original_offset = offset - model.delays[cb_index]
                        if original_offset < 0:
                            # We are not currently generating this codebook, we replace with a special token.
                            next_token[:] = initial[:, cb_index, 0]
                    depformer_tokens.append(next_token)

                assert len(depformer_tokens) == model.num_codebooks, (len(depformer_tokens), model.num_codebooks)
                next_token = torch.stack(depformer_tokens, dim=1)
                assert next_token.shape == (num_samples, model.num_codebooks), next_token.shape
                gen_sequence[..., offset + 1] = next_token
                last_offset_written = offset + 1
        unconditional_state.clear()
        output, _ = _undelay_sequence(model.delays,
                                      gen_sequence[:, :, 1: last_offset_written + 1],
                                      fill_value=ungenerated)
        output = output[:, :, :last_offset_written - max_delay]
        assert (output != ungenerated).all()
        return TTSResult.from_raw_tokens(self, output, generation_lengths, consumption_times)


@dataclass
class EncoderDecoderTTSModel:
    """A simple utilitary class to wrap an encoder-decoder TTS model."""

    compression: CompressionModel
    model: LMModel
    cfg: omegaconf.DictConfig
    device: torch.device
    random_speaker_iterator: tp.Iterator[tp.Any] | None
    asr_pipeline: object | None

    batch_size: int = 64
    target_db: float = -25.0
    random_gain_db: float = 4
    language: str = "en"

    def __post_init__(self):
        self.sample_rate = self.compression.sample_rate

    @staticmethod
    def from_sig(
        sig: str,
        epoch: int | None = None,
        compression: str | None = None,
        random_speaker_dataset_path: str | None = None,
        random_speaker_dataset_seed: int | None = None,
        device: torch.device = torch.device("cuda"),
        with_asr: bool = False,
        **kwargs,
    ) -> "EncoderDecoderTTSModel":
        """Builds an EncoderDecoderTTSModel from a given XP signature and optional epoch number.
        Allows to override the compression model if required."""
        from copy import deepcopy

        from . import builders as model_builders, providers
        from .. import train
        from ..solvers import builders as solver_builders
        from ..solvers.compression import CompressionSolver

        xp = train.main.get_xp_from_sig(sig)
        xp.cfg.device = str(device)
        if xp.cfg.fsdp.use:
            xp.cfg.dtype = xp.cfg.fsdp.param_dtype
        elif xp.cfg.autocast:
            xp.cfg.dtype = xp.autocast_dtype
        compression_checkpoint = compression or xp.cfg.compression_model_checkpoint
        compression_model = CompressionSolver.wrapped_model_from_checkpoint(xp.cfg, compression_checkpoint).to(device)
        if xp.cfg.compression_model_n_q:
            compression_model.set_num_codebooks(xp.cfg.compression_model_n_q)

        model = model_builders.get_lm_model(
            xp.cfg, n_q=compression_model.num_codebooks, cardinality=compression_model.cardinality
        )
        model.eval()
        if epoch:
            ckpt = xp.folder / f"checkpoint_{epoch}.th"
        else:
            ckpt = xp.folder / "checkpoint.th"
        pkg = torch.load(ckpt, "cpu")
        if xp.cfg.fsdp.use:
            model.load_state_dict(pkg["fsdp_best_state"]["model"])
        else:
            model.load_state_dict(pkg["best_state"]["model"])
        dtype = getattr(torch, xp.cfg.dtype)
        model.autocast = TorchAutocast(enabled=True, dtype=dtype, device_type=device.type)

        random_speaker_iterator = None
        if random_speaker_dataset_path is not None:
            dummy_cfg = deepcopy(xp.cfg)
            dummy_cfg.datasource.train = random_speaker_dataset_path
            dummy_cfg.dataset.batch_size = 1
            dummy_cfg.seed = random_speaker_dataset_seed or dummy_cfg.seed
            train_loader = solver_builders.get_audio_datasets(dummy_cfg, solver_builders.DatasetType.SPEECH)["train"]
            random_speaker_iterator = iter(train_loader)

        asr_pipeline = None
        if with_asr:
            asr_pipeline = providers.ASRPipelineProvider.get_model(device=device)

        return EncoderDecoderTTSModel(
            compression_model, model, xp.cfg, device, random_speaker_iterator, asr_pipeline, **kwargs
        )

    def compute_cer(self, turns, generated_audio):
        import jiwer

        cer_normalization = jiwer.transforms.Compose(
            [
                jiwer.transforms.ToLowerCase(),
                jiwer.transforms.ExpandCommonEnglishContractions(),
                jiwer.transforms.RemoveKaldiNonWords(),
                jiwer.transforms.RemovePunctuation(),
                jiwer.transforms.RemoveWhiteSpace(replace_by_space=True),
                jiwer.transforms.RemoveMultipleSpaces(),
                jiwer.transforms.ReduceToListOfListOfChars(),
            ]
        )

        gt_transcript = " ".join([turn[1] for turn in turns])
        generated_transcript = self.asr_pipeline(
            convert_audio(generated_audio, self.sample_rate, 16000, 1)[0].numpy(),
            chunk_length_s=30,
            batch_size=24,
            generate_kwargs={"task": "transcribe", "language": self.language},
        )["text"]
        return jiwer.cer(
            gt_transcript,
            generated_transcript,
            reference_transform=cer_normalization,
            hypothesis_transform=cer_normalization,
        )

    def _sample_speaker_wav(self):
        speaker_wav = torch.zeros(1, 1)
        while torch.all(speaker_wav == 0):
            _, meta = next(self.random_speaker_iterator)
            speaker_wav = meta[0].speaker_wavs.wav[0, 0]
        return speaker_wav

    def _normalize(self, audio: torch.Tensor, random_gain: bool = False):
        vol = 10 * torch.log10(audio.pow(2).mean())
        tgt = self.target_db
        if random_gain:
            tgt += abs(self.random_gain_db) * random.uniform(-1, 1)
        rescale = 10 ** ((tgt - vol) / 20)
        return audio * rescale

    def _prepare_conditions(
        self, turns: list[tp.Tuple[torch.Tensor | str, str]], duration: float, start_with_main: bool, num_repeats: int
    ):
        from ..data.audio_dataset import AudioMeta
        from ..data.info_audio_dataset import SegmentInfoWithAttributes, SegmentInfo
        from ..conditioners.audio import WavCondition

        num_turns = len(turns)
        sample_rate = self.sample_rate
        m = AudioMeta(path="", sample_rate=sample_rate, duration=duration)
        si = SegmentInfo(
            meta=m,
            seek_time=0,
            n_frames=sample_rate * duration,
            total_frames=sample_rate * duration,
            sample_rate=sample_rate,
            channels=1,
        )
        wav = torch.zeros(1, 1, int(sample_rate * duration))
        # We repeat via a for loop to ensure random speakers (if any) are resampled between repetitions.
        repeated_conditions = []
        for _ in range(num_repeats):
            conditions = [
                SegmentInfoWithAttributes.from_wav_info(wav[0], si).to_condition_attributes() for _ in range(num_turns)
            ]

            random_speaker_ids = set([turn[0] for turn in turns if isinstance(turn[0], str)])
            if len(random_speaker_ids) > 0:
                assert (
                    self.random_speaker_iterator is not None
                ), "Random speaker conditioning requested but no iterator provided."
            random_speaker_wavs = {
                random_speaker_id: self._sample_speaker_wav() for random_speaker_id in random_speaker_ids
            }
            max_speakers = self.cfg.dataset.max_speakers
            speaker_conditioning_duration = self.cfg.dataset.max_speaker_conditioning_duration
            for turn_idx, (speaker_conditioning, text) in enumerate(turns):
                is_main_speaker = (turn_idx % 2) == 1 - int(bool(start_with_main))
                conditions[turn_idx]["text"]["diarized_transcript_in_segment"] = "<s00> " + text
                speaker_cond = torch.zeros((1, max_speakers, int(speaker_conditioning_duration * sample_rate)))
                if isinstance(speaker_conditioning, str):
                    speaker_wav = random_speaker_wavs[speaker_conditioning]
                else:
                    speaker_wav = speaker_conditioning
                speaker_wav = self._normalize(speaker_wav, random_gain=not is_main_speaker)
                speaker_cond[0, 0] = speaker_wav.squeeze(0)
                conditions[turn_idx]["wav"]["speaker_wavs"] = WavCondition(
                    wav=speaker_cond,
                    length=torch.Tensor([sample_rate * speaker_conditioning_duration]),
                    sample_rate=sample_rate,
                    path=[""],
                    seek_time=[0.0],
                )
            repeated_conditions.extend(conditions)
        return repeated_conditions

    def _get_null_conditions(self, conditions):
        """This function creates negative conditioning for speaker cloning: it contains all conditions except for the
        speaker conditioning. This way, increasing the classifier free guidance coef only affects speaker fidelity."""
        from copy import deepcopy
        from ..conditioners.base import nullify_wav

        null_cond = deepcopy(conditions)
        for cond, actual_cond in zip(null_cond, conditions):
            cond.wav["speaker_wavs"] = nullify_wav(actual_cond.wav["speaker_wavs"])
        return null_cond

    def _basic_callback(self, curr_step: int, total_steps: int):
        print(f"{curr_step: 5d} / {total_steps: 5d}", end="\r")

    @torch.no_grad()
    def generate(
        self,
        turns: list[tp.Tuple[torch.Tensor | str, str]],
        duration: float,
        start_with_main: bool,
        silence_padding: float = 0.0,
        callback: tp.Optional[tp.Callable[[int, int], None]] = None,
        temp: float = 0.8,
        top_k: int = 250,
        cfg_coef: float = 5.0,
        return_stereo=False,
        **kwargs,
    ):
        """Generate audio from a list of (speaker_conditioning, text) tuples.

        Each speaker_conditioning is either a [1, T] waveform, or a string. If a string, it will be replaced by a random
        speaker conditioning from self.random_speaker_iterator. To ensure consistency of speaker identity along a
        script, every string conditioning with the same value will result in the same waveform conditioning.

        Example:
            [
                "jack", "Hello, how are you?",
                "firmin", "I'm fine, thanks!",
                "jack", "Goodbye!"
            ]

        The above example will use the same conditioning for the first and the third turns.

        Args:
            turns (list): list of (speaker_conditioning, text) tuples.
            duration (int):
            start_with_main (bool): whether the first speaker is the main speaker. Used at the moment to apply a
                constant gain rather than a variable one.
            silence_padding (float): duration of silence to add between turns.
            callback (callable): function to call at each generation step.
            temp (float): temperature for sampling.
            top_k (int): top k for sampling.
            cfg_coef (float): coefficient for the classifier-free guidance.
            return_stereo (bool): whether to return stereo audio.
            **kwargs: additional arguments to pass to the model's generate method.

        """
        num_turns = len(turns)
        num_repeats = max(1, self.batch_size // num_turns)
        num_samples = num_repeats * num_turns
        conditions = self._prepare_conditions(turns, duration, start_with_main, num_repeats)
        with self.model.autocast:
            tokens = self.model.generate(
                None,
                conditions=conditions,
                num_samples=num_samples,
                get_null_conditions=self._get_null_conditions,
                max_gen_len=int(duration * self.compression.frame_rate),
                callback=callback,
                temp=temp,
                top_k=top_k,
                cfg_coef=cfg_coef,
                **kwargs,
            )
        output_audios = [audio.float().cpu() for audio in self.compression._decode_to_list(tokens, None)]
        if silence_padding != 0.0:
            output_audios = [
                torch.cat([output_audio, torch.zeros(1, 1, int(silence_padding * self.sample_rate))], dim=-1)
                for output_audio in output_audios[:-1]
            ] + [output_audios[-1]]
        if return_stereo:
            for turn_idx, output_audio in enumerate(output_audios):
                is_main_speaker = (turn_idx % 2) == 1 - int(bool(start_with_main))
                output_audio = torch.cat([output_audio, torch.zeros_like(output_audio)], dim=1)
                output_audio = torch.roll(output_audio, 1 if not is_main_speaker else 0, dims=1)
                output_audios[turn_idx] = output_audio
        output_candidates = []
        for i in range(num_repeats):
            output_candidate = torch.cat(output_audios[i * num_turns:(i + 1) * num_turns], dim=-1)[0]
            # We reject sequences < 1 second.
            if output_candidate.shape[-1] >= self.sample_rate:
                output_candidates.append(output_candidate)
        cers = [self.compute_cer(turns, output_candidate) for output_candidate in output_candidates]
        sorted_outputs = sorted(zip(output_candidates, cers), key=lambda x: x[1])
        return sorted_outputs
