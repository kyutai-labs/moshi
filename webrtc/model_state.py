import os
import logging
import sentencepiece
import random
import torch
import time
import msh
import numpy as np

from livekit import rtc
from livekit.agents import utils


DEVICE = "cuda:0"

SAMPLE_RATE = msh.models.moshi.SAMPLE_RATE  # 24000
NUM_CHANNELS = 1
STEPS_PER_SEC = 12.5
SAMPLES_PER_CHANNEL = int(SAMPLE_RATE / STEPS_PER_SEC)  # 1920 (input t)
MAX_GEN_LENGTH = 2048

DEFAULT_MIMI_FILENAME = "tokenizer-de0e421d-checkpoint40.safetensors"
DEFAULT_TOKENIZER_FILENAME = "tokenizer_spm_32k_3.model"
DEFAULT_MOSHI_FILENAME = "mimi_0abbed5f@100.safetensors"


logger = logging.getLogger("moshi-agent")


def _seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_seed_all(42424242)


class ModelState:
    def __init__(
        self,
        ec: msh.models.EncodecModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: msh.models.LMModel,
    ):
        self.ec = ec
        self.text_tokenizer = text_tokenizer
        self.lm = lm

        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, SAMPLES_PER_CHANNEL
        )

    @staticmethod
    def load() -> "ModelState":
        """Load the model weights from the environment."""

        default_models_path = os.getenv("KYUTAI_MODELS", "/kyutai_models")
        default_mimi_path = os.path.join(default_models_path, DEFAULT_MIMI_FILENAME)
        default_tokenizer_path = os.path.join(
            default_models_path, DEFAULT_TOKENIZER_FILENAME
        )
        default_moshi_path = os.path.join(default_models_path, DEFAULT_MOSHI_FILENAME)

        mimi_path = os.getenv("MIMI_MODEL_PATH", default_mimi_path)
        tokenizer_path = os.getenv("TOKENIZER_MODEL_PATH", default_tokenizer_path)
        moshi_path = os.getenv("MOSHI_MODEL_PATH", default_moshi_path)

        logger.info(f"using mimi model: {mimi_path}")
        logger.info(f"using tokenizer model: {tokenizer_path}")
        logger.info(f"using moshi model: {moshi_path}")

        logger.info(f"sample rate: {SAMPLE_RATE}")
        logger.info(f"num channels: {NUM_CHANNELS}")
        logger.info(f"steps per second: {STEPS_PER_SEC}")
        logger.info(f"samples per channel: {SAMPLES_PER_CHANNEL}")

        logger.info("loading mimi model...")
        ec = msh.models.moshi.get_encodec(mimi_path, DEVICE)
        logger.info("mimi loaded")
        text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        logger.info("loading moshi model...")
        lm = msh.models.moshi.get_lm(moshi_path, DEVICE)
        logger.info("moshi loaded")
        return ModelState(ec, text_tokenizer, lm)

    def warmup(self):
        logger.info("warming up models...")
        self.lm.reset_streaming()
        self.ec.reset_streaming()
        lm_gen = msh.models.LMGen(self.lm, check=True, max_gen_len=64)
        with self.ec.streaming():
            while True:
                chunk = torch.zeros(1, 1, 1920, dtype=torch.float32, device=DEVICE)
                codes, _scale = self.ec.encode(chunk)
                main_pcm = None
                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[0, :, c].tolist())
                    if all([t < self.ec.cardinality for t in tokens[1:]]):
                        tokens = torch.tensor(tokens[1:], device=DEVICE).reshape(
                            (1, 8, 1)
                        )
                        main_pcm = self.ec.decode(tokens, scale=None)

                if main_pcm is not None:
                    break

    def reset(self):
        self.lm.reset_streaming()
        self.ec.reset_streaming()
        self.lm_gen = msh.models.LMGen(self.lm, check=True, max_gen_len=MAX_GEN_LENGTH)

    @torch.no_grad
    def __call__(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        assert frame.sample_rate == msh.models.moshi.SAMPLE_RATE, "invalid sample rate"
        assert frame.num_channels == 1, "invalid number of channels"

        start_time = time.perf_counter()

        frames = []
        texts = []

        for f in self._bstream.write(frame.data.tobytes()):
            int16_data = np.frombuffer(f.data, dtype=np.int16)
            float_data = int16_data.astype(np.float32) / 32768.0
            chunk = torch.tensor(float_data, device=DEVICE)[None, None]
            codes, _scale = self.ec.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[0, :, c].tolist())
                text_token = tokens[0]
                if text_token not in (0, 3):
                    _text = self.text_tokenizer.id_to_piece(text_token)
                    texts.append(_text)
                if all([t < self.ec.cardinality for t in tokens[1:]]):
                    tokens = torch.tensor(tokens[1:], device=DEVICE).reshape((1, 8, 1))
                    main_pcm = self.ec.decode(tokens, scale=None)
                    main_pcm_data = main_pcm.cpu().numpy()[0][0]
                    int16_data = (main_pcm_data * 32768).astype(np.int16)
                    frames.append(
                        rtc.AudioFrame(
                            data=int16_data.tobytes(),
                            sample_rate=SAMPLE_RATE,
                            num_channels=NUM_CHANNELS,
                            samples_per_channel=len(int16_data),
                        )
                    )

        dt = round((time.perf_counter() - start_time) * 1000, 2)

        frames_ms = 0
        for f in frames:
            frames_ms += round((f.samples_per_channel / f.sample_rate) * 1000, 2)

        logger.info(
            f"runned inference in {dt}ms {len(frames)} ({frames_ms}ms of audio) {texts}"
        )
        return frames
