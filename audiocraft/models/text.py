"""Anything related to text generation with a multi modal model."""
from functools import reduce
import logging
import typing as tp

import sentencepiece
import torch

from .. import train
from ..data.text_dataset import DEFAULT_TOKENIZER_PATH
from ..environment import AudioCraftEnvironment
from ..models import builders, LMModel
from ..utils.autocast import TorchAutocast
from ..utils.utils import cross_entropy


logger = logging.getLogger(__name__)


def tokenize(tokenizer: sentencepiece.SentencePieceProcessor, text: str, bos: bool = True):
    """Tokenize the given string, accounting for new lines, potentially adding a BOS token."""
    nl_piece = tokenizer.encode('\n')[-1]
    tokens = tokenizer.encode(text.split('\n'))
    tokens = reduce(lambda a, b: a + [nl_piece] + b, tokens)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return tokens


class MoshiTextLM:
    """Wrapper around the LMModel to work with text.

    Args:
        model (LMModel): LMModel to wrap.
        tokenizer_path (str): path to the sentencepiece tokenizer.
        add_bos (bool): if True, will add the BOS token at the beginning of the sequences.
    """
    def __init__(self, model: LMModel, tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
                 add_bos: bool = False) -> None:
        super().__init__()
        self.model = model
        self.device = next(iter(self.model.parameters())).device
        self.add_bos = add_bos
        tokenizer_path = str(AudioCraftEnvironment.resolve_reference_path(tokenizer_path))
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=tokenizer_path, num_threads=1)

    @staticmethod
    def from_checkpoint(sig: str, epoch: int = 0, device='cuda', **kwargs) -> 'MoshiTextLM':
        """Create a MoshiLM from a signature and epoch."""
        from ..solvers.compression import CompressionSolver
        xp = train.main.get_xp_from_sig(sig)
        dtype = xp.cfg.fsdp.param_dtype
        xp.cfg.dtype = dtype
        xp.cfg.device = device
        audio_tokenizer = CompressionSolver.model_from_checkpoint(xp.cfg.compression_model_checkpoint)
        # We need to load the audio tokenizer to get the number of codebooks and cardinality, even
        # if we do not use it...
        n_q = xp.cfg.compression_model_n_q or audio_tokenizer.num_codebooks
        # The following flags will speed up the creation of the model by skipping the initialization.
        xp.cfg.transformer_lm.weight_init = None
        xp.cfg.transformer_lm.depthwise_init = None
        xp.cfg.transformer_lm.zero_bias_init = None
        model = builders.get_lm_model(xp.cfg, n_q=n_q, cardinality=audio_tokenizer.cardinality)
        model.eval()
        logger.info("Model ready")
        suffix = ""
        if epoch:
            suffix = f"_{epoch}"
        ckpt_name = f"checkpoint{suffix}.th"
        ckpt = xp.folder / ckpt_name
        logger.info("Loading checkpoint %s", ckpt)
        pkg = torch.load(ckpt, 'cpu')
        model.load_state_dict(pkg['fsdp_best_state']['model'])
        model.autocast = TorchAutocast(enabled=True, dtype=getattr(torch, dtype), device_type='cuda')
        tokenizer_path = xp.cfg.dataset.text.tokenizer_path
        return MoshiTextLM(model, tokenizer_path=tokenizer_path, **kwargs)

    def get_logits(self, sequences: tp.List[str]
                   ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, tp.List[int]]:
        """Returns tensor with one cross entropy per sequence (unnormalized)."""
        all_tokenized = [tokenize(self.tokenizer, s, self.add_bos) for s in sequences]
        lengths = [len(t) for t in all_tokenized]
        T = max(lengths)
        B = len(all_tokenized)
        tokens = torch.empty(B, self.model.num_codebooks, T, dtype=torch.long, device=self.device)
        tokens[:, :] = self.model.zero_token_id
        mask = torch.zeros(B, 1, T, dtype=torch.bool, device=self.device)
        for idx, one_tokenized in enumerate(all_tokenized):
            tokens[idx, 0, :len(one_tokenized)] = torch.tensor(one_tokenized, dtype=torch.long, device=self.device)
            mask[idx, 0, :len(one_tokenized)] = True

        with torch.no_grad():
            res = self.model.compute_predictions(tokens, [], {}, text_or_audio='text')
        assert res.text_mask is not None
        assert res.text_logits is not None
        return tokens[:, 0], res.text_logits[:, 0], mask[:, 0], lengths

    def get_scores(self, sequences: tp.List[str]) -> torch.Tensor:
        """Returns tensor with one cross entropy per sequence (unnormalized)."""
        tokens, logits, mask, _ = self.get_logits(sequences)
        result = cross_entropy(logits[:, None], tokens[:, None], mask[:, None])
        return result.sum(dim=2)[:, 0].cpu()
