# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import typing as tp

import torch
from torch import nn

from ..utils import utils
from ..utils.autocast import TorchAutocast
from ..modules.streaming import StreamingModule
from ..modules.transformer import (
    StreamingTransformer,
    create_norm_fn,
    set_attention_context,
)


logger = logging.getLogger(__name__)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y


def _undelay_sequence(
    delays: tp.List[int],
    tensor: torch.Tensor,
    fill_value: tp.Union[int, float] = float("NaN"),
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all([delay == 0 for delay in delays]):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: tp.Optional[torch.Tensor]  # [B, K, T, card]
    mask: tp.Optional[torch.Tensor]  # [B, K, T]
    text_logits: tp.Optional[torch.Tensor]  # [B, 1, T, text_card]
    text_mask: tp.Optional[torch.Tensor]  # [B, 1, T]


class LMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        autocast (TorchAutocast): autocast to use when evaluating the LM. This is better than
            wrapping calls to the LMModel with autocast, as this allows to exclude the conditioning
            computation.
        existing_text_padding_id (bool): if True, will use a different token for the initial text token, and
            the text padding token.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        delays: tp.List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_pos_emb: str = "sin",
        autocast: TorchAutocast = TorchAutocast(enabled=False),
        existing_text_padding_id: tp.Optional[int] = None,
        context: tp.Optional[int] = None,
        same_initial: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        self.same_initial = same_initial
        self.autocast = autocast
        kwargs["context"] = context
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        # Text card + padding token (if not in the original tokenizer)
        extra_text = self.existing_text_padding_id is None
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(text_card + 1, dim)
        self.text_linear = nn.Linear(dim, text_card + extra_text, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return 0

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token

    def forward_text(
        self,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(input_sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb

        transformer_out = self.transformer(input_)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        # Depformer doesn't care about past latent space of the transformers, in particular for the prompt.
        # We only need the timestep for which we need to provide a prediction.
        transformer_out = transformer_out[:, -1:]
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0:
            self.depformer.reset_streaming()
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            assert sequence.shape[2] == 1
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        logits = logits[:, :, -1, :].float()  # [B, K, card]
        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token


class LMGen(StreamingModule):
    def __init__(
        self,
        lm_model: LMModel,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        check: bool = False,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        lm_model.reset_streaming()
        lm_model._set_streaming(True)
        device = lm_model.device
        self.lm_model = lm_model
        self.max_gen_len = max_gen_len
        self.use_sampling = use_sampling
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.check = check
        self.ungenerated = (
            lm_model.ungenerated_token_id
        )  # special value to indicate tokens to generate
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.

        self.num_samples = 1
        self.initial = lm_model._get_initial_token().expand(self.num_samples, -1, -1)
        self.gen_sequence = torch.full(
            (
                self.num_samples,
                lm_model.num_codebooks,
                max_gen_len + self.max_delay + 1,
            ),
            self.ungenerated,
            device=device,
            dtype=torch.long,
        )
        for cb_idx, delay in enumerate(lm_model.delays):
            for i in range(1 + delay):
                self.gen_sequence[:, cb_idx, i] = self.initial[:, cb_idx, 0]
        self.zero = torch.full(
            [1], lm_model.zero_token_id, device=device, dtype=torch.long
        )
        self.audio_offset = lm_model.audio_offset
        set_attention_context(lm_model.transformer, lm_model.context)
        self.offset = 0
        self.depformer_graph = None

    @torch.no_grad()
    def step(
        self,
        input_tokens: tp.List[int],
    ) -> tp.Union[None, tp.List[int]]:
        lm_model = self.lm_model
        # `offset` measures position in the output tensor with no delays.
        # In particular, there is a shift of 1 with the `gen_sequence` that includes
        # the initial empty token.
        logger.debug("Offset %d / %d", self.offset, self.max_gen_len + self.max_delay)

        for k, tok in enumerate(input_tokens):
            kk = lm_model.dep_q + 1 + k
            delay = lm_model.delays[kk]
            idx = self.offset + delay
            if self.gen_sequence[:, kk, idx] == self.ungenerated:
                self.gen_sequence[:, kk, idx] = tok

        input_ = self.gen_sequence[:, :, self.offset : self.offset + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == self.ungenerated).any(), (self.offset, input_)
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        transformer_out, text_logits = lm_model.forward_text(input_)
        next_token = lm_model._sample_next_token(
            text_logits,
            self.use_sampling,
            self.temp,
            self.top_k,
            self.top_p,
        )
        assert next_token.shape[-1] == 1
        next_token = next_token[:, :, 0]  # shape is [B, K]
        this_gen_step = self.gen_sequence[:, :, self.offset + 1]

        if self.depformer_graph is None:
            self.depformer_graph = torch.cuda.CUDAGraph()
            self.depformer_in = next_token.clone()
            self.this_gen_step = this_gen_step.clone()
            self.transformer_out = transformer_out.clone()
            with torch.cuda.graph(self.depformer_graph):
                self.depformer_out = self.depformer_step(
                    self.depformer_in,
                    self.this_gen_step,
                    self.transformer_out,
                )
            # It is important to evaluate the graph here as otherwise self.depformer_out would
            # not contain the appropriate values.
            self.depformer_graph.replay()
        else:
            self.depformer_in.copy_(next_token)
            self.this_gen_step.copy_(this_gen_step)
            self.transformer_out.copy_(transformer_out)
            self.depformer_graph.replay()
        next_token = self.depformer_out
        print(self.offset, next_token)

        # ensure we don't overwrite prompt tokens, we only write over ungenerated tokens
        self.offset += 1
        self.gen_sequence[..., : lm_model.dep_q + 1, self.offset] = next_token

        out = []
        for k in range(1 + lm_model.dep_q):
            delay = lm_model.delays[k]
            if self.offset < delay:
                return None
            _out = self.gen_sequence[0, k, self.offset - delay].item()
            out.append(_out)

        return out

    def depformer_step(
        self,
        next_token: torch.Tensor,
        this_gen_step: torch.Tensor,
        transformer_out: torch.Tensor,
    ):
        lm_model = self.lm_model
        # Depformer gives us tokens one by one instead of K at once.
        assert next_token.shape[1] == 1, next_token.shape[1]
        next_token = next_token[:, 0]  # Now shape is B.
        depformer_tokens: tp.List[torch.Tensor] = []
        for cb_index in range(lm_model.dep_q + 1):
            if cb_index == 0:
                # No need to generate, `next_token` is actually the next text token.
                # We just need to only keep the new token if the value wasn't provided
                # in the prompt.
                next_token = torch.where(
                    this_gen_step[:, 0] == self.ungenerated,
                    next_token,
                    this_gen_step[:, 0],
                )
            else:
                input_ = next_token[:, None, None]
                if False:  # self.check:
                    # Check that we are not feeding in any value that is not generated yet.
                    assert not (input_ == self.ungenerated).any()
                    if cb_index == 1:
                        assert (input_ <= lm_model.text_card).all()
                    else:
                        assert (input_ <= lm_model.card).all()
                logits = lm_model.forward_depformer(
                    cb_index - 1, input_, transformer_out
                )
                next_token = lm_model._sample_next_token(
                    logits,
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                    self.top_p,
                )
                assert next_token.shape[-1] == 1
                next_token = next_token[:, 0, 0]  # shape is [B, K]
                next_token = torch.where(
                    this_gen_step[:, cb_index] == self.ungenerated,
                    next_token,
                    this_gen_step[:, cb_index],
                )

            # TODO(laurent): does the following really matter?
            # original_offset = self.offset - lm_model.delays[cb_index]
            # if original_offset < 0:
            #     # We are not currently generating this codebook, we replace with a special token.
            #     next_token[:] = self.initial[:, cb_index, 0]
            depformer_tokens.append(next_token)

        assert len(depformer_tokens) == lm_model.dep_q + 1, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        next_token = torch.stack(depformer_tokens, dim=1)
        assert next_token.shape == (
            self.num_samples,
            lm_model.dep_q + 1,
        ), next_token.shape
        return next_token
