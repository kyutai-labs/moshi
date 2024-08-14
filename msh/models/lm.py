# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import typing as tp

from einops import rearrange
import torch
from torch import nn

from ..utils import utils
from ..utils.autocast import TorchAutocast
from ..modules.streaming import StreamingModule, State
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
        n_q (int): Number of parallel streams to model.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary. Activates text support
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer (bool): whether to use a smaller Transformer along the codebooks for predicting them.
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
        card: int = 1024,
        text_card: tp.Optional[int] = None,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer: bool = False,
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
        self.card = card
        self.text_card = text_card
        assert len(delays) > 0, "Delays must be non empty"
        assert len(delays) <= self.num_codebooks, "Too many delays"
        if len(delays) < self.num_codebooks:
            delays = delays + [delays[-1]] * (self.num_codebooks - len(delays))
            logger.info("Extended delay to %r", delays)
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
        if text_card:
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
        self.depformer: tp.Optional[nn.Module] = None
        self.depformer_multi_linear = depformer_multi_linear
        if depformer:
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
                kwargs_dep["weights_per_step"] = n_q
            if depformer_multi_linear:
                # One linear layer per codebook to project different informations from the main model.
                self.depformer_in = nn.ModuleList(
                    [nn.Linear(dim, depformer_dim, bias=False) for _ in range(n_q)]
                )
            else:
                self.depformer_in = nn.ModuleList(
                    [nn.Linear(dim, depformer_dim, bias=False)]
                )
            # Only using up to n_q - 1 because the last codebook is never an input to Depformer.
            self.depformer_emb = nn.ModuleList(
                [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(n_q - 1)]
            )
            if text_card is not None:
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
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)]
        )

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        assert self.text_card is not None
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        assert self.text_card is not None
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
        return self.n_q + int(self.has_text)

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return int(self.has_text)

    @property
    def has_text(self) -> bool:
        return self.text_card is not None

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

    @torch.no_grad()
    def generate(
        self,
        prompt: tp.Optional[torch.Tensor] = None,
        num_samples: tp.Optional[int] = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        check: bool = False,
        callback: tp.Optional[tp.Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (torch.Tensor, optional): Prompt tokens of shape [B, Kt + Ka, T]. When the model supports text,
                `Kt` is 1. The text token is at index 0. `Ka` is the number of audio codebooks.
            num_samples (int, optional): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            check (bool): Whether to apply further checks on generated sequence.
            callback (Callback, optional): Callback function to report generation progress.
         Returns:
            torch.Tensor: Generated tokens, with shape `[B, Kt + Ka, T]`. Note that even if only one modality
                is generated, the output always contains `Kt + Ka` tokens.

        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        else:
            possible_num_samples.append(1)
        assert [
            x == possible_num_samples[0] for x in possible_num_samples
        ], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]
        assert isinstance(num_samples, int)

        initial = self._get_initial_token().expand(num_samples, -1, -1)
        max_delay = max(
            self.delays
        )  # with delays, we need to generate a few more time steps.
        ungenerated = (
            self.ungenerated_token_id
        )  # special value to indicate tokens to generate
        gen_sequence = torch.full(
            (num_samples, self.num_codebooks, max_gen_len + max_delay + 1),
            ungenerated,
            device=device,
            dtype=torch.long,
        )
        # special token for the beginning of the sequence.
        gen_sequence[:, :, :1] = initial
        start_offset = 0

        if prompt is not None:
            assert start_offset < max_gen_len
            PT = prompt.shape[-1]
            for cb in range(self.num_codebooks):
                delay = self.delays[cb]
                gen_sequence[:, cb, : delay + 1] = initial[:, cb]
                gen_sequence[:, cb, delay + 1 : delay + 1 + PT] = prompt[:, cb, :]
            # We look for the first time step that is ungenerated, as we allow for partial teacher
            # forcing, for instance by providing the text and generating the audio or the opposite.
            ungenerated_steps = (gen_sequence == ungenerated).nonzero()[:, 2]
            if not ungenerated_steps.numel():
                raise RuntimeError("Nothing to generate.")
            # start offset will be one step before the first value to generate.
            # The `-1` offset is because time step T is generated as the output of
            # timestep T - 1.
            start_offset = int(ungenerated_steps.amin()) - 1
            assert start_offset >= 0
            logger.debug("Start offset is %d", start_offset)

        set_attention_context(self.transformer, self.context)
        with self.streaming(), self.autocast:
            for offset in range(start_offset, max_gen_len + max_delay):
                # `offset` measures position in the output tensor with no delays.
                # In particular, there is a shift of 1 with the `gen_sequence` that includes
                # the initial empty token.
                logger.debug("Offset %d / %d", offset, max_gen_len + max_delay)
                # get current sequence (note that the streaming API is providing the caching over previous offsets)

                if offset == start_offset:
                    input_ = gen_sequence[:, :, : offset + 1]
                else:
                    input_ = gen_sequence[:, :, offset : offset + 1]

                if check:
                    # Check that we are not feeding in any value that is not generated yet.
                    assert not (input_ == ungenerated).any(), (offset, input_)
                    assert (input_[:, self.audio_offset :] <= self.card).all(), input_
                    if self.has_text:
                        assert (input_[:, :1] <= self.text_card).all()

                transformer_out, text_logits = self.forward_text(input_)
                next_token = self._sample_next_token(
                    text_logits,
                    use_sampling,
                    temp,
                    top_k,
                    top_p,
                )
                assert next_token.shape[-1] == 1
                next_token = next_token[:, :, 0]  # shape is [B, K]
                this_gen_step = gen_sequence[:, :, offset + 1]
                assert self.depformer is not None

                # Depformer gives us tokens one by one instead of K at once.
                assert next_token.shape[1] == 1, next_token.shape[1]
                next_token = next_token[:, 0]  # Now shape is B.
                depformer_tokens: tp.List[torch.Tensor] = []
                for cb_index in range(self.num_codebooks):
                    if cb_index == 0:
                        # No need to generate, `next_token` is actually the next text token.
                        # We just need to only keep the new token if the value wasn't provided
                        # in the prompt.
                        next_token = torch.where(
                            this_gen_step[:, 0] == ungenerated,
                            next_token,
                            this_gen_step[:, 0],
                        )
                    else:
                        input_ = next_token[:, None, None]
                        if check:
                            # Check that we are not feeding in any value that is not generated yet.
                            assert not (input_ == ungenerated).any()
                            if self.has_text and cb_index == 1:
                                assert (input_ <= self.text_card).all()
                            else:
                                assert (input_ <= self.card).all()
                        logits = self.forward_depformer(
                            cb_index - 1, input_, transformer_out
                        )
                        next_token = self._sample_next_token(
                            logits,
                            use_sampling,
                            temp,
                            top_k,
                            top_p,
                        )
                        assert next_token.shape[-1] == 1
                        next_token = next_token[:, 0, 0]  # shape is [B, K]
                        next_token = torch.where(
                            this_gen_step[:, cb_index] == ungenerated,
                            next_token,
                            this_gen_step[:, cb_index],
                        )

                    original_offset = offset - self.delays[cb_index]
                    if original_offset < 0:
                        # We are not currently generating this codebook, we replace with a special token.
                        next_token[:] = initial[:, cb_index, 0]
                    depformer_tokens.append(next_token)

                assert len(depformer_tokens) == self.num_codebooks, (
                    len(depformer_tokens),
                    self.num_codebooks,
                )
                next_token = torch.stack(depformer_tokens, dim=1)
                assert next_token.shape == (
                    num_samples,
                    self.num_codebooks,
                ), next_token.shape

                # ensure we don't overwrite prompt tokens, we only write over ungenerated tokens
                gen_sequence[..., offset + 1] = next_token
                if callback is not None:
                    callback(
                        1 + offset - start_offset,
                        max_gen_len + max_delay - start_offset,
                    )

        output, mask = _undelay_sequence(
            self.delays, gen_sequence[:, :, 1:], fill_value=ungenerated
        )
        assert mask[:, :, :max_gen_len].all()
        output = output[:, :, :max_gen_len]
        tgt_shape = (num_samples, self.num_codebooks, max_gen_len)
        assert output.shape == tgt_shape, (output.shape, tgt_shape)
        # ensure sequence has been entirely filled
        assert not (output == ungenerated).any()
        return output


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
        self.gen_sequence[:, :, :1] = self.initial
        self.zero = torch.full(
            [1], lm_model.zero_token_id, device=device, dtype=torch.long
        )
        self.audio_offset = lm_model.audio_offset
        set_attention_context(lm_model.transformer, lm_model.context)
        self.offset = 0

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

        for k, delay in enumerate(lm_model.delays):
            if self.offset < delay or input_tokens[k] == self.ungenerated:
                continue
            if self.gen_sequence[:, k, self.offset - delay] == self.ungenerated:
                self.gen_sequence[:, k, self.offset - delay] = input_tokens[k]

        input_ = self.gen_sequence[:, :, self.offset : self.offset + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == self.ungenerated).any(), (self.offset, input_)
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            if lm_model.has_text:
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

        assert lm_model.depformer is not None
        # Depformer gives us tokens one by one instead of K at once.
        assert next_token.shape[1] == 1, next_token.shape[1]
        next_token = next_token[:, 0]  # Now shape is B.
        depformer_tokens: tp.List[torch.Tensor] = []
        for cb_index in range(lm_model.num_codebooks):
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
                if self.check:
                    # Check that we are not feeding in any value that is not generated yet.
                    assert not (input_ == self.ungenerated).any()
                    if lm_model.has_text and cb_index == 1:
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

            original_offset = self.offset - lm_model.delays[cb_index]
            if original_offset < 0:
                # We are not currently generating this codebook, we replace with a special token.
                next_token[:] = self.initial[:, cb_index, 0]
            depformer_tokens.append(next_token)

        assert len(depformer_tokens) == lm_model.num_codebooks, (
            len(depformer_tokens),
            lm_model.num_codebooks,
        )
        next_token = torch.stack(depformer_tokens, dim=1)
        assert next_token.shape == (
            self.num_samples,
            lm_model.num_codebooks,
        ), next_token.shape

        # ensure we don't overwrite prompt tokens, we only write over ungenerated tokens
        # TODO(laurent): find out how this is supposed to work.
        self.offset += 1
        self.gen_sequence[..., self.offset] = next_token

        out = []
        for k, delay in enumerate(lm_model.delays):
            if self.offset < delay:
                return None
            _out = self.gen_sequence[0, k, self.offset - delay].item()
            out.append(_out)

        return out
