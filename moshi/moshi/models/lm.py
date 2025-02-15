# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
import logging
import typing as tp

import torch
from torch import nn

from ..conditioners import ConditionProvider, ConditionFuser, ConditionTensors
from ..utils.sampling import sample_token
from ..utils.compile import CUDAGraphed
from ..utils import quantize
from ..modules.streaming import StreamingContainer, StreamingModule, State
from ..modules.transformer import (
    StreamingTransformer,
    quantize_transformer,
    create_norm_fn,
)


logger = logging.getLogger(__name__)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
        low_rank (int | None): if provided, uses low rank embedding with a linear layer to reach
            the desired dimension. Quite efficient for reducing the number of weights for very large vocabs.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 *args, norm: bool = False, zero_idx: int = -1,
                 low_rank: int | None = None, **kwargs):
        super().__init__(num_embeddings, low_rank or embedding_dim, *args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx
        self.low_rank = None
        if low_rank is not None:
            self.low_rank = nn.Linear(low_rank, embedding_dim, bias=False)

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        if self.low_rank is not None:
            y = quantize.linear(self.low_rank, y)
        return y


class LMModel(StreamingContainer):
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
        depformer_weights_per_step_schedule (list[int] | None): mapping `CODEBOOK_INDEX -> WEIGHT_INDEX`, allowing
        depformer_low_rank_embeddings (int | None): if provided, uses low rank embeddings, with a linear
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
        depformer_weights_per_step_schedule: list[int] | None = None,
        depformer_low_rank_embeddings: int | None = None,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: tp.Optional[int] = None,
        context: tp.Optional[int] = None,
        condition_provider: tp.Optional[ConditionProvider] = None,
        fuser: tp.Optional[ConditionFuser] = None,
        quantize: bool = False,
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
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
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
            quantize=quantize,
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
            # One linear layer per codebook to project different information from the main model.
            num_in = dep_q
            if depformer_weights_per_step_schedule:
                num_in = max(depformer_weights_per_step_schedule) + 1
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False) for _ in range(num_in)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(dim, depformer_dim, bias=False)]
            )
        EmbeddingFactory = partial(EmbeddingFactory, low_rank=depformer_low_rank_embeddings)
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
            weights_per_step_schedule=depformer_weights_per_step_schedule,
            quantize=quantize,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        # Depformer follow its own cycle of streaming entirely contained in one time step
        # and should not follow the streaming of the steps dimensions.
        self.depformer.set_streaming_detached(True)
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.to(device=device, dtype=dtype)
        if quantize:
            quantize_transformer(self)

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
        sequence: torch.Tensor, sum_condition: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if sum_condition is not None:
            input_ = input_ + sum_condition.to(input_)
        transformer_out = self.transformer(input_)

        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = quantize.linear(self.text_linear, transformer_out)
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
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: tp.Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            in_index = depformer_cb_index
            if self.depformer_weights_per_step_schedule is not None:
                in_index = self.depformer_weights_per_step_schedule[in_index]
            depformer_input = quantize.linear(self.depformer_in[in_index], depformer_input)
        else:
            depformer_input = quantize.linear(self.depformer_in[0], depformer_input)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        assert last_token_input is not None
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(depformer_input)
        logits = quantize.linear(self.linears[depformer_cb_index], dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits


@dataclass
class _LMGenState(State):
    batch_size: int
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed
    condition_sum: torch.Tensor | None = None
    offset: int = 0
    exit_stack: ExitStack = field(default_factory=ExitStack)
    reset_callback: tp.Callable[[], None] | None = None

    def reset(self):
        self.offset = 0
        if self.reset_callback is not None:
            self.reset_callback()

    def __enter__(self):
        self.exit_stack.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_stack.__exit__(exc_type, exc_value, traceback)


class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        cfg_coef: float = 1.,
        check: bool = False,
        condition_tensors: ConditionTensors | None = None,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.lm_model.set_streaming_detached(True)
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.cfg_coef = cfg_coef
        self.check = check
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )
        self.condition_tensors = condition_tensors
        if self.cfg_coef != 1.:
            assert self.lm_model.fuser is not None, "Model has no fuser, cannot do CFG."
            assert self.condition_tensors, "Missing condition tensors for CFG."

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )

        if self.lm_model.fuser is None:
            assert not self.condition_tensors
            condition_sum = None
        else:
            assert self.condition_tensors is not None
            condition_sum = self.lm_model.fuser.get_sum(self.condition_tensors)

        disable = lm_model.device.type != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        state = _LMGenState(
            batch_size, cache, initial, graphed_main, graphed_depth,
            condition_sum=condition_sum)

        if self.cfg_coef != 1.:
            batch_size *= 2
            if state.condition_sum is not None:
                assert state.condition_sum.shape[0] == batch_size, "CFG requires 2x more conditions."
        state.exit_stack.enter_context(self.lm_model.streaming(batch_size))
        state.reset_callback = self.lm_model.reset_streaming
        return state

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
        B, Ki, S = input_tokens.shape
        assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
        assert S == 1, "Only support being given steps one by one."
        needed_tokens = lm_model.num_codebooks - lm_model.dep_q - 1
        assert (
            Ki == needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        CT = state.cache.shape[2]

        for q_other in range(input_tokens.shape[1]):
            k = lm_model.dep_q + 1 + q_other
            delay = lm_model.delays[k]
            write_position = (state.offset + delay) % CT
            state.cache[:, k, write_position : write_position + 1] = input_tokens[
                :, q_other
            ]

        position = state.offset % CT
        for k, delay in enumerate(lm_model.delays):
            # Only for the very beginning, we extend the initial token for the acoustic
            # token that are delayed, and thus have no good value to take.
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]
        input_ = state.cache[:, :, position : position + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        if self.cfg_coef != 1.:
            input_ = input_.repeat(2, 1, 1)
        transformer_out, text_logits = state.graphed_main(input_, state.condition_sum)
        if self.cfg_coef != 1.:
            logits, logits_null = text_logits.chunk(2)
            text_logits = logits_null + (logits - logits_null) * self.cfg_coef
        # Shape of text_logits should be [B, K_text=1, T=1, Card_text]
        text_token = sample_token(
            text_logits.float(),
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert text_token.dim() == 3, text_token.shape
        assert text_token.shape[2] == 1
        assert text_token.shape[1] == 1, "Only one text stream supported."
        text_token = text_token[:, 0, 0]  # shape is [B]
        audio_tokens = state.graphed_depth(text_token, transformer_out)

        # ensure we don't overwrite prompt tokens, we only write over ungenerated tokens
        state.offset += 1
        position = state.offset % CT
        state.cache[:, 0, position] = text_token
        state.cache[:, 1 : lm_model.dep_q + 1, position] = audio_tokens

        if state.offset <= self.max_delay:
            return None
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)
        return out

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, = text_token.shape
        B_cfg = B
        if self.cfg_coef != 1.:
            B_cfg = 2 * B
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B_cfg):
            assert lm_model.depformer.is_streaming
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                if self.cfg_coef != 1.:
                    input_ = input_.repeat(2, 1, 1)
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                if self.cfg_coef != 1.:
                    logits, logits_null = logits.chunk(2)
                    logits = logits_null + (logits - logits_null) * self.cfg_coef
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                depformer_tokens.append(next_token)
                prev_token = next_token

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, lm_model.dep_q), out.shape
        return out
