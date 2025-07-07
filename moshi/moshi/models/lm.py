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
from ..utils.quantize import replace_linear_with_qlinear
from ..modules.streaming import StreamingContainer, StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from .lm_utils import (_delay_sequence,
                       _undelay_sequence,
                       _init_layer,
                       ScaledEmbedding)


logger = logging.getLogger(__name__)


def scatter_with_mask_(tensor: torch.Tensor, dim: int,
                       index: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> None:
    """Scatter but skipping the updates that are masked."""
    old_value = tensor.gather(dim, index)
    value = torch.where(mask, value, old_value)
    tensor.scatter_(dim, index, value)


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]


class LMModel(StreamingContainer):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        text_card_out (int or None): Cardinality of output text, if different from the input.
        demux_second_text_stream: (bool): Whether two text streams are muxed together with a cartesian product.
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
        existing_text_padding_id (int): token to use for the padding.
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
        text_card_out: int | None = None,
        demux_second_text_stream: bool = False,
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
        existing_text_padding_id: int = 3,
        existing_text_end_padding_id: int = 0,
        extra_heads_num_heads: int = 0,
        extra_heads_dim: int = 6,
        context: tp.Optional[int] = None,
        causal: bool = True,
        condition_provider: tp.Optional[ConditionProvider] = None,
        fuser: tp.Optional[ConditionFuser] = None,
        quantize: bool = False,
        device=None,
        dtype=None,
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        text_card_out = text_card if text_card_out is None else text_card_out
        assert len(delays) == self.num_codebooks, f"expected {self.num_codebooks} delays, got {len(delays)}."
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.existing_text_end_padding_id = existing_text_end_padding_id
        self.context = context
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
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
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(text_card + 1, dim, demux_second_stream=demux_second_text_stream)

        self.text_linear = nn.Linear(dim, text_card_out, bias=bias_proj)
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
            context=context,
            causal=causal,
            checkpointing=gradient_checkpointing,
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
        kwargs_dep["cross_attention"] = False
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
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
        if dep_q > 0:
            # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
            self.depformer_emb = nn.ModuleList(
                [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
            )
            self.depformer_text_emb = EmbeddingFactory(
                text_card + 1,
                depformer_dim,
                demux_second_stream=demux_second_text_stream,
            )
            if depformer_dim_feedforward is None:
                depformer_dim_feedforward = int(hidden_scale * depformer_dim)
            self.depformer = StreamingTransformer(
                d_model=depformer_dim,
                dim_feedforward=depformer_dim_feedforward,
                norm=norm,
                weights_per_step_schedule=depformer_weights_per_step_schedule,
                causal=causal,
                quantize=quantize,
                checkpointing=gradient_checkpointing,
                device=device,
                dtype=dtype,
                **kwargs_dep,
            )
            # Depformer follow its own cycle of streaming entirely contained in one time step
            # and should not follow the streaming of the steps dimensions.
            self.depformer.set_streaming_detached(True)
        else:  # No-Depformer --- e.g., an ASR model
            self.depformer_emb = None
            self.depformer_text_emb = None
            self.depformer = None

        self.extra_heads = nn.ModuleList(
            [nn.Linear(dim, extra_heads_dim, bias=False) for _ in range(extra_heads_num_heads)]
        )

        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )
        self.to(device=device, dtype=dtype)
        # We always keep the condition provider as float32.
        self.condition_provider = condition_provider
        self.fuser = fuser
        if self.condition_provider is not None:
            self.condition_provider.to(device=device)
        if self.fuser is not None:
            self.fuser.to(device=device)
        self._init_weights()
        if quantize:
            replace_linear_with_qlinear(self)

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
        return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return self.existing_text_end_padding_id

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
    def device(self) -> torch.device:
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def dtype(self) -> torch.dtype:
        first_param = next(iter(self.text_emb.parameters()))
        return first_param.dtype

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

    def forward(
            self, codes: torch.Tensor,
            condition_tensors: tp.Optional[ConditionTensors] = None) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, returns the logits
        along with masks indicating the valid positions at which to compute the loss.
        The logits time steps are aligned with those in the input `code`.
        Should only be used for training, not inference (use `LMGen` for that).

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps. When text is supported,
                the first 'codebook' corresponds to the text, and the remaining codebooks are for the  audio.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning tensors.
        Returns:
            LMOutput: Language model outputs, containing either text or audio logits, or both.
                logits (torch.Tensor, or None) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor, or None) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
                text_logits (torch.Tensor, or None) of shape [B, 1, T, text_card].
                text_mask (torch.Tensor, or None) of shape [B, 1, T], mask over the valid positions for the text.
        """
        B, K, T = codes.shape
        assert K == self.num_codebooks, (K, self.num_codebooks)
        # Delaying codes and removing the last time step that will never be an input.
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        # Inserting the empty tokens for the first time step.
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        sum_condition: torch.Tensor | None = None
        cross_attention_src: torch.Tensor | None = None
        if condition_tensors is None:
            assert self.fuser is None
        else:
            assert self.fuser is not None
            sum_condition = self.fuser.get_sum(condition_tensors)
            cross_attention_src = self.fuser.get_cross(condition_tensors)

        transformer_out, text_logits = self.forward_text(delayed_codes[:, :, :-1], sum_condition, cross_attention_src)
        assert transformer_out.shape[0] == delayed_codes.shape[0]
        assert transformer_out.shape[1] == delayed_codes.shape[2] - 1
        logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens. We will with NaN values invalid positions
        # to ensure they properly handled.
        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset:self.audio_offset + self.dep_q],
            logits, fill_value=float('NaN'))
        logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
        text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
        text_logits_mask &= (codes[:, :1] != self.zero_token_id)
        return LMOutput(logits, logits_mask, text_logits, text_logits_mask)

    def forward_text(
        self,
        sequence: torch.Tensor, sum_condition: torch.Tensor | None = None,
        cross_attention_src: torch.Tensor | None = None,
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
        if cross_attention_src is not None:
            cross_attention_src = cross_attention_src.to(input_)
        transformer_out = self.transformer(input_, cross_attention_src=cross_attention_src)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_depformer_training(
        self,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        assert self.depformer_text_emb
        assert self.depformer_emb
        assert self.depformer

        B, K, T = sequence.shape
        Ka = self.dep_q
        assert (
            K == self.num_codebooks
        ), f"Codebooks for Depformer training should be passed all at once, got {K}."
        depformer_inputs = []
        for cb_index in range(Ka):
            if self.depformer_multi_linear:
                linear_index = cb_index
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[cb_index]
                transformer_in = self.depformer_in[linear_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)
            if cb_index == 0:
                token_in = self.depformer_text_emb(sequence[:, 0])
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2)
        # depformer_input is [B, T, K, depformer_dim], reshaping to [B * T, K, D]
        depformer_input = depformer_input.view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = []
        for cb_index in range(Ka):
            logits = self.linears[cb_index](depformer_output[:, cb_index])
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack(all_logits, 1)
        assert logits.dim() == 4, logits.shape  # [B, Ka, T, card]
        return logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        assert self.depformer_text_emb is not None
        assert self.depformer_emb is not None
        assert self.depformer is not None
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
            depformer_input = self.depformer_in[in_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
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
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    def _init_weights(self):
        """Initialization of the transformer module weights.
        Mostly truncated gaussian, with `std = 1 / sqrt(dim_in)`.
        Embeddings are also initialized with `1 / sqrt(dim)` rather than `1`.
        Some layers are not going to be properly initialized:
            - in_proj in MHA.
            - depth transformer layers.
        This is to match how our models were trained so far.
        """

        for emb_layer in self.emb:
            _init_layer(emb_layer)
        if self.depformer_emb is not None:
            for emb_layer in self.depformer_emb:
                _init_layer(emb_layer)
        _init_layer(self.text_emb)
        if self.depformer_text_emb is not None:
            _init_layer(self.depformer_text_emb)
        _init_layer(self.text_linear)

        for tr_layer in self.transformer.layers:
            tr_layer.apply(_init_layer)

        for linear in self.linears:
            _init_layer(linear)


@dataclass
class _LMGenState(State):
    cache: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_depth: CUDAGraphed | None
    offsets: torch.Tensor
    offset_cpu: int = 0
    condition_sum: torch.Tensor | None = None
    condition_cross: torch.Tensor | None = None
    cfg_is_masked_until: torch.Tensor | None = None
    exit_stack: ExitStack = field(default_factory=ExitStack)
    reset_callback: tp.Callable[[torch.Tensor], None] | None = None
    set_exec_mask_callback: tp.Callable[[torch.Tensor], None] | None = None

    def reset(self, reset_mask: torch.Tensor) -> None:
        super().reset(reset_mask)
        self.offsets[:] = torch.where(reset_mask, torch.zeros_like(self.offsets), self.offsets)
        self.offset_cpu = 0
        if self.reset_callback is not None:
            self.reset_callback(reset_mask)

    def set_exec_mask(self, exec_mask: torch.Tensor):
        super().set_exec_mask(exec_mask)
        if self.set_exec_mask_callback is not None:
            self.set_exec_mask_callback(exec_mask)

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
        on_text_hook: tp.Optional[tp.Callable[[torch.Tensor], None]] = None,
        on_text_logits_hook: tp.Optional[tp.Callable[[torch.Tensor], None]] = None,
        on_audio_hook: tp.Optional[tp.Callable[[torch.Tensor], None]] = None,
        support_out_of_sync: bool = False,
        cfg_is_masked_until: list[int] | None = None,
        cfg_is_no_text: bool = False,
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
        self.on_text_hook = on_text_hook
        self.on_text_logits_hook = on_text_logits_hook
        self.on_audio_hook = on_audio_hook
        self.support_out_of_sync = support_out_of_sync
        self.cfg_is_masked_until = cfg_is_masked_until
        self.cfg_is_no_text = cfg_is_no_text
        if self.cfg_coef != 1.:
            if not self.cfg_is_no_text and not self.cfg_is_masked_until:
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
        offsets = torch.zeros(batch_size, device=lm_model.device, dtype=torch.long)

        if self.lm_model.fuser is None:
            assert not self.condition_tensors
            condition_sum = None
            condition_cross = None
        else:
            assert self.condition_tensors is not None
            condition_sum = self.lm_model.fuser.get_sum(self.condition_tensors)
            condition_cross = self.lm_model.fuser.get_cross(self.condition_tensors)
            if condition_sum is not None:
                condition_sum = condition_sum.to(self.lm_model.dtype)
            if condition_cross is not None:
                condition_cross = condition_cross.to(self.lm_model.dtype)

        disable = lm_model.device.type != 'cuda'
        graphed_main = CUDAGraphed(lm_model.forward_text, disable=disable)
        if lm_model.depformer is not None:
            graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)
        else:
            graphed_depth = None

        if self.cfg_is_masked_until is None:
            cfg_is_masked_until = None
        else:
            cfg_is_masked_until = torch.tensor(self.cfg_is_masked_until, dtype=torch.long, device=lm_model.device)

        state = _LMGenState(
            batch_size, lm_model.device, cache, initial, graphed_main, graphed_depth,
            offsets, condition_sum=condition_sum, condition_cross=condition_cross,
            cfg_is_masked_until=cfg_is_masked_until)

        if self.cfg_coef != 1.:
            batch_size *= 2
            if state.condition_sum is not None:
                assert state.condition_sum.shape[0] == batch_size, "cfg requires 2x more conditions."
            if state.condition_cross is not None:
                assert state.condition_cross.shape[0] == batch_size, "cfg requires 2x more conditions."
        state.exit_stack.enter_context(self.lm_model.streaming(batch_size))

        def _reset_callback(reset_mask: torch.Tensor) -> None:
            if self.cfg_coef != 1.:
                reset_mask = reset_mask.repeat(2)
            self.lm_model.reset_streaming(reset_mask)

        def _set_exec_mask_callback(exec_mask: torch.Tensor) -> None:
            if self.cfg_coef != 1.:
                exec_mask = exec_mask.repeat(2)
            self.lm_model.set_exec_mask(exec_mask)

        state.reset_callback = _reset_callback
        state.set_exec_mask_callback = _set_exec_mask_callback
        return state

    @torch.no_grad()
    def _step(self, input_tokens: torch.Tensor,
              depformer_replace_tokens: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, torch.Tensor] | None:
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
            Ki >= needed_tokens
        ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

        if Ki > needed_tokens:
            input_tokens = input_tokens[:, :needed_tokens, :]

        CT = state.cache.shape[2]

        delays = self.delays_cuda[lm_model.dep_q + 1:]
        write_positions = (state.offsets[:, None, None] + delays[:, None]) % CT
        scatter_with_mask_(state.cache[:, lm_model.dep_q + 1:], -1, write_positions, input_tokens,
                           state.exec_mask[:, None, None])

        is_init = state.offsets[:, None, None] <= self.delays_cuda[:, None]
        is_init |= ~state.exec_mask[:, None, None]  # we also give init tokens if not executing to avoid crashing.
        positions = (state.offsets % CT)[:, None, None].expand_as(is_init)
        input_ = state.cache.gather(dim=2, index=positions)
        input_ = torch.where(is_init, state.initial, input_)

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offsets,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()

        zero = torch.full((1,), self.lm_model.zero_token_id, dtype=torch.long, device=input_.device)
        if self.cfg_coef != 1.:
            if state.cfg_is_masked_until is not None:
                limit = self.delays_cuda[:, None] + state.cfg_is_masked_until.view(-1, 1, 1)
                is_zeroed = state.offsets[:, None, None] <= limit

                masked = torch.where(is_zeroed & ~is_init, zero, input_)
                input_ = torch.cat([input_, masked], dim=0)
            else:
                input_ = input_.repeat(2, 1, 1)
            if self.cfg_is_no_text:
                input_[B:, :1] = torch.where(~is_init[:, :1], zero, input_[B:, :1])

        transformer_out, text_logits = state.graphed_main(input_, state.condition_sum, state.condition_cross)
        if self.cfg_coef != 1.:
            logits, logits_null = text_logits.chunk(2)
            if self.cfg_is_no_text:
                text_logits = logits
            else:
                text_logits = logits_null + (logits - logits_null) * self.cfg_coef
        # Shape of text_logits should be [B, K_text=1, T=1, Card_text]
        if self.on_text_logits_hook:
            self.on_text_logits_hook(text_logits)
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
        if self.on_text_hook is not None:
            self.on_text_hook(text_token)
        if state.graphed_depth is None:
            audio_tokens = None
        elif depformer_replace_tokens is None:
            audio_tokens = state.graphed_depth(text_token, transformer_out)
            if self.on_audio_hook is not None:
                self.on_audio_hook(audio_tokens)
        else:
            assert depformer_replace_tokens.dim() == 3
            audio_tokens = depformer_replace_tokens.squeeze(-1)

        state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)
        state.offset_cpu += 1
        positions = (state.offsets % CT)[:, None, None]
        scatter_with_mask_(state.cache[:, :1], -1, positions,
                           text_token[:, None, None], state.exec_mask[:, None, None])
        if audio_tokens is not None:
            audio_tokens = audio_tokens[:, :, None]
            scatter_with_mask_(
                state.cache[:, 1 : lm_model.dep_q + 1, :],
                -1,
                positions.expand_as(audio_tokens),
                audio_tokens,
                state.exec_mask[:, None, None],
            )

        if not self.support_out_of_sync and state.offset_cpu <= self.max_delay:
            # When using out of sync exec, should not rely on this being None.
            return None
        B = state.cache.shape[0]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (state.offsets[:, None, None] - self.max_delay + gen_delays_cuda[:, None]) % CT
        out = state.cache.gather(dim=2, index=index)
        mask = (state.offsets <= self.max_delay) | ~state.exec_mask
        out[mask, :, :] = lm_model.ungenerated_token_id
        return out, transformer_out

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor,
             depformer_replace_tokens: torch.Tensor | None = None) -> torch.Tensor | None:
        out = self._step(input_tokens, depformer_replace_tokens)
        if out is None:
            return None
        return out[0]

    @torch.no_grad()
    def step_with_extra_heads(
        self,
        input_tokens: torch.Tensor,
        depformer_replace_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        out = self._step(input_tokens, depformer_replace_tokens)
        if out is None:
            return None
        out, transformer_out = out
        extra_heads = [extra_head(transformer_out) for extra_head in self.lm_model.extra_heads]
        return out, extra_heads

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
        assert lm_model.depformer
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
