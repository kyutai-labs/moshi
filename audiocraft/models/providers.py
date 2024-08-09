# Copyright (c) Kyutai
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import fasttext
import functools
import torch
import typing as tp
import weakref

from copy import deepcopy
from pyannote.audio import Model as PyannoteModel
from pyannote.audio import Inference as PyannoteInference
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

from ..environment import AudioCraftEnvironment


class ModelProvider:
    """A provider with weak references to a model.

    A provider is useful when a heavy model is used in many places and its resources should be shared,
    such as using the same pretrained model to compute several metrics.
    """

    def __init__(self, model_factory: tp.Callable):
        self._model_ref = None
        self.model_factory = model_factory
        self.model_factory_kwargs = None

    def get_model(self, **model_factory_kwargs):
        if self.model_factory_kwargs is None:
            self.model_factory_kwargs = model_factory_kwargs
        else:
            assert self.model_factory_kwargs == model_factory_kwargs, "model_factory_kwargs should be the same for all"
            " calls to ModelProvider.get_model."
        model = None
        if self._model_ref is not None:
            model = self._model_ref()
        if model is None:
            # The deepcopy avoids bugs with mutable arguments through partial to the model factory.
            model = deepcopy(self.model_factory)(**model_factory_kwargs)
            self._model_ref = weakref.ref(model)
        return model


ASRFeaturesProvider = ModelProvider(
    functools.partial(
        AutoProcessor.from_pretrained,
        pretrained_model_name_or_path="openai/whisper-large-v3",
        return_tensors="pt",
    )
)

ASRModelProvider = ModelProvider(
    functools.partial(
        AutoModelForSpeechSeq2Seq.from_pretrained,
        pretrained_model_name_or_path="openai/whisper-large-v3",
        low_cpu_mem_usage=False,
        use_safetensors=True,
    )
)

ASRPipelineProvider = ModelProvider(
    functools.partial(
        pipeline,
        task="automatic-speech-recognition",
        model="openai/whisper-large-v3",
        model_kwargs={"low_cpu_mem_usage": False, "use_safetensors": True, "attn_implementation": "flash_attention_2"},
        torch_dtype=torch.float16,
    )
)

LangIDProvider = ModelProvider(
    functools.partial(
        fasttext.load_model, path=str(AudioCraftEnvironment.resolve_reference_path("//reference/langid/lid.176.bin"))
    ),
)

SpeakerEmbProvider = ModelProvider(
    functools.partial(
        PyannoteInference,
        model=PyannoteModel.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM"))
)
