import logging
import random
import typing as tp
import warnings

import spacy
import torch
from torch import nn
from torch.nn .utils.rnn import pad_sequence
from transformers import MT5EncoderModel, MT5Tokenizer, T5EncoderModel, T5Tokenizer
import sentencepiece


from .base import _BaseTextConditioner, ConditionType
from ..environment import AudioCraftEnvironment
from ..utils.autocast import TorchAutocast
from ..utils.utils import length_to_mask, hash_trick


logger = logging.getLogger(__name__)


class TokenizedText(tp.NamedTuple):
    tokens: torch.Tensor   # should be long tensor.
    mask: torch.Tensor     # should be bool tensor.


class TextConditioner(_BaseTextConditioner[TokenizedText]):
    ...


class Tokenizer:
    """Base tokenizer implementation
    """
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]

    Args:
        n_bins (int): we use a hash modulo the number of bins of the tokens. Should set
            the number of bins to use.
        language (str or None): if provided, name of a spacy model for doing text normalization.
            Must be provided if `lemma` or `stopwords` is True.
        lemma (bool): whether to lemmatize the text.
        stopwords (bool): whether to remove stopwords.

    """
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, language: tp.Optional[str] = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.lemma = lemma
        self.stopwords = stopwords
        self.pad_idx = n_bins
        if language is None:
            assert not lemma and not stopwords
            self.language_model = None
        else:
            try:
                self.language_model = spacy.load(language)
            except IOError:
                spacy.cli.download(language)  # type: ignore
                self.language_model = spacy.load(language)

    def process_text(self, text: str) -> tp.List[str]:
        if self.language_model is None:
            return text.split()
        else:
            # normalize text
            words = self.language_model(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                words = [word for word in words if not word.is_stop]  # type: ignore
            # remove punctuation
            words = [word for word in words if word.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            if self.lemma:
                out_text = [word.lemma_ for word in words]
            else:
                out_text = [word.text for word in words]

            return out_text

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are.
        """
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            words = self.process_text(text)
            lengths.append(len(words))
            # convert to tensor
            tokens = torch.tensor([hash_trick(word, self.n_bins) for word in words])
            output.append(tokens)

        mask = length_to_mask(torch.tensor(lengths))
        padded_output = pad_sequence(output, padding_value=self.pad_idx, batch_first=True).int()
        return TokenizedText(padded_output, mask)


class NoopTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]
    """
    def __init__(self, n_bins: int):
        self.n_bins = n_bins
        self.pad_idx = n_bins + 1

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)

        tokens = torch.tensor(output).int()
        mask = length_to_mask(torch.tensor(lengths))
        return TokenizedText(tokens, mask)


class SentencePieceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77, PAD, PAD, PAD, PAD, PAD, PAD]]

    Args:
        n_bins (int): should be equal to the number of elements in the sentencepiece tokenizer.
        tokenizer_path (str): path to the sentencepiece tokenizer model.

    """

    def __init__(self, nbins: int, tokenizer_path: str) -> None:
        self.sp = sentencepiece.SentencePieceProcessor(tokenizer_path)
        self.pad_idx = self.sp.pad_id()
        assert nbins == self.sp.vocab_size(), \
            f"sentencepiece tokenizer has vocab size={self.sp.vocab_size()} but nbins={nbins} was specified"

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> TokenizedText:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in SentencePiece.
                - And a mask indicating where the padding tokens are.
        """
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # tokenize and convert to tensor
            tokens = torch.tensor(self.sp.encode(text, out_type=int))
            lengths.append(len(tokens))
            output.append(tokens)

        mask = length_to_mask(torch.tensor(lengths))
        padded_output = pad_sequence(output, padding_value=self.pad_idx, batch_first=True).int()
        return TokenizedText(padded_output, mask)


class LUTConditioner(TextConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """
    def __init__(self, n_bins: int, tokenizer: str, **kwargs):
        if "tokenizer_path" in kwargs:
            tokenizer_path = kwargs.pop("tokenizer_path")
            assert isinstance(tokenizer_path, str)
            if tokenizer_path.startswith("//reference"):
                tokenizer_path = str(AudioCraftEnvironment.resolve_reference_path(tokenizer_path))
        else:
            tokenizer_path = None
        super().__init__(**kwargs)
        self.embed = nn.Embedding(n_bins + 1, self.dim)  # n_bins + 1 for padding.
        self.tokenizer: Tokenizer
        if tokenizer == 'whitespace':
            self.tokenizer = WhiteSpaceTokenizer(n_bins)
        elif tokenizer == 'noop':
            self.tokenizer = NoopTokenizer(n_bins)
        elif tokenizer == "sentencepiece":
            assert tokenizer_path is not None, "tokenizer_path must be specified when tokenizer=sentencepiece"
            self.tokenizer = SentencePieceTokenizer(n_bins, tokenizer_path)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return TokenizedText(tokens.to(device), mask.to(device))

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        return ConditionType(embeds, mask)


class T5Conditioner(TextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }
    ENCODER_CLS = T5EncoderModel
    TOKENIZER_CLS = T5Tokenizer

    def __init__(self, name: str, finetune: bool = False,
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False, **kwargs):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(dim=self.MODELS_DIMS[name], **kwargs)
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        if autocast_dtype is None or self.device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if self.device != 'cpu':
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = self.TOKENIZER_CLS.from_pretrained(name)
                t5 = self.ENCODER_CLS.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)
        if finetune:
            self.t5 = t5
        else:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            self.__dict__['t5'] = t5.to(self.device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            entries = [" ".join(self.text_normalizer.process_text(entry)) for entry in entries]
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                if self.word_dropout:
                    words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                    entry = " ".join(words)
                new_entries.append(entry)
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors='pt', padding=True).to(self.device)
        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
        return TokenizedText(inputs['input_ids'], mask)

    def _get_condition(self, tokenized: TokenizedText) -> ConditionType:
        tokens, mask = tokenized
        inputs = {
            'attention_mask': mask,
            'input_ids': tokens,
        }
        mask = inputs['attention_mask']
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        return ConditionType(embeds, mask)


class MultilingualT5Conditioner(T5Conditioner):
    MODELS = ["google/mt5-small", "google/mt5-base", "google/mt5-large", "google/mt5-xl", "google/mt5-xxl"]
    MODELS_DIMS = {
        "google/mt5-small": 512,
        "google/mt5-base": 768,
        "google/mt5-large": 1024,
        "google/mt5-xl": 1280,
        "google/mt5-xxl": 1280,
    }
    ENCODER_CLS = MT5EncoderModel
    TOKENIZER_CLS = MT5Tokenizer
