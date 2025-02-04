# Generated content DO NOT EDIT
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence
from os import PathLike

@staticmethod
def write_wav(filename, data, sample_rate):
    """
    Writes an audio file using the wav format based on pcm data from a numpy array.

    This only supports a single channel at the moment so the input array data is expected to have a
    single dimension.
    """
    pass

class StreamTokenizer:
    def __init__(path, *, dtype="f32", max_seq_len=None):
        pass

    def decode(self, codes):
        """ """
        pass

    def encode(self, pcm_data):
        """ """
        pass

    def get_decoded(self):
        """ """
        pass

    def get_encoded(self):
        """ """
        pass

class Tokenizer:
    def __init__(path, *, dtype="f32", max_seq_len=None):
        pass

    def decode(self, codes):
        """ """
        pass

    def decode_step(self, codes):
        """ """
        pass

    def encode(self, pcm_data):
        """ """
        pass

    def encode_step(self, pcm_data):
        """ """
        pass

    def reset(self):
        """ """
        pass
