import torchaudio
import torch

import os
from typing import Any, Union
from lhotse.utils import Seconds
from pathlib import Path
from moshi.moshi.models import loaders
import logging
# os.environ['NO_TORCH_COMPILE'] = '1'
os.environ['NO_CUDA_GRAPH'] = '1'

class MoshiMimi:
    def __init__(
        self,
        model_safetensors_path: Union[Path, str],
        device: Any = None,
        sampling_rate: int = 24000,
        frame_shift: Seconds = 0.08,
        num_quantizers: int = 8,
        audio_chunk_size: int = 4*60*24000
    ) -> None:
        # Instantiate a pretrained model
        assert sampling_rate == 24000
        assert num_quantizers in [8, 32]
        assert frame_shift == 0.08

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
        if isinstance(device, int):
            device = torch.device("cpu") if device == -1 else torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, torch.device):
            pass
        else:
            raise ValueError(f"Illegal device is used, expected: int, str(cpu, cuda:0) or None, but get {device}")

        self._device = device

        codec = loaders.get_mimi(model_safetensors_path, device)
        logging.warning('MoshiMimi only support 8 codebook for now. If you need 32 codebook, please rewrite moshi.models.get_mimi function.') 
        self.codec = codec
        self.dtype = torch.bfloat16

        self.frame_shift = frame_shift
        self.num_quantizers = num_quantizers
        self.sample_rate = sampling_rate
        self.chunk_size = audio_chunk_size

    @property
    def device(self):
        return self._device
    
    @torch.inference_mode()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        # wav.shape == [1, 1, T]
        
        if not isinstance(wav, torch.Tensor):
            raise AssertionError
        
        if wav.ndim != 3:
            wav = wav.contiguous().view(1, 1, -1)
            
        if wav.shape[-1] < self.chunk_size:
            codes = self.codec.encode(wav.to(device))
        else:
            with self.codec.streaming(1):
                codes_list = []
                for i in range(0, wav.shape[-1], self.chunk_size):
                    audio_chunk = wav[..., i : i+self.chunk_size]
                    codes = self.codec.encode(audio_chunk.to(device))
                    codes_list.append(codes)
                codes = torch.cat(codes_list, dim=-1)

        if isinstance(codes, torch.Tensor) and torch.is_floating_point(codes):
            return codes

        return codes.type(torch.int16).permute(0, 2, 1)  # BxTxQ

if __name__ == "__main__":
    import argparse
    import numpy as np
    from tqdm import tqdm
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", default=None, type=str, help='filelist.txt include the audio absolute path', required = True)
    parser.add_argument('--codes_save_folder_path', default=None, type=str, help='codes save folder path ', required=True)
    parser.add_argument("--model_ckpt_path", type=str, default='/data0/questar/users/wuzhiyue/tmp/hf_mimi_to_moshi/model.safetensors')

    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--audio_chunk_length", default=4*60*24000, type=int)
    args = parser.parse_args()
    
    # wav_path_list = ['/home/wuzhiyue/moshi_test/Introducing_GPT-4o.wav', '/home/wuzhiyue/moshi_test/Introducing_GPT-4o.wav']
    filist_path = Path(args.filelist)
    wav_path_list = [filist_path.parent / line.strip() for line in filist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    
    device = args.device
    audio_chunk_length = args.audio_chunk_length
    tokenizer = MoshiMimi(args.model_ckpt_path,
                        device = device,
                        sampling_rate=24000, 
                        num_quantizers=8,
                        audio_chunk_size=audio_chunk_length)
    
    
    with torch.no_grad():
        for path in tqdm(wav_path_list):
            audio, sr = torchaudio.load(path)
            if sr != 24000:
                audio = torchaudio.transforms.Resample(sr, 24000)(audio)
            
            if audio.shape[0] > 1:
                audios = [audio[0], audio[1]] #男女
            else:
                audios = [audio]
                
            from time import time
            start = time()
            codes_list = []
            for index, audio in enumerate(audios):
                codes = tokenizer.encode(audio)
                time_consume = time() - start
                if index == 0:
                    file_name = path.split('/')[-1][:-4]
                    np.save(os.path.join(args.codes_save_folder_path, f'{file_name}_male.npy'), codes.cpu().numpy())
                elif index == 1:
                    file_name = path.split('/')[-1][:-4]
                    np.save(os.path.join(args.codes_save_folder_path, f'{file_name}_female.npy'), codes.cpu().numpy())
                    
            print(f'{path}: {time_consume}')

        
    