import os
import random
import unittest
from moshi.moshi.models import loaders
import numpy as np
import torch
import torchaudio

os.environ['NO_CUDA_GRAPH'] = '1'
    

class MoshiMimiStreaming(unittest.TestCase):
    def test_moshi_mimi_streaming(self, audio_path: str = '/home/wuzhiyue/moshi_test/Introducing_GPT-4o.wav'):
        device_id = 0 if torch.cuda.is_available() else -1
        assert device_id > -1
        device = f"cuda:{device_id}"

        bs = 1
        model_weight = '/data0/questar/users/wuzhiyue/tmp/hf_mimi_to_moshi/model.safetensors'
        seed = 424242
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        codec = loaders.get_mimi(model_weight, device)

        
        audio, sr = torchaudio.load(audio_path)
        if sr != 24000:
            audio = torchaudio.transforms.Resample(sr, 24000)(audio).to(device).to(torch.float32)
            if audio.shape[0] > 1:
                audio = audio[0].reshape(1, 1, -1)[:, :, :60 * 24000]
            else:
                audio = audio.reshape(1, 1, -1)[:, :, :60 * 24000]
        
        audio_copy_gt_codes = audio.clone()
        audio_copy_embedding = audio.clone()
        audio_copy_embedding_gt = audio_copy_embedding.clone()

        with torch.no_grad():
            with codec.streaming(1):
                codes_list = []
                for i in range(0, audio.shape[-1], 15 * 24000):
                    audio_chunk = audio[..., i : i + 15 * 24000]
                    codes = codec.encode(audio_chunk)
                    codes_list.append(codes)
                codes = torch.cat(codes_list, dim=-1)
                
                
            with codec.streaming(1):
                codes_list_gt = []
                for i in range(0, audio_copy_gt_codes.shape[-1], 1920):
                    audio_chunk_gt = audio_copy_gt_codes[..., i : i + 1920]
                    codes_gt = codec.encode(audio_chunk_gt)
                    codes_list_gt.append(codes_gt)
                codes_gt = torch.cat(codes_list_gt, dim=-1)
            
            print(f'the codes diff nums between gt and test is {torch.sum(codes != codes_gt)}')
            print(f'the codes diff ratios between gt and test is {torch.sum(codes != codes_gt) / torch.prod(torch.tensor(codes.shape))}')
            
            
        with torch.no_grad():
            with codec.streaming(1):
                embed_list = []
                for i in range(0, audio_copy_embedding.shape[-1], 15 * 24000):
                    audio_chunk = audio_copy_embedding[..., i : i + 15 * 24000]
                    embed = codec._encode_to_unquantized_latent(audio_chunk)
                    embed_list.append(embed)
                embeddings = torch.cat(embed_list, dim=-1)
                
            with codec.streaming(1):
                embed_list_gt = []
                for i in range(0, audio_copy_embedding_gt.shape[-1], 1920):
                    audio_chunk_gt = audio_copy_embedding_gt[..., i : i + 1920]
                    embed_gt = codec._encode_to_unquantized_latent(audio_chunk_gt)
                    embed_list_gt.append(embed_gt)
                embeddings_gt = torch.cat(embed_list_gt, dim=-1)
            
            assert torch.allclose(embeddings, embeddings_gt, atol=2e-2)
                

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--audio_path", default=None, type=str, help='audio path for test', required = True)
    # args = parser.parse_args()
    
    # unittest.main(args.audio_path)
    unittest.main()