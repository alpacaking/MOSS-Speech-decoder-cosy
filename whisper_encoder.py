import os
import argparse
import uuid
import torch
import torchaudio
import logging
import os
import io
import glob
import math
import tarfile
import torch
import torchaudio
import safetensors
from torch import nn
from transformers import WhisperFeatureExtractor, AutoTokenizer
import sys
from tqdm import tqdm
from collections import OrderedDict
sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/GLM_modules')
from speech_tokenizer.modeling_whisper import WhisperVQEncoder,WhisperVQConfig
from speech_tokenizer.utils import extract_speech_token
import torch
import uuid
class GLM4Encoder(nn.Module):
    def __init__(self, tokenizer_path):
        super().__init__()
        tokenizer_path = tokenizer_path

        self.sample_rate = 16000        
        config = WhisperVQConfig.from_pretrained("/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/config.json")
        self.whisper_vqmodel = WhisperVQEncoder(config)
        ckpt=torch.load(tokenizer_path)
        state_dict = ckpt['generator']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                new_k = k[len("encoder."):]  
                new_state_dict[new_k] = v
        missing_keys, unexpected_keys = self.whisper_vqmodel.load_state_dict(new_state_dict, strict=False)
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/glm-4-voice-tokenizer')
    
    @torch.no_grad()
    def forward(self, audio_paths):
        B = len(audio_paths)
        codebook = self.whisper_vqmodel.codebook # (V, D)
        V, D = codebook.weight.shape
        hidden_states = [] # B * (T, D)
        device = next(self.parameters()).device
        for i, audio_path in enumerate(audio_paths):
            audio_tokens = extract_speech_token(self.whisper_vqmodel, self.feature_extractor, [audio_path])[0] # (T, )
            embeddings = codebook.weight[torch.tensor(audio_tokens)] # (T, D)
            hidden_states.append(embeddings)
        output_length = 375
        output_lengths = torch.tensor([hidden_state.shape[0] for hidden_state in hidden_states], device=device)
        output_hidden_states = torch.zeros((B, output_length, D), device=device)  # (B, T, D)
        for i, hidden_state in enumerate(hidden_states):
            hidden_state = hidden_state[:output_length, :] # (T, D)
            output_hidden_states[i, :hidden_state.shape[0], :] = hidden_state # (B, T, D)
        
        output_hidden_states = output_hidden_states.transpose(1, 2) # (B, D, T)
        
        return output_hidden_states, output_lengths # (B, T, D), (B, )
    
    @torch.no_grad()
    def encode_token(self, audio_path):
        codebook = self.whisper_vqmodel.codebook # (V, D)
        audio_tokens = extract_speech_token(self.whisper_vqmodel, self.feature_extractor, [audio_path])[0] # (T, )
        return audio_tokens
    
    @torch.no_grad()
    def encode_batch_token(self, audio_path_list,batch_size=128):
        codebook = self.whisper_vqmodel.codebook # (V, D)
        audio_tokens = extract_speech_token(self.whisper_vqmodel, self.feature_extractor,audio_path_list) # (T, )
        return audio_tokens
    
    # @torch.no_grad()
    # def decode(self,codes_list, device: torch.device = torch.device("cuda")):
    #     syn_wav_list = []  # B * (T_wav,)  # T_wav是生成的音频长度
    #     for codes in codes_list:  # codes: (1, T)
    #         this_uuid = str(uuid.uuid4())
    #         prompt_speech_feat = torch.zeros(1, 0, 80).to(device)  # (1, 0, 80)
    #         flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)  # (1, 0)
    #         tts_speech, _ = self.audio_decoder.token2wav( # ???
    #             codes,  # (1, T)
    #             uuid=this_uuid,
    #             prompt_token=flow_prompt_speech_token,  # (1, 0)
    #             prompt_feat=prompt_speech_feat,  # (1, 0, 80)
    #             finalize=True
    #         )  # tts_speech: (1, T_wav)
    #         syn_wav_list.append(tts_speech.squeeze())  # (T_wav,)
            
    #     return {
    #         "syn_wav_list": syn_wav_list  # B * (T_wav,)
    #     }
            
if __name__=='__main__':
    encoder=GLM4Encoder(tokenizer_path='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt').to('cuda')
    result=encoder.encode_token('/inspire/hdd/project/embodied-multimodality/public/datasets/haitianruisheng_3/segment_00_034/002c566939aec5113f6e197e74edf6f0_034.mp3')
    result2=encoder.encode_batch_token(['/inspire/hdd/project/embodied-multimodality/public/datasets/haitianruisheng_3/segment_00_034/002c566939aec5113f6e197e74edf6f0_034.mp3',
     '/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000027_000007.wav',
     '/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000034_000002.wav',
     '/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000017_000001.wav'])
    print(result)        
    print(result2[0])