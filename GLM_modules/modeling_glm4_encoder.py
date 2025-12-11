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
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
from tqdm import tqdm

from utils.helpers import set_logging, waiting_for_debug, load_audio, save_audio, find_audio_files

from speech_tokenizer.utils import extract_speech_token

class GLM4Encoder(nn.Module):
    def __init__(self, generator_params):
        super().__init__()
        tokenizer_path = generator_params['glm4_encoder_config']['tokenizer_path']

        self.sample_rate = 16000        
        self.whisper_vqmodel = WhisperVQEncoder.from_pretrained(tokenizer_path)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
    
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
        
            
            
            
            
            