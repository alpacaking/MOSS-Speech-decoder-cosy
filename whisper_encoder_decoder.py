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
from transformers import WhisperFeatureExtractor,AutoTokenizer
import sys
from tqdm import tqdm
from collections import OrderedDict
sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/GLM_modules')
from flow_inference import AudioDecoder
from speech_tokenizer.modeling_whisper import WhisperVQEncoder,WhisperVQConfig
from speech_tokenizer.utils import *
import torch
import torchaudio
import uuid
import torchaudio.compliance.kaldi as kaldi
# sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec')
# from whisper_feat_extractor import WhisperFeatureExtractor


class GLM4Encoder(nn.Module):
    def __init__(self, tokenizer_path,block_stream=False,mel_cache_len=8):
        super().__init__()
        tokenizer_path = tokenizer_path
        self.sample_rate = 16000        
        if block_stream==True:
            from speech_tokenizer.modeling_whisper_o import WhisperVQEncoder,WhisperVQConfig
            config = WhisperVQConfig.from_pretrained("/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/config_origin.json")
            self.whisper_vqmodel = WhisperVQEncoder(config)
        else:
            from speech_tokenizer.modeling_whisper import WhisperVQEncoder,WhisperVQConfig
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
        self.flow_path = "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/flow"
        self.audio_decoder = AudioDecoder(
            config_path=os.path.join(self.flow_path, "config.yaml"),
            flow_ckpt_path=os.path.join(self.flow_path, "flow.pt"),
            hift_ckpt_path=os.path.join(self.flow_path, "hift.pt"),
            campplus_model='{}/campplus.onnx'.format(self.flow_path),
            mel_cache_len=mel_cache_len
        )
        self.audio_decoder = self.audio_decoder.eval()
        
    # @property
    # def sample_rate(self) -> int:
    #     return 16000
    
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
    def encode_token_test(self, audio_path):
        codebook = self.whisper_vqmodel.codebook # (V, D)
        audio_tokens = extract_speech_token_test(self.whisper_vqmodel, self.feature_extractor, [audio_path])[0] # (T, )
        return audio_tokens
    
    @torch.no_grad()
    def encode_token_streaming(self, audio_path):
        codebook = self.whisper_vqmodel.codebook # (V, D)
        audio_tokens = extract_speech_token_streaming(self.whisper_vqmodel, self.feature_extractor, [audio_path])[0] # (T, )
        return audio_tokens
    
    @torch.inference_mode()
    def encode(self, wav_list, device: torch.device = torch.device("cuda")):
        """
        将一批音频编码为离散 token

        Args:
            wav_list: List[torch.Tensor]  # 一批音频波形，每个 tensor shape 为 [T]
            device: torch.device  # 计算设备，默认 CUDA

        Returns:
            Dict[str, Any]:
                - codes_list: List[torch.Tensor]  # 一批编码后的 token
                                                # 每个 tensor shape 为 [nq, T]
                                                # nq 是量化器数量，T 是时间步长度
        """
        audio_tokens = extract_speech_token(self.whisper_vqmodel, self.feature_extractor, wav_list) # (T, )
        audio_tokens = [
            torch.tensor(token).unsqueeze(0) if torch.tensor(token).ndim < 2 
            else torch.tensor(token)
            for token in audio_tokens
        ]
        return {
            "codes_list": audio_tokens
        }
    
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
    
    @torch.no_grad()
    def _extract_speech_feat(self, speech):
        speech_feat = self.audio_decoder.feat_extractor(speech).squeeze(dim=0).transpose(0, 1)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32)
        return speech_feat, speech_feat_len
    
    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.audio_decoder.campplus_session.run(None,
                                              {self.audio_decoder.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding])
        return embedding

    
    @torch.no_grad()
    def decode(self,codes_list, prompt_speech=None, prompt_speech_sample_rate=None, use_spk_embedding=True,use_prompt_speech=True,device: torch.device = torch.device("cuda")):
        assert os.path.exists(prompt_speech)
        prompt_speech_wav,origin_sample_rate=torchaudio.load(prompt_speech)
        # elif isinstance(prompt_speech,torch.Tensor):
        #     assert prompt_speech_sample_rate
        #     prompt_speech_wav=prompt_speech_wav
        #     origin_sample_rate=prompt_speech_sample_rate
        if self.audio_decoder.sample_rate != origin_sample_rate:
            prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=origin_sample_rate, new_freq=self.audio_decoder.sample_rate)(prompt_speech_wav)
        else:
            prompt_speech_resample=prompt_speech_wav
        speech_token = torch.tensor(self.encode_token(prompt_speech)).unsqueeze(0)
        speech_token_len=torch.tensor(speech_token.shape[-1]).unsqueeze(0).unsqueeze(0)
        speech_feat, speech_feat_len=self._extract_speech_feat(prompt_speech_resample)
        if self.audio_decoder.sample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 4), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :4 * token_len], 4 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        prompt_speech_16k =torchaudio.transforms.Resample(orig_freq=self.audio_decoder.sample_rate, new_freq=16000)(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        speech_feat=speech_feat
        speech_feat_len=speech_feat_len
        syn_wav_list = []  # B * (T_wav,)  # T_wav是生成的音频长度
        for codes in codes_list:  # codes: (1, T)
            if isinstance(codes,list):
                codes=torch.tensor(codes).unsqueeze(0)
            this_uuid = str(uuid.uuid4())
            # prompt_speech_feat = torch.zeros(1, 0, 80).to(device)  # (1, 0, 80)
            # flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)  # (1, 0)
            if use_spk_embedding and use_prompt_speech:
                tts_speech, _ = self.audio_decoder.token2wav( # ???
                    codes,  # (1, T)
                    uuid=this_uuid,
                    prompt_token=speech_token,  # (1, 0)
                    prompt_feat=speech_feat,  # (1, 0, 80)
                    embedding=embedding,
                )  # tts_speech: (1, T_wav)
            elif use_prompt_speech and not use_spk_embedding:
                tts_speech, _ = self.audio_decoder.token2wav( # ???
                    codes,  # (1, T)
                    uuid=this_uuid,
                    prompt_token=speech_token,  # (1, 0)
                    prompt_feat=speech_feat,  # (1, 0, 80)
                )  # tts_speech: (1, T_wav)
            elif not use_prompt_speech and use_spk_embedding:
                tts_speech, _ = self.audio_decoder.token2wav( # ???
                    codes,  # (1, T)
                    uuid=this_uuid,
                    embedding=embedding,
                )  # tts_speech: 1, T_wav)
            else:
                tts_speech, _ = self.audio_decoder.token2wav( # ???
                    codes,  # (1, T)
                    uuid=this_uuid,
                )  # tts_speech: (1, T_wav)
            syn_wav_list.append(tts_speech.squeeze())  # (T_wav,)
            
        return {
            "syn_wav_list": syn_wav_list  # B * (T_wav,)
        }
        
    @torch.no_grad()
    def decode_streaming(self,codes_list,prompt_speech=None, prompt_speech_sample_rate=None, use_spk_embedding=True,use_prompt_speech=True,block_size=5, max_token_len=None):
        assert os.path.exists(prompt_speech)
        prompt_speech_wav,origin_sample_rate=torchaudio.load(prompt_speech)
        # elif isinstance(prompt_speech,torch.Tensor):
        #     assert prompt_speech_sample_rate
        #     prompt_speech_wav=prompt_speech_wav
        #     origin_sample_rate=prompt_speech_sample_rate
        if self.audio_decoder.sample_rate != origin_sample_rate:
            prompt_speech_resample = torchaudio.transforms.Resample(
                orig_freq=origin_sample_rate, 
                new_freq=self.audio_decoder.sample_rate
            )(prompt_speech_wav)
        else:
            prompt_speech_resample=prompt_speech_wav
        speech_token = torch.tensor(self.encode_token(prompt_speech)).unsqueeze(0)
        speech_token_len=torch.tensor(speech_token.shape[-1]).unsqueeze(0).unsqueeze(0)
        speech_feat, speech_feat_len=self._extract_speech_feat(prompt_speech_resample)
        if self.audio_decoder.sample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 4), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :4 * token_len], 4 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        prompt_speech_16k =torchaudio.transforms.Resample(orig_freq=self.audio_decoder.sample_rate, new_freq=16000)(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        syn_wav_list = []
        for codes in codes_list:  # codes: (1, T)
            if isinstance(codes,list):
                codes=torch.tensor(codes).unsqueeze(0)
            this_uuid = str(uuid.uuid4())
            # prompt_speech_feat = torch.zeros(1, 0, 80).to(device)  # (1, 0, 80)
            # flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)  # (1, 0)
            tts_speech= self.audio_decoder.stream_inference( # ???
                codes,  # (1, T)
                prompt_token=speech_token,
                prompt_feat=speech_feat, 
                embedding=embedding,
                block_size=block_size,
                max_token_len=max_token_len
            )  # tts_speech: (1, T_wav)
            syn_wav_list.append(tts_speech.squeeze())  # (T_wav,)
            
        return {
            "syn_wav_list": syn_wav_list  # B * (T_wav,)
        }
        
if __name__=='__main__':
    import os
    import random
    import numpy as np
    import torch

    def set_seed(seed: int) -> None:
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer.")

        print(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False # 禁用cudnn的自动优化，可能影响性能
        else:
            torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # 影响Python哈希函数的随机性
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1' # TensorFlow相关的CUDNN确定性设置

    set_seed(42)
    encoder=GLM4Encoder(tokenizer_path='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt',mel_cache_len=8).to('cuda')
    
    # result2=encoder.encode_batch_token(['/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000025_000019.wav',
    #  '/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000027_000007.wav',
    #  '/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000034_000002.wav',
    #  '/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000017_000001.wav'])
    result1=encoder.encode_token('/inspire/hdd/project/embodied-multimodality/public/datasets/haitianruisheng_3/segment_00_000/00460fd0bd40915bd31e79976176c9b5_001.mp3')
    # print(result1)
    # print(result2[0])
    #result2=[4475, 13838, 11506, 14897, 10663, 1524, 11784, 12044, 9670, 11835, 3666, 14129, 7053, 14623, 4579, 15218, 15403, 11839, 7214, 6115, 13866, 6258, 1597, 1597, 3431, 8934, 12988, 9813, 13928, 3538, 14749, 1905, 14985, 5156, 10999, 3089, 6756, 10325, 15147, 6621, 816, 6621, 5710, 6090]
    wav=encoder.decode([result1],prompt_speech='/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000027_000007.wav', use_spk_embedding=True,use_prompt_speech=True)
    # # prompt_speech_wav,origin_sample_rate=torchaudio.load('/inspire/hdd/project/embodied-multimodality/public/lzjjin/prompt.wav')
    # # prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=origin_sample_rate, new_freq=24000)(prompt_speech_wav)
    # # torchaudio.save('result1.wav',torch.cat([prompt_speech_resample,wav['syn_wav_list'][0].unsqueeze(0).cpu()],dim=-1),sample_rate=24000)
    torchaudio.save('result1.wav',wav['syn_wav_list'][0].unsqueeze(0).cpu(),sample_rate=24000)
    
    wav=encoder.decode_streaming([result1],prompt_speech='/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000027_000007.wav',block_size=5,max_token_len=40)
    torchaudio.save('result2.wav',wav['syn_wav_list'][0].unsqueeze(0).cpu(),sample_rate=24000)
    