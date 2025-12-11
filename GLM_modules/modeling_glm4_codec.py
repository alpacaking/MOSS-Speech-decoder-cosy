import os
import torch
import logging
import uuid

from typing import List, Dict, Any, Optional
from torch import nn
from transformers import WhisperFeatureExtractor
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from flow_inference import AudioDecoder

class GLM4Codec(nn.Module):
    """GLM-4 Voice 的编解码器,用于音频编码和解码。
    
    该类实现了:
    1. 音频到离散 token 的编码 
    2. 离散 token 到音频的解码
    """
    
    def __init__(self):
        super().__init__()
        
        # 基础配置
        self.encoder_path = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/glm-4-voice-tokenizer/"
        self.flow_path = "/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/glm-4-voice-decoder/"
        self.sample_rate = 16000  # 输入采样率
        self.output_sample_rate = 22050  # 输出采样率
        self.nq = 1  # 量化器数量
        
        # 加载编码器模型和特征提取器
        self.whisper_vqmodel = WhisperVQEncoder.from_pretrained(self.encoder_path)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.encoder_path)
        
        # 加载解码器
        self.audio_decoder = AudioDecoder(
            config_path=os.path.join(self.flow_path, "config.yaml"),
            flow_ckpt_path=os.path.join(self.flow_path, "flow.pt"),
            hift_ckpt_path=os.path.join(self.flow_path, "hift.pt"),
        )
        
        # 将模型移至 GPU
        self.whisper_vqmodel = self.whisper_vqmodel.eval()
        self.audio_decoder = self.audio_decoder.eval()
    
        self.pooling_kernel_size = self.whisper_vqmodel.config.pooling_kernel_size or 1
        self.stride = self.whisper_vqmodel.conv1.stride[0] * self.whisper_vqmodel.conv2.stride[0] * self.pooling_kernel_size * self.feature_extractor.hop_length
    
    @torch.inference_mode()
    def encode(self, wav_list: List[torch.Tensor], device: torch.device = torch.device("cuda")) -> Dict[str, Any]:
        """
        Args:
            wav_list: B * (T,)  # B是batch size, T是每个音频的长度
        """
        # 准备特征
        wav_list = [w.squeeze().cpu().numpy() for w in wav_list] # B * (T, )
        features = self.feature_extractor(
            wav_list,
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=self.stride
        ).to(device)
        # features.input_features: (B, 80, T//hop_length)  # 80是mel频谱维度
        # features.attention_mask: (B, T//hop_length)
        
        # 编码
        outputs = self.whisper_vqmodel(**features)
        speech_tokens = outputs.quantized_token_ids  # (B, T//stride)  # stride = conv1.stride * conv2.stride
        
        # 处理 attention mask
        attention_mask = features.attention_mask  # (B, T//hop_length)
        attention_mask = attention_mask[:, ::self.whisper_vqmodel.conv1.stride[0] * self.whisper_vqmodel.conv2.stride[0]]  # (B, T//stride), 把 mel 的 attention mask 变成 encoder 输出 的 attention mask
        if self.whisper_vqmodel.config.pooling_kernel_size: # encoder output -> quantizer token (4 倍下采样)
            attention_mask = attention_mask[:, ::self.whisper_vqmodel.config.pooling_kernel_size]  # (B, T//(stride*pool_size))
        # ??? attention_mask 是什么，为什么要切片
        
        # 提取有效 token
        codes_list = []  # B * (1, T_valid)  # T_valid 是每个样本实际的token长度(去掉padding)
        for tokens, mask in zip(speech_tokens, attention_mask):  # tokens: (T//stride), mask: (T//stride)
            valid_tokens = tokens[mask.bool()].unsqueeze(0)  # (1, T_valid)
            codes_list.append(valid_tokens)
                
        return {
            "codes_list": codes_list  # B * (1, T_valid)
        }

    @torch.inference_mode()
    def decode(self, codes_list: List[torch.Tensor], device: torch.device = torch.device("cuda")) -> Dict[str, Any]:
        """
        Args:
            codes_list: B * (1, T)  # B是batch size, T是每个样本的token序列长度
        """
        syn_wav_list = []  # B * (T_wav,)  # T_wav是生成的音频长度
        
        for codes in codes_list:  # codes: (1, T)
            # 初始化提示信息
            this_uuid = str(uuid.uuid4())
            prompt_speech_feat = torch.zeros(1, 0, 80).to(device)  # (1, 0, 80)
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)  # (1, 0)
            
            # 解码
            tts_speech, _ = self.audio_decoder.token2wav( # ???
                codes,  # (1, T)
                uuid=this_uuid,
                prompt_token=flow_prompt_speech_token,  # (1, 0)
                prompt_feat=prompt_speech_feat,  # (1, 0, 80)
                finalize=True
            )  # tts_speech: (1, T_wav)
            
            syn_wav_list.append(tts_speech.squeeze())  # (T_wav,)
            
        return {
            "syn_wav_list": syn_wav_list  # B * (T_wav,)
        }
    @classmethod
    def load_from_checkpoint(cls):
        """从检查点加载模型
        
        Returns:
            GLM4Codec: 加载好的模型实例
        """
        model = cls()
        return model