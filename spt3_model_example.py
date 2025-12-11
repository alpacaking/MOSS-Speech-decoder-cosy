# -*- coding: utf-8 -*-
import yaml
import copy
import logging
import torch.nn as nn
import torch
import torch.nn.functional as F

from contextlib import contextmanager

from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperProcessor

from speechtokenizer.audiodec_modules.decoder import HifiGANDecoder 

from speechtokenizer.modules.seanet import SEANetDecoder

from speechtokenizer.baichuan_modules.audio_modeling_omni import OmniAudioEncoder, OmniAudioDecoder, ResidualDownConv, UpConv, Transformer
from speechtokenizer.baichuan_modules.vector_quantize import ResidualVQ, SplitResidualVQ, SpecializedResidualVQ
from speechtokenizer.baichuan_modules.mel_feature_extractor import MelFeatureExtractor

from speechtokenizer.maskgct_modules.vocos import Vocos

from speechtokenizer.moshi_modules.modules import ProjectedTransformer as MoshiTransformer
from speechtokenizer.moshi_modules.modules import SEANetEncoder as MoshiSEANetEncoder
from speechtokenizer.moshi_modules.modules import SEANetDecoder as MoshiSEANetDecoder
from speechtokenizer.moshi_modules.modules import ConvDownsample1d as MoshiConvDownsample1d
from speechtokenizer.moshi_modules.modules import ConvTrUpsample1d as MoshiConvTrUpsample1d
from speechtokenizer.moshi_modules.modules.streaming import StreamingModule

from speechtokenizer.modules.reshaped_module import PatchedPretransform
from utils.debug_utils import *

class SpeechTokenizer(nn.Module):
    def __init__(self, generator_params):
        super().__init__()
        # 基本参数
        self.version = generator_params['version']
        assert self.version in ['3.23.1.b', '3.23.1.d', '3.23.2.b', '3.23.2.d', '3.23.1.e', '3.23.2.e', '3.23.1.f', '3.23.1.g', '3.23.1.h', '3.23.1.m', '3.23.2.m', '3.23.1.n', '4.26.1.a', '4.26.2.a', '4.26.1.b', '4.26.1.c', '4.26.1.d'], \
                                f"版本 {self.version} 不在支持的列表中"

        if self.version in ['4.26.1.a']:
            self.sample_rate = generator_params['sample_rate']
            self.sampling_rate = self.sample_rate  # 采样率
            self.pre_stage_ckpt_file = generator_params['pre_stage_ckpt_file']  # 预训练权重文件路径
            self.downsample_rate = generator_params['downsample_rate']
            self.code_dim = generator_params['quantizer_kwargs']['input_dim']
            self.causal_transformer_context_duration = generator_params['causal_transformer_context_duration']

            if 'skip_layer_frame_rate' in generator_params:
                self.skip_layer_frame_rate = generator_params['skip_layer_frame_rate']
                logging.info(f"Using skip layer in encoder, {self.skip_layer_frame_rate = }")
            else:
                self.skip_layer_frame_rate = None
                logging.info(f"Not using skip layer in encoder, {self.skip_layer_frame_rate = }")

            ## Codec 部分
            current_frame_rate = self.sample_rate
            
            ### Encoder
            self.encoder_kwargs = generator_params['encoder_kwargs']
            self.encoder = nn.ModuleList()
            for encoder_kwargs_i in self.encoder_kwargs:
                if encoder_kwargs_i['module_type'] == 'PatchedPretransform':
                    self.encoder.append(
                        PatchedPretransform(
                            **encoder_kwargs_i,
                            is_downsample=True
                        )
                    )
                    current_frame_rate = current_frame_rate / self.encoder[-1].downsample_ratio
                elif encoder_kwargs_i['module_type'] == 'Transformer':
                    self.encoder.append(
                        MoshiTransformer(
                            **encoder_kwargs_i,
                            context=int(current_frame_rate * self.causal_transformer_context_duration)
                        )
                    )
                    current_frame_rate = current_frame_rate / self.encoder[-1].downsample_ratio
                else:
                    assert False
            
            quantizer_frame_rate = current_frame_rate
            ### Quantizer
            if generator_params['quantizer_kwargs']['quantizer_type'] == "rvq":
                self.quantizer = ResidualVQ(**generator_params['quantizer_kwargs'])
            elif generator_params['quantizer_kwargs']['quantizer_type'] == "spec_rvq":
                self.quantizer = SpecializedResidualVQ(**generator_params['quantizer_kwargs'])
            else:
                assert False
            
            ### Decoder
            self.decoder_kwargs = copy.deepcopy(generator_params['reversed_decoder_kwargs'][::-1])
            self.decoder = nn.ModuleList()
            for decoder_kwargs_i in self.decoder_kwargs:
                if decoder_kwargs_i['module_type'] == 'PatchedPretransform':
                    self.decoder.append(
                        PatchedPretransform(
                            **decoder_kwargs_i,
                            is_downsample=False
                        )
                    )
                    current_frame_rate = current_frame_rate * self.decoder[-1].downsample_ratio
                elif decoder_kwargs_i['module_type'] == 'Transformer':
                    # Decoder 和 Encoder 对称，因此输入输出的 dimension 需要发生变化
                    decoder_kwargs_i['input_dimension'], decoder_kwargs_i['output_dimension'] = decoder_kwargs_i['output_dimension'], decoder_kwargs_i['input_dimension']
                    self.decoder.append(
                        MoshiTransformer(
                            **decoder_kwargs_i,
                            context=int(current_frame_rate * self.causal_transformer_context_duration)
                        )
                    )
                    current_frame_rate = current_frame_rate * self.decoder[-1].downsample_ratio

            assert int(current_frame_rate) == self.sample_rate, f"current_frame_rate = {current_frame_rate}, self.sample_rate = {self.sample_rate}"

            self.load_from_pre_state(self.pre_stage_ckpt_file)

        elif self.version in ['4.26.2.a']:
            self.sample_rate = generator_params['sample_rate']
            self.sampling_rate = self.sample_rate  # 采样率
            self.pre_stage_ckpt_file = generator_params['pre_stage_ckpt_file']  # 预训练权重文件路径
            self.downsample_rate = generator_params['downsample_rate']
            self.code_dim = generator_params['quantizer_kwargs']['input_dim']
            self.causal_transformer_context_duration = generator_params['causal_transformer_context_duration']
            
            ## Codec 部分
            current_frame_rate = self.sample_rate
            
            ### Encoder
            self.encoder_kwargs = generator_params['encoder_kwargs']
            self.encoder = nn.ModuleList()
            for encoder_kwargs_i in self.encoder_kwargs:
                if encoder_kwargs_i['module_type'] == 'PatchedPretransform':
                    self.encoder.append(
                        PatchedPretransform(
                            **encoder_kwargs_i,
                            is_downsample=True
                        )
                    )
                    current_frame_rate = current_frame_rate / self.encoder[-1].downsample_ratio
                elif encoder_kwargs_i['module_type'] == 'Transformer':
                    self.encoder.append(
                        MoshiTransformer(
                            **encoder_kwargs_i,
                            context=int(current_frame_rate * self.causal_transformer_context_duration)
                        )
                    )
                    current_frame_rate = current_frame_rate / self.encoder[-1].downsample_ratio
                else:
                    assert False

            for param in self.encoder.parameters():
                param.requires_grad = False

            ### Quantizer
            self.quantizer = ResidualVQ(**generator_params['quantizer_kwargs'])

            for param in self.quantizer.parameters():
                param.requires_grad = False

            ### Decoder
            self.decoder_kwargs = copy.deepcopy(generator_params['reversed_decoder_kwargs'][::-1])
            self.decoder = nn.ModuleList()
            for decoder_kwargs_i in self.decoder_kwargs:
                if decoder_kwargs_i['module_type'] == 'PatchedPretransform':
                    self.decoder.append(
                        PatchedPretransform(
                            **decoder_kwargs_i,
                            is_downsample=False
                        )
                    )
                    current_frame_rate = current_frame_rate * self.decoder[-1].downsample_ratio
                elif decoder_kwargs_i['module_type'] == 'Transformer':
                    # Decoder 和 Encoder 对称，因此输入输出的 dimension 需要发生变化
                    decoder_kwargs_i['input_dimension'], decoder_kwargs_i['output_dimension'] = decoder_kwargs_i['output_dimension'], decoder_kwargs_i['input_dimension']
                    self.decoder.append(
                        MoshiTransformer(
                            **decoder_kwargs_i,
                            context=int(current_frame_rate * self.causal_transformer_context_duration)
                        )
                    )
                    current_frame_rate = current_frame_rate * self.decoder[-1].downsample_ratio

            assert int(current_frame_rate) == self.sample_rate, f"current_frame_rate = {current_frame_rate}, self.sample_rate = {self.sample_rate}"
            
            self.load_from_pre_state(self.pre_stage_ckpt_file)

        else:
            assert False
    def load_from_pre_state(self, pre_stage_ckpt_file):
        # 加载预训练模型权重
        if pre_stage_ckpt_file is not None:
            logging.info(f"SpeechTokenizer 的预训练文件路径 = {pre_stage_ckpt_file}，尝试从预训练阶段加载...")
            pre_stage_ckpt = torch.load(pre_stage_ckpt_file, map_location="cpu", weights_only=False)
            load_result = self.load_state_dict(pre_stage_ckpt["generator"], strict=False)
            logging.info(f"成功加载版本 {self.version} 的 SpeechTokenizer 权重，从 {pre_stage_ckpt_file}!")
            logging.info(f"缺失的键: {load_result.missing_keys}")
            logging.info(f"意外的键: {load_result.unexpected_keys}")
        else:
            logging.info("SpeechTokenizer 的预训练文件路径为空，不加载预训练权重")

    def forward(
            self, 
            x: torch.Tensor, # (B, 1, T)
            input_lengths: torch.Tensor, # (B, )
            llm_inputs_info: dict, 
            output_recon: bool = True, 
            output_text: bool = False
        ):
        '''
        参数:
            x: 原始波形, (B, 1, T)
            input_lengths: 原始波形长度, (B, )
            llm_inputs_info: LLM 输入，需要在 dataset 和 dataloader 中提前准备好
            output_recon: 是否重建音频
            output_text: 是否做 ASR

        返回:
            y: 合成音频，形状为 (B, 1, T)
            vq_loss: 量化损失
            zq: 用于后续微调（如 CTC）的量化嵌入，形状为 (B, D, T)
            output_length: 合成音频长度, (B, )
            
            llm_loss: LLM 损失
            audio_features: 音频特征，形状为 (B, T, D), 用于适配 LLM 的推理接口
        '''
        if self.version in ['4.26.1.a', '4.26.1.b']:
            # Encoder
            current_frame_rate = self.sample_rate
            e, e_lengths = x, input_lengths
            
            for encoder_module_i in self.encoder:
                
                if self.skip_layer_frame_rate is not None and encoder_module_i.module_type == "Transformer" \
                    and self.skip_layer_frame_rate == current_frame_rate:
                        e_before_module = e.clone()
                
                e, e_lengths = encoder_module_i(e, e_lengths)
                
                if self.skip_layer_frame_rate is not None and encoder_module_i.module_type == "Transformer" \
                    and self.skip_layer_frame_rate == current_frame_rate:
                        e = e + e_before_module
                
                current_frame_rate = current_frame_rate / encoder_module_i.downsample_ratio
            
            encoder_output, encoder_output_length = e, e_lengths # (B, D, T)
            
            # Quantizer
            zq, codes, vq_loss, _, quantizer_output_length, rvq1_output = self.quantizer(encoder_output, encoder_output_length)  # zq 形状: (B, D, T), 12.5hz
            
            # Decoder
            d, d_lengths = zq, quantizer_output_length
            for decoder_module_i in self.decoder:
                d, d_lengths = decoder_module_i(d, d_lengths)
            decoder_output, decoder_output_length = d, d_lengths # (B, D, T)
            
            return {
                # Codec
                "y": decoder_output,  # (B, 1, T)
                "vq_loss": vq_loss.sum(),  # 量化损失
                "zq": zq,  # (B, D, T)
                "output_length": decoder_output_length,
            }
            
        elif self.version in ['4.26.2.a']:
            with torch.no_grad():
                # Encoder
                e, e_lengths = x, input_lengths
                for encoder_module_i in self.encoder:
                    e, e_lengths = encoder_module_i(e, e_lengths)
                encoder_output, encoder_output_length = e, e_lengths # (B, D, T)
                
                # Quantizer
                zq, codes, vq_loss, _, quantizer_output_length, _ = self.quantizer(encoder_output, encoder_output_length)  # zq 形状: (B, D, T), 12.5hz
            
            # Decoder
            d, d_lengths = zq, quantizer_output_length
            for decoder_module_i in self.decoder:
                d, d_lengths = decoder_module_i(d, d_lengths)
            decoder_output, decoder_output_length = d, d_lengths # (B, D, T)
            
            return {
                # Codec
                "y": decoder_output,  # (B, 1, T)
                "vq_loss": vq_loss.sum(),  # 量化损失
                "zq": zq,  # (B, D, T)
                "output_length": decoder_output_length,
            }
   
        else:
            assert False

    @torch.inference_mode()
    def inference(self, x: torch.Tensor, n_q = None):
        '''
        推理模式：
        - 冻结所有参数（包括 VQ）
        - 不使用 LLM（use_llm=False）
        - audio_attention_mask 设置为全 1
        - 返回包含 "zq" 的 dict

        参数:
            x : (B, 1, T)

        返回:
            dict : 包含以下 key-value pairs:
                "y": 合成音频，形状为 (B, 1, T)
                "vq_loss": 量化损失
                "zq": 量化嵌入，形状为 (B, D, T)
                "output_length": 输出长度
        '''
        if self.version in ['4.26.1.a', '4.26.1.b', '4.26.2.a']:
            # preprocess inputs
            device = x[0].device
            input_lengths = torch.tensor(
                [xi.shape[-1] for xi in x]
            ).to(device) # (B, )
            if x.shape[-1] % self.downsample_rate != 0:
                pad_lengths = self.downsample_rate - (x.shape[-1] % self.downsample_rate)
                x = torch.nn.functional.pad(x, (0, pad_lengths))

            # same as forward, and support variable nq
            # Encoder
            e, e_lengths = x, input_lengths
            for encoder_module_i in self.encoder:
                e, e_lengths = encoder_module_i(e, e_lengths)
            encoder_output, encoder_output_length = e, e_lengths # (B, D, T)
            
            # Quantizer
            zq, codes, vq_loss, _, quantizer_output_length, _ = self.quantizer(encoder_output, encoder_output_length)  # zq 形状: (B, D, T)
            if n_q:
                codes = codes[:n_q, :, :] # (n_q, B, T)
                zq = self.quantizer.decode_codes(codes) # (B, D, T)
                vq_loss = vq_loss[:n_q] # (n_q)

            # Decoder
            d, d_lengths = zq, quantizer_output_length
            for decoder_module_i in self.decoder:
                d, d_lengths = decoder_module_i(d, d_lengths)
            decoder_output, decoder_output_length = d, d_lengths # (B, D, T)
            
            llm_loss = None
            audio_features = None
            
            return {
                # Codec
                "y": decoder_output,  # (B, 1, T)
                "vq_loss": vq_loss.sum(),  # 量化损失
                "zq": zq,  # (B, D, T)
                "output_length": decoder_output_length,
                
                # LLM
                "llm_loss": llm_loss,
                "audio_features": audio_features # LLM 的 audio 部分的真实输入 # (B, D, T)
            }
            
        


            forward_result = self(
                x=x, # (B, 1, T)
                input_lengths=input_lengths, # (B, )
                llm_inputs_info=None
            )
            return forward_result
        
        else:
            assert False

    @torch.inference_mode()
    def inference_tokenize(self, x, input_lengths):
        """
            Input:
                x: Waveform tensor # (B, 1, T), T <= 30s * sample_rate
                input_lengths: Valid length for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "zq": Quantized embeddings # (B, D, T)
                    "codes": Quantization codes # (nq, B, T)
                    "codes_lengths": Quantization code lengths # (B,)
        """
        if self.version in ['4.26.1.a', '4.26.1.b', '4.26.2.a']:
            # preprocess inputs
            device = x[0].device
            if x.shape[-1] % self.downsample_rate != 0:
                pad_lengths = self.downsample_rate - (x.shape[-1] % self.downsample_rate)
                x = torch.nn.functional.pad(x, (0, pad_lengths))

            # same as forward, and support variable nq
            # Encoder
            e, e_lengths = x, input_lengths
            for encoder_module_i in self.encoder:
                e, e_lengths = encoder_module_i(e, e_lengths)
            encoder_output, encoder_output_length = e, e_lengths # (B, D, T)
            
            # Quantizer
            zq, codes, vq_loss, _, quantizer_output_length, _ = self.quantizer(encoder_output, encoder_output_length)  # zq 形状: (B, D, T)
            
            return {
                "zq": zq, # (B, D, T)
                "codes": codes, # (nq, B, T)
                "codes_lengths": quantizer_output_length # (B,)
            }
        else:
            assert False
      
    @torch.inference_mode()  
    def inference_detokenize(self, codes, codes_lengths):
        """
            Input:
                codes: Quantization codes # (nq, B, T)
                codes_lengths: Quantization code lengths for each sample # (B,)
            Output:
                dict: Contains the following key-value pairs
                    "y": Synthesized audio waveform # (B, 1, T)
                    "output_length": Output lengths # (B,)
        """
        if self.version in ['4.26.1.a', '4.26.1.b', '4.26.2.a']:
            # Decoder
            zq = self.quantizer.decode_codes(codes) # (B, D, T)
            quantizer_output_length = codes_lengths # (B, )
            
            d, d_lengths = zq, quantizer_output_length
            for decoder_module_i in self.decoder:
                d, d_lengths = decoder_module_i(d, d_lengths)
            decoder_output, decoder_output_length = d, d_lengths # (B, D, T)
            
            return {
                "y": decoder_output, # (B, 1, T)
                "output_length": decoder_output_length # (B, )
            }
            
        else:
            assert False

    def _start_streaming(self, batch_size):
        # * Moshi Module
        def _start_streaming(module):
            if isinstance(module, StreamingModule):
                module._streaming_state = module._init_streaming_state(batch_size) 
        self.apply(_start_streaming)
    
    def _stop_streaming(self, batch_size):
        # * Moshi Module
        def _stop_streaming(module):
            if isinstance(module, StreamingModule):
                module._streaming_state = None
        self.apply(_stop_streaming)
    
        
    @contextmanager
    def streaming(self, batch_size=1):
        assert batch_size == 1
        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming(batch_size)

    @torch.inference_mode()
    def encode(self, wav_list, chunk_duration=-1):
        """
            Input:
                wav_list: List of audio waveforms, each with potentially different length, may exceed 30 seconds # B * (T,)
                overlap_seconds: Overlap in seconds, process 30 seconds at a time, keeping (30 - overlap_seconds) seconds of valid output
            Output:
                dict: Contains the following key-value pairs
                    "codes_list": List of quantization codes # B * (nq, T)
        """
        assert len(wav_list) == 1
        assert chunk_duration == -1 or chunk_duration <= self.causal_transformer_context_duration
        chunk_length = int(chunk_duration * self.sampling_rate)
        assert chunk_duration == -1 or chunk_length % self.downsample_rate == 0
        
        x = wav_list[0].reshape(1, 1, -1) # (B=1, 1, T)
        input_lengths = torch.tensor([x.shape[-1]]).to(x.device) # (B=1, )
        
        if input_lengths[0].item() <= chunk_length or chunk_duration == -1:
            result = self.inference_tokenize(x, input_lengths) # {"zq": (B, D, T'), "codes": (nq, B, T'), "codes_lengths": (B,)}
            codes = result["codes"] # (nq, B, T')
        else:
            codes = [] # N_chunk * (nq, B=1, T')
            with self.streaming():
                for start_idx in range(0, x.shape[-1], chunk_length): # start_idx = {0, chunk_length, 2 * chunk_length, ...}
                    input_lengths_i = min(chunk_length, x.shape[-1] - start_idx)
                    input_lengths_i = torch.tensor([input_lengths_i]).to(x.device) # (B=1, )
                    x_i = x[:, :, start_idx: start_idx + input_lengths_i[0].item()] # (B=1, 1, T)
                    result_i = self.inference_tokenize(x_i, input_lengths_i)
                    codes_i = result_i["codes"] # (nq, B, T')
                    codes.append(codes_i) 
            codes = torch.cat(codes, dim=-1) # (nq, B=1, T')
            
        codes_list = [codes[:, 0, :]] # B=1 * (nq, T')
        return {
            "codes_list": codes_list # B * (nq, T)
        }
        
    @torch.inference_mode()
    def decode(self, codes_list, chunk_duration=-1):
        """
            Input:
                codes_list: List of quantization codes # B * (nq, T)
                overlap_seconds: Overlap in seconds, process 30 seconds at a time, keeping (30 - overlap_seconds) seconds of valid output
            Output:
                dict: Contains the following key-value pairs
                    "syn_wav_list": List of synthesized audio waveforms # B * (T,)
        """
        assert len(codes_list) == 1
        assert chunk_duration == -1 or chunk_duration <= self.causal_transformer_context_duration
        chunk_length = int(chunk_duration * self.sampling_rate)
        chunk_frame_length = chunk_length // self.downsample_rate
        assert chunk_duration == -1 or chunk_duration * self.sampling_rate % self.downsample_rate == 0
        codes = codes_list[0].unsqueeze(1) # (nq, B=1, T)
        codes_lengths = torch.tensor([codes.shape[-1]]).to(codes.device) # (B=1, )
        
        if codes_lengths[0].item() <= chunk_frame_length or chunk_duration == -1:
            result = self.inference_detokenize(codes, codes_lengths)
            wav = result["y"] # (B=1, 1, T)
        else:
            wav = []
            with self.streaming():
                for start_idx in range(0, codes.shape[-1], chunk_frame_length): # start_idx = {0, chunk_frame_length, 2 * chunk_frame_length, ...}
                    codes_lengths_i = min(chunk_frame_length, codes.shape[-1] - start_idx)
                    codes_lengths_i = torch.tensor([codes_lengths_i]).to(codes.device) # (B=1, )
                    codes_i = codes[:, :, start_idx: start_idx + codes_lengths_i[0].item()] # (nq, B=1, T)
                    result_i = self.inference_detokenize(codes_i, codes_lengths_i)
                    wav_i = result_i["y"] # (B=1, 1, T)
                    wav.append(wav_i)
            wav = torch.cat(wav, dim=-1) # (B=1, 1, T)
        
        syn_wav_list = [wav[0].squeeze(0)] # B=1 * (T, )
        return {
            "syn_wav_list": syn_wav_list # B * (T,)
        }
    

    @classmethod
    def load_from_checkpoint(cls, config_path: str, ckpt_path: str):
        # 从配置文件和检查点加载模型
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        model = cls(cfg['generator_params'])

        # 加载检查点
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if 'generator' in ckpt.keys(): # SpeechTokenizer3 格式
            params = ckpt['generator']
        elif 'model' in ckpt.keys(): # Codec 格式
            params = ckpt['model']
        else: # Release 格式
            params = ckpt

        load_result = model.load_state_dict(params, strict=False)
        logging.info(f"成功加载 SpeechTokenizer 版本 {model.version}")
        logging.info(f"缺失的键: {load_result.missing_keys}")
        logging.info(f"意外的键: {load_result.unexpected_keys}")

        return model