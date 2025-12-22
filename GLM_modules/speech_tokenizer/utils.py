import os
import io
import glob
import math
import tarfile
import torch
import torchaudio
import safetensors
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
# sys.path.append('/inspire/hdd/project/embodied-multimodality/public/yfxu/Full-Duplex-Bench/model_inference/atlas/voicetokenizers/StreamingCodec/GLM_modules/speech_tokenizer')
from configuration_whisper import WhisperVQConfig
from generation_whisper import WhisperGenerationMixin
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


def load_quantize_encoder(model_path):
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(model, feature_extractor, utts, batch_size=128):
    device=model.codebook.weight.device
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            elif isinstance(utt,torch.Tensor):
                if utt.ndim==2:
                    audio = utt
                elif utt.ndim==1:
                    audio=utt.unsqueeze(0)
                sample_rate=16000
            else:
                audio, sample_rate = torchaudio.load(utt)
            audio = audio.to(device)
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to(device)
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = batch_size
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device=device,
                                         padding="longest", pad_to_multiple_of=stride)
            features = features.to(device=device)

            outputs = model.forward(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
        return all_speech_tokens


def extract_speech_token_test(model, feature_extractor, utts,batch_size=128):
    device=model.codebook.weight.device
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = torchaudio.load(utt)
            audio = audio.to(device)
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to(device)
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = batch_size

        for start in range(0, len(audios), batch_size):
            audios=audios[0]
            input_features_all,max_log_spec = feature_extractor(audios, sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device=device,
                                         padding="longest", pad_to_multiple_of=stride,return_max_log_spec=True)
            input_features_all=input_features_all.to(device=device)['input_features']
            audios=torch.tensor(audios)
            audios=audios.unsqueeze(0).unsqueeze(0)
            n=4
            audio_parts = split_features_with_1280_multiple_adjusted_to_n_parts(audios,num=n)
            
            input_features_parts = []
            speech_tokens_parts = []
            for i in range(n):
                current_audio_part = audio_parts[i]
                if current_audio_part.shape[-1] == 0:
                    input_features_parts.append(None) 
                    speech_tokens_parts.append([])
                    continue

                squeezed_audio_part = current_audio_part.squeeze(0).squeeze(0)
                current_input_features = feature_extractor(squeezed_audio_part, sampling_rate=16000,
                                                 return_attention_mask=True, return_tensors="pt", device=device,
                                                 padding="longest", pad_to_multiple_of=stride).to(device=device)['input_features']
                input_features_parts.append(current_input_features)
            past_cache = None
            conv1_cache = None
            conv2_cache = None

            for i in range(n):
                current_input_features = input_features_parts[i]
                
                if current_input_features is None: 
                    speech_tokens_parts.append([])
                    continue

                if i == 0:
                    outputs = model.forward_causal(current_input_features)
                else:
                    outputs = model.forward_causal(current_input_features, past_key_values=past_cache, 
                                                   conv1_cache=conv1_cache, conv2_cache=conv2_cache)
                
                speech_tokens_parts.append(outputs.quantized_token_ids.cpu().tolist()[0])
                past_cache = outputs.past_key_value
                conv1_cache = outputs.conv1_cache
                conv2_cache = outputs.conv2_cache
            
            return [sum(speech_tokens_parts, [])]
        return all_speech_tokens

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# 假设 WhisperFeatureExtractor 和模型 (model) 已经定义并加载
# _resample_buffer 是一个全局字典，用于缓存重采样器
_resample_buffer = {}


def extract_speech_token_streaming(model, feature_extractor, utts, chunk_len_ms=80):
    """
    使用真正的流式方法提取语音token，以固定的chunk大小逐步处理音频。
    
    Args:
        model: 你的WhisperVQEncoder模型，需要有 forward_causal 方法。
        feature_extractor: Whisper的特征提取器。
        utts (list): 音频文件路径列表或 (waveform, sample_rate) 元组列表。
        chunk_len_ms (int): 每个流式块的音频长度（毫秒）。1280帧对应80ms (1280 / 16000 * 1000)。
    
    Returns:
        list: 一个列表，每个元素是对应输入音频的token序列。
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_speech_tokens = []
    
    with torch.no_grad():
        for utt in tqdm(utts, desc="Processing Utterances"):
            # --- 1. 音频加载和预处理 ---
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = torchaudio.load(utt)
            
            audio = audio.to(device)
            # 重采样到16kHz
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=16000
                    ).to(device)
                audio = _resample_buffer[sample_rate](audio)
            
            # 确保是单声道
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # --- 2. 初始化流式处理参数 ---
            chunk_size_samples = int(16000 * (chunk_len_ms / 1000.0))  # 例如 1280
            total_samples = audio.shape[1]
            
            # 获取模型对输入长度的步幅要求
            # stride = model.conv1.stride[0] * model.conv2.stride[0] * (model.config.pooling_kernel_size or 1) * feature_extractor.hop_length
            # 假设这个值是固定的，比如 320。如果不是，需要从模型动态获取。
            # 为了代码健壮性，我们假设有一个方法或属性可以获取它。
            # 这是一个典型值，但你需要根据你的模型确认。
            model_stride_req = 320 

            # 初始化缓存
            past_cache = None
            conv1_cache = None
            conv2_cache = None
            
            current_utt_tokens = []
            
            # --- 3. 流式处理循环 ---
            # 使用 start_pos 作为指针，在整个音频上滑动
            for start_pos in range(0, total_samples, chunk_size_samples):
                end_pos = start_pos + chunk_size_samples
                
                # 获取当前音频块
                # 注意：这里我们不需要担心最后一块的长度，特征提取器会处理padding
                current_audio_chunk = audio[:, start_pos:end_pos]
                
                # 如果块为空，则跳过
                if current_audio_chunk.shape[1] == 0:
                    continue

                # --- 4. 特征提取和模型推理 ---
                # 为当前块提取特征
                input_features = feature_extractor(
                    current_audio_chunk.squeeze(0).cpu().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="longest", # 即使是流式，padding到模型要求的倍数也是安全的
                    pad_to_multiple_of=model_stride_req 
                ).input_features.to(device)
                
                # 流式前向传播
                if start_pos == 0: # 第一个块
                    outputs = model.forward_causal(input_features)
                else: # 后续块，传入缓存
                    outputs = model.forward_causal(
                        input_features, 
                        past_key_values=past_cache, 
                        conv1_cache=conv1_cache, 
                        conv2_cache=conv2_cache
                    )
                
                # --- 5. 收集结果并更新缓存 ---
                # 收集生成的token
                current_utt_tokens.extend(outputs.quantized_token_ids.cpu().tolist()[0])
                # 更新缓存以供下一个块使用
                past_cache = outputs.past_key_value
                conv1_cache = outputs.conv1_cache
                conv2_cache = outputs.conv2_cache

            all_speech_tokens.append(current_utt_tokens)
            
    return all_speech_tokens



import torch

def split_features_with_16_multiple_adjusted(input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits input_features into 4 parts along the last dimension,
    based on an initial average split length (input_features.shape[-1] // 4).
    Ensures that the first three parts (input_features_1, 2, 3) have
    lengths that are multiples of 16. The fourth part takes all remaining length.

    Args:
        input_features (torch.Tensor): The input tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the four split tensors.
    """
    total_dim = input_features.shape[-1]
    
    # 原始的平均切分长度
    base_split_len_per_part = total_dim // 4
    
    multiple = 8

    # 计算前三个部分的长度，使其是16的倍数
    # 我们将 base_split_len_per_part 向下取整到最近的16的倍数。
    # 如果向下取整后为0，但我们有足够的空间（>=16），则至少分配16。
    
    # len1
    len1 = (base_split_len_per_part // multiple) * multiple
    if len1 == 0 and total_dim >= multiple: # 如果初始计算为0，但总长度允许，则至少给16
        len1 = multiple
    len1 = min(len1, total_dim) # 确保不超过总长度

    # len2
    # 对于len2，我们从剩余的维度中考虑，但仍然基于原始的 base_split_len_per_part
    remaining_after_len1 = total_dim - len1
    len2 = (base_split_len_per_part // multiple) * multiple
    if len2 == 0 and remaining_after_len1 >= multiple:
        len2 = multiple
    len2 = min(len2, remaining_after_len1) # 确保不超过剩余长度

    # len3
    remaining_after_len1_len2 = total_dim - len1 - len2
    len3 = (base_split_len_per_part // multiple) * multiple
    if len3 == 0 and remaining_after_len1_len2 >= multiple:
        len3 = multiple
    len3 = min(len3, remaining_after_len1_len2) # 确保不超过剩余长度

    # len4 吸收所有剩余的维度
    len4 = total_dim - len1 - len2 - len3

    # 确保所有长度都是非负数
    len1 = max(0, len1)
    len2 = max(0, len2)
    len3 = max(0, len3)
    len4 = max(0, len4) # len4可能会因为前三部分分配过多而为负，这里是兜底

    split_sizes = [len1, len2, len3, len4]

    # 验证和调试信息
    print(f"Total dimension: {total_dim}")
    print(f"Base split length per part (//4): {base_split_len_per_part}")
    print(f"Calculated split sizes: {split_sizes}")
    print(f"input_features_1 length ({len1}) is multiple of 16: {len1 % multiple == 0}")
    print(f"input_features_2 length ({len2}) is multiple of 16: {len2 % multiple == 0}")
    print(f"input_features_3 length ({len3}) is multiple of 16: {len3 % multiple == 0}")
    print(f"Sum of split sizes: {sum(split_sizes)}")
    if sum(split_sizes) != total_dim:
        print(f"Warning: Sum of split sizes ({sum(split_sizes)}) does not match total dimension ({total_dim}). Adjusting last part.")
        # 如果总和不匹配，将差异加到len4上（理论上不应该发生如果前面的min/max处理正确）
        len4 += (total_dim - sum(split_sizes))
        split_sizes = [len1, len2, len3, max(0, len4)] # 再次确保len4非负

    input_features_1, input_features_2, input_features_3, input_features_4 = torch.split(input_features, split_sizes, dim=-1)
    return input_features_1, input_features_2, input_features_3, input_features_4

def split_features_with_1280_multiple_adjusted(input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits input_features into 4 parts along the last dimension,
    based on an initial average split length (input_features.shape[-1] // 4).
    Ensures that the first three parts (input_features_1, 2, 3) have
    lengths that are multiples of 16. The fourth part takes all remaining length.

    Args:
        input_features (torch.Tensor): The input tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing the four split tensors.
    """
    total_dim = input_features.shape[-1]
    
    # 原始的平均切分长度
    base_split_len_per_part = total_dim // 4
    
    multiple = 1280

    # 计算前三个部分的长度，使其是16的倍数
    # 我们将 base_split_len_per_part 向下取整到最近的16的倍数。
    # 如果向下取整后为0，但我们有足够的空间（>=16），则至少分配16。
    
    # len1
    len1 = (base_split_len_per_part // multiple) * multiple
    if len1 == 0 and total_dim >= multiple: # 如果初始计算为0，但总长度允许，则至少给16
        len1 = multiple
    len1 = min(len1, total_dim) # 确保不超过总长度

    # len2
    # 对于len2，我们从剩余的维度中考虑，但仍然基于原始的 base_split_len_per_part
    remaining_after_len1 = total_dim - len1
    len2 = (base_split_len_per_part // multiple) * multiple
    if len2 == 0 and remaining_after_len1 >= multiple:
        len2 = multiple
    len2 = min(len2, remaining_after_len1) # 确保不超过剩余长度

    # len3
    remaining_after_len1_len2 = total_dim - len1 - len2
    len3 = (base_split_len_per_part // multiple) * multiple
    if len3 == 0 and remaining_after_len1_len2 >= multiple:
        len3 = multiple
    len3 = min(len3, remaining_after_len1_len2) # 确保不超过剩余长度

    # len4 吸收所有剩余的维度
    len4 = total_dim - len1 - len2 - len3

    # 确保所有长度都是非负数
    len1 = max(0, len1)
    len2 = max(0, len2)
    len3 = max(0, len3)
    len4 = max(0, len4) # len4可能会因为前三部分分配过多而为负，这里是兜底

    split_sizes = [len1, len2, len3, len4]

    # 验证和调试信息
    print(f"Total dimension: {total_dim}")
    print(f"Base split length per part (//4): {base_split_len_per_part}")
    print(f"Calculated split sizes: {split_sizes}")
    print(f"input_features_1 length ({len1}) is multiple of 16: {len1 % multiple == 0}")
    print(f"input_features_2 length ({len2}) is multiple of 16: {len2 % multiple == 0}")
    print(f"input_features_3 length ({len3}) is multiple of 16: {len3 % multiple == 0}")
    print(f"Sum of split sizes: {sum(split_sizes)}")
    if sum(split_sizes) != total_dim:
        print(f"Warning: Sum of split sizes ({sum(split_sizes)}) does not match total dimension ({total_dim}). Adjusting last part.")
        # 如果总和不匹配，将差异加到len4上（理论上不应该发生如果前面的min/max处理正确）
        len4 += (total_dim - sum(split_sizes))
        split_sizes = [len1, len2, len3, max(0, len4)] # 再次确保len4非负

    input_features_1, input_features_2, input_features_3, input_features_4 = torch.split(input_features, split_sizes, dim=-1)
    return input_features_1, input_features_2, input_features_3, input_features_4

def split_features_with_1280_multiple_adjusted_to_n_parts(input_features: torch.Tensor, num=16) -> tuple[torch.Tensor, ...]:
    """
    Splits input_features into 16 parts along the last dimension,
    based on an initial average split length (input_features.shape[-1] // 16).
    Ensures that the first fifteen parts have lengths that are multiples of 1280.
    The sixteenth part takes all remaining length.

    Args:
        input_features (torch.Tensor): The input tensor.

    Returns:
        tuple[torch.Tensor, ...]: A tuple containing the sixteen split tensors.
    """
    total_dim = input_features.shape[-1]
    
    # 原始的平均切分长度，现在是分成16份
    base_split_len_per_part = total_dim // num
    
    multiple = 1280
    
    split_sizes = []
    current_offset = 0

    # 计算前15个部分的长度，使其是1280的倍数
    for i in range(num-1):
        remaining_dim_for_part = total_dim - current_offset
        
        # 目标长度基于平均分配
        target_len = (base_split_len_per_part // multiple) * multiple
        
        # 如果向下取整后为0，但我们有足够的空间（>= multiple），则至少分配 multiple
        if target_len == 0 and remaining_dim_for_part >= multiple:
            target_len = multiple
        
        # 确保计算出的长度不超过当前剩余的总长度
        # 并且，如果剩余空间不足以分配一个multiple，则分配0
        len_i = min(target_len, remaining_dim_for_part)
        
        # 最终确保它还是multiple的倍数，除非它被min限制到了小于multiple
        # 在这种情况下，我们可能需要重新评估，或者接受它不是multiple的倍数
        # 但这里我们希望前15个是multiple的倍数，所以如果len_i不是multiple的倍数，并且足够长，
        # 那么问题出在target_len的计算上。
        # 最稳妥的方式是：在remaining_dim_for_part中，能取多少个multiple，就取多少。
        
        len_i = (remaining_dim_for_part // multiple) * multiple
        len_i = min(len_i, target_len) # 再次限制，避免分配过多，偏离平均值太多
        
        # 兜底：如果计算出来还是0，并且还有足够的空间，就分配一个multiple，但要确保不超剩余
        if len_i == 0 and remaining_dim_for_part >= multiple:
            len_i = multiple
        
        # 再次检查，确保不超过剩余维度
        len_i = min(len_i, remaining_dim_for_part)
        
        split_sizes.append(len_i)
        current_offset += len_i

    # 第16个部分吸收所有剩余的维度
    len16 = total_dim - current_offset
    split_sizes.append(len16)

    # 确保所有长度都是非负数
    split_sizes = [max(0, l) for l in split_sizes]

    # 验证和调试信息
    print(f"Total dimension: {total_dim}")
    print(f"Base split length per part (//16): {base_split_len_per_part}")
    print(f"Calculated split sizes: {split_sizes}")
    
    for i, length in enumerate(split_sizes[:-1]): # 前15个
        print(f"input_features_{i+1} length ({length}) is multiple of {multiple}: {length % multiple == 0}")
    
    print(f"Sum of split sizes: {sum(split_sizes)}")
    if sum(split_sizes) != total_dim:
        print(f"Warning: Sum of split sizes ({sum(split_sizes)}) does not match total dimension ({total_dim}). Adjusting last part.")
        # 如果总和不匹配，将差异加到最后一个部分上
        split_sizes[-1] += (total_dim - sum(split_sizes))
        split_sizes[-1] = max(0, split_sizes[-1]) # 再次确保非负

    # 如果有任何 split_size 是 0，并且输入特征不允许 0 长度切分，这可能会导致错误。
    # PyTorch的torch.split通常允许0长度的切分。
    
    # 最终切分
    split_tensors = torch.split(input_features, split_sizes, dim=-1)
    return split_tensors

 # input_features_1,max_log_spec = feature_extractor(audio1, sampling_rate=16000,
            #                              return_attention_mask=True, return_tensors="pt", device=device,
            #                              padding="longest", pad_to_multiple_of=stride,return_max_log_spec=True)
            # input_features_1=input_features_1.to(device=device)['input_features']
            # max_log_spec=max_log_spec.to(device=device)
            # input_features_1=torch.cat([max_log_spec.unsqueeze(0).expand(1,128).unsqueeze(-1),input_features_1[:,:,:-1]],dim=-1)
            
            
            # input_features_2 = feature_extractor(torch.cat([audio1,audio2]), sampling_rate=16000,
            #                              return_attention_mask=True, return_tensors="pt", device=device,
            #                              padding="longest", pad_to_multiple_of=stride,max_log_spec=torch.tensor(max_log_spec.item())).to(device=device)['input_features'][:,:,input_features_1.shape[-1]-1:-1]
            
            # input_features_3 = feature_extractor(torch.cat([audio1,audio2,audio3]), sampling_rate=16000,
            #                              return_attention_mask=True, return_tensors="pt", device=device,
            #                              padding="longest", pad_to_multiple_of=stride,max_log_spec=torch.tensor(max_log_spec.item())).to(device=device)['input_features'][:,:,(input_features_1.shape[-1]+input_features_2.shape[-1]-1):-1]
            
            
            # input_features_4 = feature_extractor(torch.cat([audio1,audio2,audio3,audio4]), sampling_rate=16000,
            #                              return_attention_mask=True, return_tensors="pt", device=device,
            #                              padding="longest", pad_to_multiple_of=stride,max_log_spec=torch.tensor(max_log_spec.item())).to(device=device)['input_features'][:,:,(input_features_1.shape[-1]+input_features_2.shape[-1]+input_features_2.shape[-1]-1):-1]