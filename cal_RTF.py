import os
import sys
import argparse
import time
import glob
from collections import OrderedDict

import torch
import torchaudio
from torch import nn
from transformers import WhisperFeatureExtractor
from tqdm import tqdm
import numpy as np

# ================== 自己的路径 ==================
# GLM4 / WhisperVQ 相关
sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/GLM_modules')
from speech_tokenizer.modeling_whisper import WhisperVQEncoder, WhisperVQConfig

# DAC / Mimi 相关
sys.path.append("/inspire/hdd/project/embodied-multimodality/public/ytgong/modelings_for_tts")
from modeling_dac import DACModel
from modeling_mimi import Mimi

# ================== 常量：DAC / Mimi config / model ==================
DAC_CONFIG_PATH = "/inspire/hdd/project/embodied-multimodality/public/ytgong/modelings_for_tts/dac_24khz_rvq8.yaml"
DAC_MODEL_PATH = "/inspire/hdd/project/embodied-multimodality/public/ytgong/modelings_for_tts/dac24khz.pth"

MIMI_CONFIG_PATH = "/inspire/hdd/project/embodied-multimodality/public/ytgong/modelings_for_tts/mimi32.yaml"
MIMI_MODEL_PATH = "/inspire/hdd/project/embodied-multimodality/public/ytgong/modelings_for_tts/mimi.safetensors"

# 全局重采样缓存（给 WhisperVQ 用）
_resample_buffer = {}


# ======================================================================
#  工具函数：加载 + 重采样（给 GLM4 / WhisperVQ 使用）
# ======================================================================
def _load_and_resample_whisper(utt, target_sr=16000, device="cpu"):
    """
    支持:
      - 文件路径
      - (waveform, sr) tuple
    返回: (mono_waveform [1, T], target_sr)
    """
    if isinstance(utt, tuple):
        audio, sample_rate = utt
    else:
        audio, sample_rate = torchaudio.load(utt)

    audio = audio.to(device)

    # 重采样
    if sample_rate != target_sr:
        if sample_rate not in _resample_buffer:
            _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            ).to(device)
        if hasattr(_resample_buffer[sample_rate], "kernel"):
            _resample_buffer[sample_rate].kernel = _resample_buffer[sample_rate].kernel.to(device)
        audio = _resample_buffer[sample_rate](audio)

    # 单声道
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    return audio, target_sr


# ======================================================================
#  非流式 WhisperVQ：整段 mel + 整段 forward
# ======================================================================
def extract_speech_token_non_streaming(model, feature_extractor, utts):
    """
    非流式：整段音频一次性过 mel，再一次 forward。
    Args:
        model: WhisperVQEncoder
        feature_extractor: WhisperFeatureExtractor
        utts: list[str] 或 list[(waveform, sr)]
    Returns:
        List[List[int]]: 每个 utt 的 token 序列
    """
    device = next(model.parameters()).device
    model.eval()

    all_speech_tokens = []

    with torch.no_grad():
        for utt in tqdm(utts, desc="Processing Utterances (Non-streaming)"):
            # 1. 加载 + 重采样 + 单声道
            audio, sample_rate = _load_and_resample_whisper(utt, target_sr=16000, device=device)

            # 2. 特征提取
            audio_np = audio.squeeze(0).cpu().numpy()

            inputs = feature_extractor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest",
            )

            input_features = inputs.input_features.to(device)  # [B=1, n_mels, T_mel]

            # 3. attention_mask
            bsz, n_mels, seq_len = input_features.shape
            attention_mask = torch.ones(
                (bsz, seq_len),
                dtype=torch.long,
                device=device,
            )

            # 4. 前向
            outputs = model(
                input_features,
                attention_mask=attention_mask,
            )

            token_ids = outputs.quantized_token_ids
            if token_ids is None:
                raise RuntimeError(
                    "quantized_token_ids is None; 请检查 WhisperVQConfig 是否开启量化配置。"
                )

            all_speech_tokens.append(token_ids[0].detach().cpu().tolist())

    return all_speech_tokens


# ======================================================================
#  流式 WhisperVQ：chunk + forward_causal
# ======================================================================
def extract_speech_token_streaming(model, feature_extractor, utts, chunk_len_ms=80):
    """
    流式：按 chunk 逐步处理音频，通过 forward_causal 维护 cache。
    """
    device = next(model.parameters()).device
    model.eval()

    all_speech_tokens = []

    with torch.no_grad():
        for utt in tqdm(utts, desc="Processing Utterances (Streaming)"):
            # 1. 加载 + 重采样 + 单声道
            audio, _ = _load_and_resample_whisper(utt, target_sr=16000, device=device)

            # 2. 流式参数
            chunk_size_samples = int(16000 * (chunk_len_ms / 1000.0))  # 比如 1280
            total_samples = audio.shape[1]

            model_stride_req = 320  # 如需更准，可以从模型中取

            # cache
            past_cache = None
            conv1_cache = None
            conv2_cache = None

            current_utt_tokens = []

            # 3. 流式循环
            for start_pos in range(0, total_samples, chunk_size_samples):
                end_pos = start_pos + chunk_size_samples
                current_audio_chunk = audio[:, start_pos:end_pos]

                if current_audio_chunk.shape[1] == 0:
                    continue

                # 4. 特征提取
                input_features = feature_extractor(
                    current_audio_chunk.squeeze(0).cpu().numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="longest",
                    pad_to_multiple_of=model_stride_req,
                ).input_features.to(device)

                # 5. 流式前向
                if start_pos == 0:
                    outputs = model.forward_causal(input_features)
                else:
                    outputs = model.forward_causal(
                        input_features,
                        past_key_values=past_cache,
                        conv1_cache=conv1_cache,
                        conv2_cache=conv2_cache,
                    )

                # 6. 收集 token + 更新缓存
                current_utt_tokens.extend(outputs.quantized_token_ids.cpu().tolist()[0])
                past_cache = outputs.past_key_value
                conv1_cache = outputs.conv1_cache
                conv2_cache = outputs.conv2_cache

            all_speech_tokens.append(current_utt_tokens)

    return all_speech_tokens


# ======================================================================
#  GLM4Encoder 封装（WhisperVQ）
# ======================================================================
class GLM4Encoder(nn.Module):
    def __init__(self, tokenizer_path, config_path, feature_extractor_path, streaming=False):
        super().__init__()
        self.sample_rate = 16000
        self.streaming = streaming  # True=流式, False=整段非流式

        config = WhisperVQConfig.from_pretrained(config_path)
        self.whisper_vqmodel = WhisperVQEncoder(config)

        # 加载权重
        try:
            ckpt = torch.load(tokenizer_path, map_location='cpu')
            state_dict = ckpt.get('generator', ckpt)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("encoder."):
                    new_k = k[len("encoder."):]
                    new_state_dict[new_k] = v
            missing_keys, unexpected_keys = self.whisper_vqmodel.load_state_dict(
                new_state_dict, strict=False
            )
            print("State dict loading report:")
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found at {tokenizer_path}. Model weights are random.")
        except Exception as e:
            print(f"An error occurred while loading the state dict: {e}")

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(feature_extractor_path)

    @torch.no_grad()
    def encode_token(self, audio_path: str):
        """
        单个音频 -> token 序列
        """
        if self.streaming:
            audio_tokens = extract_speech_token_streaming(
                self.whisper_vqmodel,
                self.feature_extractor,
                [audio_path],
            )[0]
        else:
            audio_tokens = extract_speech_token_non_streaming(
                self.whisper_vqmodel,
                self.feature_extractor,
                [audio_path],
            )[0]
        return audio_tokens


# ======================================================================
#  DAC Encoder 封装
# ======================================================================
class DACEncoder(nn.Module):
    """
    封一层，让接口和 GLM4Encoder 一样：有 encode_token(audio_path)。
    """
    def __init__(self):
        super().__init__()
        self.speech_tokenizer = DACModel.load_from_checkpoint(DAC_CONFIG_PATH, DAC_MODEL_PATH)
        self.sample_rate = self.speech_tokenizer.sample_rate  # DAC 的采样率

    @torch.no_grad()
    def encode_token(self, audio_path: str):
        """
        单个 audio_path -> DAC codes (nq, T_code)
        - 先采样成 speech_tokenizer.sample_rate
        - wav_list 是 [B, T]，这里 B=1
        """
        device = next(self.speech_tokenizer.parameters()).device

        wav, sr = torchaudio.load(audio_path)  # [C, T]
        wav = wav.to(device)

        # 单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # [1, T]

        # 重采样到 DAC 的 sample_rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            ).to(device)
            wav = resampler(wav)  # [1, T']

        # 去掉 channel，构造 wav_list: [B, T]
        wav_1d = wav.squeeze(0)           # [T]
        wav_list = wav_1d.unsqueeze(0)    # [1, T]

        # 编码
        result = self.speech_tokenizer.encode(wav_list, device=device)
        codes_list = result["codes_list"]     # B * (nq, T)
        codes = codes_list[0]                # (nq, T_code) tensor

        return codes


# ======================================================================
#  Mimi Encoder 封装
# ======================================================================
class MimiEncoder(nn.Module):
    """
    Mimi 分支，接口同样是 encode_token(audio_path)
    """
    def __init__(self):
        super().__init__()
        self.speech_tokenizer = Mimi.load_from_checkpoint(MIMI_CONFIG_PATH, MIMI_MODEL_PATH)
        self.sample_rate = self.speech_tokenizer.sample_rate  # Mimi 的采样率

    @torch.no_grad()
    def encode_token(self, audio_path: str):
        """
        单个 audio_path -> Mimi codes
        假设 Mimi.encode 接口与 DAC 一致：encode(wav_list, device=..) -> {"codes_list": [...]}
        """
        device = next(self.speech_tokenizer.parameters()).device

        wav, sr = torchaudio.load(audio_path)  # [C, T]
        wav = wav.to(device)

        # 单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 重采样到 Mimi 的 sample_rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate
            ).to(device)
            wav = resampler(wav)

        wav_1d = wav.squeeze(0)        # [T]
        wav_list = wav_1d.unsqueeze(0) # [1, T]

        result = self.speech_tokenizer.encode(wav_list, device=device)
        codes_list = result["codes_list"]  # B * (nq, T)
        codes = codes_list[0]

        return codes


# ======================================================================
#  RTF 计算：调用 model.encode_token(audio_path)
# ======================================================================
def calculate_rtf(model, audio_files, device, num_warmup=10):
    """
    Calculates the Real-Time Factor (RTF) for the given model and audio files.
    model 必须实现 encode_token(audio_path)。
    """
    if not audio_files:
        print("No audio files found. Exiting.")
        return

    model.to(device)
    model.eval()

    print(f"Warming up the model for {num_warmup} iterations on {device.type}...")
    warmup_files = audio_files[:num_warmup]
    if len(warmup_files) < num_warmup:
        print(f"Warning: Not enough unique files for warmup. Using {len(warmup_files)} files.")

    for audio_path in tqdm(warmup_files, desc="Warm-up"):
        try:
            _ = model.encode_token(audio_path)
        except Exception as e:
            print(f"Warning: Error during warmup with file {audio_path}: {e}")
            continue

    total_processing_time = 0.0
    total_audio_duration = 0.0

    print(f"Starting RTF calculation for {len(audio_files)} files...")

    for audio_path in tqdm(audio_files, desc="Calculating RTF"):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_duration = waveform.shape[1] / sample_rate
            total_audio_duration += audio_duration

            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.perf_counter()

            _ = model.encode_token(audio_path)

            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                processing_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                processing_time = time.perf_counter() - start_time

            total_processing_time += processing_time

        except Exception as e:
            print(f"Skipping file {audio_path} due to an error: {e}")
            continue

    if total_audio_duration == 0:
        print("Could not process any audio files. Total audio duration is zero.")
        return

    average_rtf = total_processing_time / total_audio_duration

    print("\n" + "=" * 50)
    print("RTF Calculation Summary")
    print(f"  - Device: {device}")
    print(f"  - Total audio files processed: {len(audio_files)}")
    print(f"  - Total audio duration: {total_audio_duration:.2f} seconds")
    print(f"  - Total processing time: {total_processing_time:.2f} seconds")
    print(f"  - Average RTF: {average_rtf:.6f}")
    print("=" * 50)

    return average_rtf


# ======================================================================
#  main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Calculate Real-Time Factor (RTF) for GLM4Encoder / DAC / Mimi.")
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100',
        help='Directory of the dataset.'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt',
        help='Path to the WhisperVQ tokenizer model checkpoint (for type=glm4).'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/config.json',
        help='Path to the WhisperVQConfig json file (for type=glm4).'
    )
    parser.add_argument(
        '--feature_extractor_path',
        type=str,
        default='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/glm-4-voice-tokenizer',
        help='Path to the WhisperFeatureExtractor (for type=glm4).'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run the benchmark on.'
    )
    parser.add_argument(
        '--limit_files',
        type=int,
        default=128,
        help='Limit the number of files to process for a quick test.'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='(Only for type=glm4) Use streaming mode (chunked forward_causal).'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='mimi',
        choices=['glm4', 'dac', 'mimi'],
        help='Tokenizer type: "glm4" for WhisperVQEncoder, "dac" for DACModel, "mimi" for Mimi.'
    )
    args = parser.parse_args()

    # 1. 找 wav
    print(f"Searching for .wav files in {args.dataset_dir}...")
    audio_files = glob.glob(os.path.join(args.dataset_dir, '**', '*.wav'), recursive=True)

    if args.limit_files:
        audio_files = audio_files[:args.limit_files]

    if not audio_files:
        print(f"Error: No .wav files found in {args.dataset_dir}. Please check the path.")
        return

    print(f"Found {len(audio_files)} .wav files.")

    device = torch.device(args.device)

    # 2. 初始化对应的 tokenizer / encoder
    if args.type == 'glm4':
        print(f"Initializing GLM4Encoder (streaming={args.streaming})...")
        model = GLM4Encoder(
            tokenizer_path=args.tokenizer_path,
            config_path=args.config_path,
            feature_extractor_path=args.feature_extractor_path,
            streaming=args.streaming,
        )
    elif args.type == 'dac':
        print("Initializing DACEncoder...")
        model = DACEncoder()
    else:
        print("Initializing MimiEncoder...")
        model = MimiEncoder()

    # 3. 计算 RTF
    calculate_rtf(model, audio_files, device=device)


if __name__ == '__main__':
    main()
