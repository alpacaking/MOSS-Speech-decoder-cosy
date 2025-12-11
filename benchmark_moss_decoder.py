import os
import json
import torch
import torchaudio
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec')
from whisper_encoder_decoder import GLM4Encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MOSS Speech Decoder on Seed-TTS benchmark")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Benchmark dataset directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--lang", type=str, choices=['en', 'zh'], required=True, help="Language to evaluate")
    parser.add_argument("--block_size", type=int, default=5, help="Block size for streaming inference")
    parser.add_argument("--mel_cache_len", type=int, default=8, help="Block size for streaming inference")
    parser.add_argument("--max_token_len", type=int, default=40, help="Block size for streaming inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def load_model(tokenizer_path, mel_cache_len,device):
    """加载 GLM4Encoder 模型"""
    print("Loading GLM4Encoder model...")
    encoder = GLM4Encoder(tokenizer_path=tokenizer_path,mel_cache_len=mel_cache_len)
    encoder = encoder.to(device)
    encoder.eval()
    print("Model loaded successfully")
    return encoder


def get_benchmark_data(benchmark_dir, lang):
    """
    获取评测数据
    
    Returns:
        List of tuples: (sample_name, prompt_wav_path, label_wav_path, prompt_text, label_text)
    """
    data = []
    lang_dir = Path(benchmark_dir) / lang
    
    for sample_dir in lang_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        
        sample_name = sample_dir.name
        prompt_wav = sample_dir / "prompt.wav"
        label_wav = sample_dir / "label.wav"
        prompt_txt = sample_dir / "prompt.txt"
        label_txt = sample_dir / "label.txt"
        
        # 检查必需文件是否存在
        if not all([prompt_wav.exists(), label_wav.exists(), prompt_txt.exists(), label_txt.exists()]):
            print(f"Warning: Missing files in {sample_name}, skipping...")
            continue
        
        # 读取文本
        with open(prompt_txt, 'r', encoding='utf-8') as f:
            prompt_text = f.readline().strip()
        with open(label_txt, 'r', encoding='utf-8') as f:
            label_text = f.readline().strip()
        
        data.append((sample_name, str(prompt_wav), str(label_wav), prompt_text, label_text))
    
    return data


def process_single_sample(
    encoder,
    sample_name,
    prompt_wav_path,
    label_wav_path,
    prompt_text,
    label_text,
    save_dir,
    block_size,
    max_token_len,
    lang,
    device
):
    """
    处理单个样本
    
    Args:
        encoder: GLM4Encoder 实例
        sample_name: 样本名称
        prompt_wav_path: prompt 音频路径
        label_wav_path: label 音频路径 (用于编码)
        prompt_text: prompt 文本
        label_text: label 文本
        save_dir: 保存目录
        block_size: 流式处理块大小
        device: 计算设备
    """
    sample_output_dir = Path(save_dir) / sample_name
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: 对 label audio 进行编码得到 token
        print(f"  [1/3] Encoding label audio to tokens...")
        label_tokens = encoder.encode_token(label_wav_path)
        print(f"        Generated {len(label_tokens)} tokens")
        
        # Step 2: 使用 prompt audio 和 label tokens 进行解码
        print(f"  [2/3] Decoding with prompt audio and label tokens (decode_streaming)...")
        result = encoder.decode_streaming(
            codes_list=[label_tokens],
            prompt_speech=prompt_wav_path,
            use_spk_embedding=True,
            use_prompt_speech=True,
            block_size=block_size,
            max_token_len=max_token_len
        )
        
        generated_wav = result['syn_wav_list'][0]  # (T,)
        
        # Step 3: 生成 prompt_concat_pred.wav (拼接 prompt 和 generated)
        print(f"  [3/3] Saving results...")
        prompt_wav, prompt_sr = torchaudio.load(prompt_wav_path)
        if prompt_sr != 24000:
            prompt_wav = torchaudio.functional.resample(prompt_wav, prompt_sr, 24000)
        
        # 拼接 prompt 和 generated
        if prompt_wav.dim() == 2:
            prompt_wav = prompt_wav[0]  # 取单声道
        prompt_concat_pred_wav = torch.cat([prompt_wav, generated_wav], dim=-1)
        
        # 保存 prompt_concat_pred.wav
        torchaudio.save(
            str(sample_output_dir / "prompt_concat_pred.wav"),
            prompt_concat_pred_wav.unsqueeze(0).cpu(),
            sample_rate=24000
        )
        
        # 保存 pred.wav (仅生成部分)
        torchaudio.save(
            str(sample_output_dir / "pred.wav"),
            generated_wav.unsqueeze(0).cpu(),
            sample_rate=24000
        )
        
        # 保存 metadata.json
        if lang == 'zh':
            concat_text = prompt_text + label_text
        elif lang == 'en':  # en
            concat_text = prompt_text + " " + label_text
        
        metadata = {
            "prompt_concat_pred_text": concat_text,
            "prompt_audio_path": prompt_wav_path
        }
        
        with open(sample_output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        
        print(f"  ✓ Successfully processed {sample_name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {sample_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    args = parse_args()
    
    # 设置设备
    rank = int(os.environ.get("RANK", "0"))
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank % world_size)
    device = torch.device(f"cuda:{rank % world_size}")
    
    # 加载模型
    encoder = load_model(args.tokenizer_path, args.mel_cache_len, device)
    
    # 获取评测数据
    print(f"\nLoading benchmark data for language: {args.lang}")
    benchmark_data = get_benchmark_data(args.benchmark_dir, args.lang)
    
    # 多卡分配数据
    benchmark_data = benchmark_data[rank::world_size]
    print(f"Rank {rank}/{world_size}: Processing {len(benchmark_data)} samples")
    
    if len(benchmark_data) == 0:
        print("No data to process. Exiting.")
        exit(0)
    
    # 创建输出目录
    save_dir = Path(args.save_dir) / args.lang
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个样本
    success_count = 0
    failed_count = 0
    
    print(f"\nStarting inference for {len(benchmark_data)} samples...")
    for sample_name, prompt_wav, label_wav, prompt_text, label_text in tqdm(
        benchmark_data, 
        desc=f"Rank {rank} processing"
    ):
        # 检查是否已处理
        if (save_dir / sample_name / "pred.wav").exists():
            print(f"Skipping {sample_name} (already processed)")
            success_count += 1
            continue
        
        print(f"\nProcessing: {sample_name}")
        success = process_single_sample(
            encoder=encoder,
            sample_name=sample_name,
            prompt_wav_path=prompt_wav,
            label_wav_path=label_wav,
            prompt_text=prompt_text,
            label_text=label_text,
            save_dir=save_dir,
            block_size=args.block_size,
            max_token_len=args.max_token_len,
            lang=args.lang,
            device=device
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # 打印统计信息
    print("\n" + "="*60)
    print(f"RANK {rank} SUMMARY")
    print("="*60)
    print(f"Total samples:   {len(benchmark_data)}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {failed_count}")
    print(f"Success rate:    {success_count/len(benchmark_data)*100:.2f}%")
    print("="*60)