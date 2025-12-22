import gradio as gr
import torch
import torchaudio
import sys
import os
import tempfile
import numpy as np
import argparse  # 新增
from pathlib import Path
from torchaudio import transforms as T
import base64

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'GLM_modules'))
sys.path.append(current_dir)
from whisper_encoder_decoder import GLM4Encoder


# 全局变量
encoder = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# 全局变量用于 tokenizer_path
TOKENIZER_PATH = None

# 全局变量用于模型路径
CONFIG_PATH = None
FEATURE_EXTRACTOR_PATH = None
FLOW_PATH = None

MAX_DURATION=90.0

# 全局变量默认值 (将在 main 中根据参数更新)
OUTPUT_DIR = Path("./.gradio_outputs").resolve()

def initialize_model(mel_cache_len=8):
    """初始化模型"""
    global encoder
    if encoder is None:
        print("="*60)
        print("正在加载模型...")
        print("="*60)
        tokenizer_path = TOKENIZER_PATH
        feature_extractor_path = FEATURE_EXTRACTOR_PATH
        flow_path = FLOW_PATH
        encoder = GLM4Encoder(tokenizer_path=tokenizer_path, feature_extractor_path = feature_extractor_path, flow_path = flow_path,  mel_cache_len=mel_cache_len).to(device)
        encoder.eval()
        print("="*60)
        print("✅ 模型加载完成！")
        print(f"设备: {device}")
        print(f"Mel cache length: {mel_cache_len}")
        print("="*60)
    return encoder



def calculate_rms(waveform):
    """计算音频的 RMS 值"""
    return torch.sqrt(torch.mean(waveform ** 2)).item()

def normalize_volume(waveform, target_rms):
    """归一化音频音量到目标 RMS 值"""
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    if current_rms > 0:
        scale = target_rms / current_rms
        waveform = waveform * scale
    return waveform

def find_loudest_segment(waveform, sr, segment_duration, window_size=0.1):
    """
    找出音频中音量最大的连续片段
    
    Args:
        waveform: 音频波形 [1, samples]
        sr: 采样率
        segment_duration: 目标片段时长（秒）
        window_size: 滑动窗口大小（秒）
    
    Returns:
        截取的音频片段 [1, samples]
    """
    if waveform.shape[1] <= segment_duration * sr:
        return waveform
    
    segment_samples = int(segment_duration * sr)
    window_samples = int(window_size * sr)
    audio_1d = waveform.squeeze(0)
    
    # 使用较大的步长提高效率
    hop_length = window_samples // 4
    energies = []
    for i in range(0, len(audio_1d) - window_samples + 1, hop_length):
        window = audio_1d[i:i + window_samples]
        energy = torch.sqrt(torch.mean(window ** 2))
        energies.append(energy.item())
    
    energies = np.array(energies)
    
    # 平滑能量曲线
    kernel_size = max(1, int(segment_duration / window_size))
    kernel = np.ones(kernel_size) / kernel_size
    if len(energies) >= kernel_size:
        smoothed_energies = np.convolve(energies, kernel, mode='valid')
    else:
        smoothed_energies = energies
    
    # 找到能量最大的位置
    max_idx = np.argmax(smoothed_energies)
    start_sample = max_idx * hop_length
    end_sample = start_sample + segment_samples
    
    if end_sample > waveform.shape[1]:
        end_sample = waveform.shape[1]
        start_sample = max(0, end_sample - segment_samples)
    
    print(f"[INFO] 找到最响片段: {start_sample/sr:.2f}s - {end_sample/sr:.2f}s")
    return waveform[:, start_sample:end_sample]


def calculate_rms(waveform):
    """计算音频的 RMS 值"""
    return torch.sqrt(torch.mean(waveform ** 2)).item()

def normalize_volume(waveform, target_rms):
    """归一化音频音量到目标 RMS 值"""
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    if current_rms > 0:
        scale = target_rms / current_rms
        waveform = waveform * scale
    return waveform

def find_loudest_segment(waveform, sr, segment_duration, window_size=0.1):
    """找出音频中音量最大的连续片段"""
    if waveform.shape[1] <= segment_duration * sr:
        return waveform
    segment_samples = int(segment_duration * sr)
    window_samples = int(window_size * sr)
    audio_1d = waveform.squeeze(0)
    hop_length = window_samples // 4
    energies = []
    for i in range(0, len(audio_1d) - window_samples + 1, hop_length):
        window = audio_1d[i:i + window_samples]
        energy = torch.sqrt(torch.mean(window ** 2))
        energies.append(energy.item())
    energies = np.array(energies)
    kernel_size = max(1, int(segment_duration / window_size))
    kernel = np.ones(kernel_size) / kernel_size
    if len(energies) >= kernel_size:
        smoothed_energies = np.convolve(energies, kernel, mode='valid')
    else:
        smoothed_energies = energies
    max_idx = np.argmax(smoothed_energies)
    start_sample = max_idx * hop_length
    end_sample = start_sample + segment_samples
    if end_sample > waveform.shape[1]:
        end_sample = waveform.shape[1]
        start_sample = max(0, end_sample - segment_samples)
    print(f"[INFO] 找到最响片段: {start_sample/sr:.2f}s - {end_sample/sr:.2f}s")
    return waveform[:, start_sample:end_sample]


def calculate_rms(waveform):
    """计算音频的 RMS 值"""
    return torch.sqrt(torch.mean(waveform ** 2)).item()

def normalize_volume(waveform, target_rms):
    """归一化音频音量到目标 RMS 值"""
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    if current_rms > 0:
        scale = target_rms / current_rms
        waveform = waveform * scale
    return waveform

def find_loudest_segment(waveform, sr, segment_duration, window_size=0.1):
    """找出音频中音量最大的连续片段"""
    if waveform.shape[1] <= segment_duration * sr:
        return waveform
    segment_samples = int(segment_duration * sr)
    window_samples = int(window_size * sr)
    audio_1d = waveform.squeeze(0)
    hop_length = window_samples // 4
    energies = []
    for i in range(0, len(audio_1d) - window_samples + 1, hop_length):
        window = audio_1d[i:i + window_samples]
        energy = torch.sqrt(torch.mean(window ** 2))
        energies.append(energy.item())
    energies = np.array(energies)
    kernel_size = max(1, int(segment_duration / window_size))
    kernel = np.ones(kernel_size) / kernel_size
    if len(energies) >= kernel_size:
        smoothed_energies = np.convolve(energies, kernel, mode='valid')
    else:
        smoothed_energies = energies
    max_idx = np.argmax(smoothed_energies)
    start_sample = max_idx * hop_length
    end_sample = start_sample + segment_samples
    if end_sample > waveform.shape[1]:
        end_sample = waveform.shape[1]
        start_sample = max(0, end_sample - segment_samples)
    print(f"[INFO] 找到最响片段: {start_sample/sr:.2f}s - {end_sample/sr:.2f}s")
    return waveform[:, start_sample:end_sample]

def process_gradio_audio(audio_data, max_duration=30.0):
    """
    处理 Gradio 音频数据，参考 stable-audio 的处理方式
    
    Args:
        audio_data: tuple (sample_rate, audio_array)
        max_duration: 最大音频时长（秒），默认30秒
    
    Returns:
        tuple: (torch.Tensor, sample_rate, duration) 处理后的音频张量 [channels, samples]
    """
    sample_rate, audio_array = audio_data
    
    print(f"[DEBUG] 原始音频信息:")
    print(f"  - Sample rate: {sample_rate}")
    print(f"  - Array shape: {audio_array.shape}")
    print(f"  - Array dtype: {audio_array.dtype}")
    print(f"  - Array range: [{audio_array.min():.6f}, {audio_array.max():.6f}]")
    
    # 根据 dtype 转换为 torch tensor (参考 stable-audio)
    if audio_array.dtype == np.float32:
        audio = torch.from_numpy(audio_array)
    elif audio_array.dtype == np.int16:
        audio = torch.from_numpy(audio_array).float().div(32767)
    elif audio_array.dtype == np.int32:
        audio = torch.from_numpy(audio_array).float().div(2147483647)
    else:
        raise ValueError(f"Unsupported audio data type: {audio_array.dtype}")
    
    # 处理维度
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # [1, n]
    elif audio.dim() == 2:
        audio = audio.transpose(0, 1)  # [n, 2] -> [2, n]
    
    # 如果是多声道，取平均到单声道
    if audio.shape[0] > 1:
        print(f"[INFO] 检测到 {audio.shape[0]} 声道音频，转换为单声道")
        audio = audio.mean(dim=0, keepdim=True)
    
    # 计算音频时长
    duration = audio.shape[1] / sample_rate
    print(f"[INFO] 音频时长: {duration:.2f} 秒")
    
    # 限制最大时长
    if max_duration is not None and duration > max_duration:
        max_samples = int(max_duration * sample_rate)
        audio = audio[:, :max_samples]
        print(f"[WARNING] 音频超过最大时长 {max_duration}秒，已截断到 {max_duration}秒")
        duration = max_duration
    
    print(f"[DEBUG] 转换后音频:")
    print(f"  - Tensor shape: {audio.shape}")
    print(f"  - Tensor range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"  - Duration: {duration:.2f}s")
    
    return audio, sample_rate, duration


def save_audio_for_gradio(audio_tensor, sample_rate, prefix="output"):
    """
    保存音频文件供 Gradio 使用
    参考 stable-audio 的保存方式
    """
    try:
        # 确保音频是 2D 张量 [channels, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # 转换为 int16 格式 (参考 stable-audio)
        # audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        audio_normalized = audio_tensor.to(torch.float32)
        
        # 归一化到 [-1, 1]
        max_val = torch.max(torch.abs(audio_normalized))
        if max_val > 0:
            audio_normalized = audio_normalized.div(max_val)
        audio_normalized = audio_normalized.clamp(-1, 1)
        
        # 转换为 int16
        audio_int16 = audio_normalized.mul(32767).to(torch.int16).cpu()
        
        # 使用时间戳防止浏览器缓存
        import time
        timestamp = int(time.time() * 1000)
        output_path = OUTPUT_DIR / f"{prefix}_{timestamp}.wav"
        
        print(f"[INFO] 保存音频到: {output_path}")
        print(f"[INFO] 音频形状: {audio_int16.shape}")
        print(f"[INFO] 采样率: {sample_rate}")
        print(f"[INFO] 音频范围: [{audio_int16.min()}, {audio_int16.max()}]")
        
        torchaudio.save(
            str(output_path),
            audio_int16,
            sample_rate=sample_rate
        )
        
        print(f"[SUCCESS] 音频已保存: {output_path}")
        
        # 验证文件
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"[INFO] 文件大小: {file_size} bytes")
            
            # 读取验证
            try:
                verify_audio, verify_sr = torchaudio.load(str(output_path))
                print(f"[INFO] 验证读取成功: shape={verify_audio.shape}, sr={verify_sr}")
            except Exception as e:
                print(f"[ERROR] 验证读取失败: {e}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"[ERROR] 保存音频失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def load_audio_for_frontend(audio_path: str):
    if not audio_path or not os.path.exists(audio_path):
        print(f"[WARN] 无法找到音频文件: {audio_path}")
        return None
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(torch.float32)
    if waveform.dim() > 1 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)
    return sample_rate, waveform.cpu().numpy()

def get_audio_html(file_path):
    """将音频文件转换为 HTML 播放器代码"""
    if not file_path or not os.path.exists(file_path):
        return "<div>无音频文件</div>"
    
    try:
        with open(file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode('utf-8')
            
        filename = os.path.basename(file_path)
        return f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa;">
            <div style="margin-bottom: 8px; font-size: 0.9em; color: #666;">📄 {filename}</div>
            <audio controls style="width: 100%">
                <source src="data:audio/wav;base64,{b64_data}" type="audio/wav">
            </audio>
            <div style="margin-top: 5px; text-align: right;">
                <a href="data:audio/wav;base64,{b64_data}" download="{filename}" target="_blank">⬇️ 下载</a>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div>加载失败: {str(e)}</div>"

def reload_audio(audio_path: str):
    return get_audio_html(audio_path)

def process_audio_nonstreaming(
    input_audio,
    reference_audio,
    reference_ratio = 0.8,
    use_spk_embedding = True,
    use_prompt_speech = True,
    mel_cache_len = 8
):
    """非流式音频处理"""
    try:
        print("\n" + "="*60)
        print("开始非流式处理")
        print("="*60)
        
        # 初始化模型
        model = initialize_model(mel_cache_len=mel_cache_len)
        
        if input_audio is None:
            return None, "❌ 请上传输入音频", None
        if reference_audio is None:
            return None, "❌ 请上传参考音频", None
        
        # 处理输入音频（限制最大30秒）
        print("\n[STEP 1] 处理输入音频...")
        input_tensor, input_sr, input_duration = process_gradio_audio(input_audio, max_duration=MAX_DURATION)
        input_rms = calculate_rms(input_tensor)
        input_rms = calculate_rms(input_tensor)
        temp_input = OUTPUT_DIR / "temp_input.wav"
        torchaudio.save(str(temp_input), input_tensor, input_sr)
        print(f"[INFO] 临时输入文件已保存: {temp_input}")
        print(f"[INFO] 输入音频 RMS: {input_rms:.6f}")
        print(f"[INFO] 输入音频 RMS: {input_rms:.6f}")
        
        # 处理参考音频（智能截取 + 音量归一化）
        print("\n[STEP 2] 处理参考音频...")
        ref_tensor, ref_sr, ref_duration = process_gradio_audio(reference_audio, max_duration=10.0)
        
        # 智能截取最响片段
        target_duration = reference_ratio * min(ref_duration, 10.0)
        if ref_duration <= target_duration:
            print(f"[INFO] 参考音频时长 {ref_duration:.2f}s <= 目标时长 {target_duration:.2f}s，不需要截取")
            ref_segment = ref_tensor
        else:
            ref_segment = find_loudest_segment(ref_tensor, ref_sr, target_duration)
        
        # 音量归一化
        ref_rms_before = calculate_rms(ref_segment)
        ref_segment = normalize_volume(ref_segment, input_rms)
        ref_rms_after = calculate_rms(ref_segment)
        print(f"[INFO] 参考音频 RMS: {ref_rms_before:.6f} -> {ref_rms_after:.6f}")
        
        ref_duration_final = ref_segment.shape[1] / ref_sr
        temp_reference = OUTPUT_DIR / "temp_reference.wav"
        torchaudio.save(str(temp_reference), ref_segment, ref_sr)
        print(f"[INFO] 临时参考文件已保存: {temp_reference}")
        print(f"[INFO] 参考音频处理后时长: {ref_duration_final:.2f}s")
        
        # 编码输入音频
        print("\n[STEP 3] 正在编码音频...")
        audio_tokens = model.encode_token(str(temp_input))
        print(f"[INFO] ✅ 生成了 {len(audio_tokens)} 个 tokens")
        
        # 非流式解码
        print(f"\n[STEP 4] 正在进行非流式解码...")
        print(f"[INFO] 参数: use_spk_embedding={use_spk_embedding}, use_prompt_speech={use_prompt_speech}")
        
        result = model.decode(
            [audio_tokens],
            prompt_speech=str(temp_reference),
            use_spk_embedding=use_spk_embedding,
            use_prompt_speech=use_prompt_speech,
            device=device
        )
        
        # 保存输出
        print(f"\n[STEP 5] 保存输出音频...")
        output_audio = result['syn_wav_list'][0]
        print(f"[INFO] 输出音频形状: {output_audio.shape}")
        print(f"[INFO] 输出音频范围: [{output_audio.min():.6f}, {output_audio.max():.6f}]")
        
        output_path = save_audio_for_gradio(
            output_audio,
            sample_rate=24000,
            prefix="nonstreaming_output"
        )
                
        if output_path is None:
            return None, "❌ 保存音频失败", None
        
        info = (
            "✅ 非流式解码完成\n"
            f"输入音频时长: {input_duration:.2f}秒\n"
            f"参考音频时长: {ref_duration_final:.2f}秒\n"
            f"参考音频截取比例: {reference_ratio}\n"
            f"Token 数量: {len(audio_tokens)}\n"
            f"使用说话人嵌入: {use_spk_embedding}\n"
            f"使用提示语音: {use_prompt_speech}\n"
            f"Mel cache length: {mel_cache_len}\n"
            f"输出文件: {output_path}"
        )
        
        print("\n" + "="*60)
        print(info)
        print("="*60 + "\n")
        
        # 修改：直接返回 output_path
        return get_audio_html(output_path), info, output_path        
    
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理失败: {str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        print("\n" + "="*60)
        print(error_msg)
        print("="*60 + "\n")
        return None, error_msg, None



def create_ui():
    """创建 Gradio 界面"""
    # 先初始化模型
    model = initialize_model(mel_cache_len=8)
    
    with gr.Blocks(title="Streaming Codec 变声器测试", theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
            <div style='text-align: center'>
                <h1>🎙️ Streaming Codec 变声器测试</h1>
                <p>上传输入音频和参考音频，测试非流式和流式解码效果</p>
            </div>
            """
        )
        
        nonstream_state = gr.State(value=None)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输入设置")
                
                input_audio = gr.Audio(
                    label="输入音频 (待转换的音频)",
                    type="numpy",
                    sources=["upload", "microphone"]
                )
                
                reference_audio = gr.Audio(
                    label="参考音频 (目标音色)",
                    type="numpy",
                    sources=["upload", "microphone"]
                )
                
                gr.Markdown("### ⚙️ 参数设置")
                
                reference_ratio = gr.Slider(
                    minimum=0.3,
                    maximum=1.0,
                    step=0.1,
                    value=0.8,
                    label="参考音频截取比例",
                    info="从参考音频中截取最响部分的比例 (0.3-1.0)"
                )
                
                mel_cache_len = 8
                use_spk_embedding = True
                use_prompt_speech = True
                # mel_cache_len = gr.Slider(
                #     minimum=4,
                #     maximum=16,
                #     step=1,
                #     value=8,
                #     label="Mel Cache Length",
                #     info="Vocoder解码时overlap的长度，影响音质连续性"
                # )
                
                # use_spk_embedding = gr.Checkbox(
                #     label="使用说话人嵌入",
                #     value=True
                # )
                # use_prompt_speech = gr.Checkbox(
                #     label="使用提示语音",
                #     value=True
                # )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎵 非流式解码")
                
                with gr.Row():
                    nonstream_button = gr.Button(
                        "🚀 非流式解码",
                        variant="primary",
                        size="lg"
                    )
                
                nonstream_output = gr.HTML(label="非流式输出")
                nonstream_reload = gr.Button("🔁 重新加载非流式音频")
                nonstream_info = gr.Textbox(
                    label="处理信息",
                    lines=8,
                    interactive=False
                )
        
        # 绑定事件
        nonstream_button.click(
            fn=process_audio_nonstreaming,
            inputs=[input_audio, reference_audio, reference_ratio],
            outputs=[nonstream_output, nonstream_info, nonstream_state]
        )


        nonstream_reload.click(
            fn=reload_audio,
            inputs=[nonstream_state],
            outputs=[nonstream_output]
        )
        
        # 使用提示
        gr.Markdown("### 💡 使用提示")
        gr.Markdown(f"""
        1. **输入音频**: 你想要转换的原始音频
        2. **参考音频**: 目标音色的参考音频
        3. **Mel Cache Length**: Vocoder解码时overlap的长度，建议 mel_cache_len / 4 ≤ block_size
        4. **非流式解码**: 一次性处理整个音频，质量更好
        8. **输出目录**: `{OUTPUT_DIR}`
        
        **重要提示**: 
        - prompt 音频会被截断至前10s。请提供清晰的 prompt 音频
        """)
    
    return demo


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Streaming Codec Gradio Demo")
    parser.add_argument("--output_dir", type=str, default=None, help="指定音频输出目录")
    parser.add_argument("--port", type=int, default=7860, help="指定服务端口")
    parser.add_argument("--tokenizer_path", type=str, default="./SpeechTokenizerTrainer_final/generator_ckpt", help="Path to tokenizer checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to config.json (optional)")
    parser.add_argument("--feature_extractor_path", type=str, default=None, help="Path to glm-4-voice-tokenizer (optional)")
    parser.add_argument("--flow_path", type=str, default=None, help="Path to flow directory (optional)")

    args = parser.parse_args()
    
    TOKENIZER_PATH = args.tokenizer_path
    CONFIG_PATH = args.config_path
    FEATURE_EXTRACTOR_PATH = args.feature_extractor_path
    FLOW_PATH = args.flow_path

    # 确定输出目录: 命令行参数 > 环境变量 > 默认值
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).resolve()
    elif os.getenv("GRADIO_OUTPUT_DIR"):
        OUTPUT_DIR = Path(os.getenv("GRADIO_OUTPUT_DIR")).resolve()
    
    # 创建目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录 (绝对路径): {OUTPUT_DIR}")
    print(f"服务端口: {args.port}")

    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        allowed_paths=[str(OUTPUT_DIR)]
    )
    
    
    
    
