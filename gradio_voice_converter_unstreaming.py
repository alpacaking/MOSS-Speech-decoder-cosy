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

sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/GLM_modules')
sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec')
from whisper_encoder_decoder import GLM4Encoder


# 全局变量
encoder = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# 全局变量默认值 (将在 main 中根据参数更新)
OUTPUT_DIR = Path("./.gradio_outputs").resolve()

def initialize_model(mel_cache_len=8):
    """初始化模型"""
    global encoder
    if encoder is None:
        print("="*60)
        print("正在加载模型...")
        print("="*60)
        tokenizer_path = '/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt'
        encoder = GLM4Encoder(tokenizer_path=tokenizer_path, mel_cache_len=mel_cache_len).to(device)
        encoder.eval()
        print("="*60)
        print("✅ 模型加载完成！")
        print(f"设备: {device}")
        print(f"Mel cache length: {mel_cache_len}")
        print("="*60)
    return encoder


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
    if duration > max_duration:
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
        input_tensor, input_sr, input_duration = process_gradio_audio(input_audio, max_duration=30.0)
        temp_input = OUTPUT_DIR / "temp_input.wav"
        torchaudio.save(str(temp_input), input_tensor, input_sr)
        print(f"[INFO] 临时输入文件已保存: {temp_input}")
        
        # 处理参考音频（限制最大10秒）
        print("\n[STEP 2] 处理参考音频...")
        ref_tensor, ref_sr, ref_duration = process_gradio_audio(reference_audio, max_duration=10.0)
        temp_reference = OUTPUT_DIR / "temp_reference.wav"
        torchaudio.save(str(temp_reference), ref_tensor, ref_sr)
        print(f"[INFO] 临时参考文件已保存: {temp_reference}")
        
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
            f"参考音频时长: {ref_duration:.2f}秒\n"
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


def process_audio_streaming(
    input_audio,
    reference_audio,
    block_size,
    max_token_len,
    use_spk_embedding = True,
    use_prompt_speech = True,
    mel_cache_len = 8
):
    """流式音频处理"""
    try:
        print("\n" + "="*60)
        print("开始流式处理")
        print("="*60)
        
        # 初始化模型
        model = initialize_model(mel_cache_len=mel_cache_len)
        
        if input_audio is None:
            return None, "❌ 请上传输入音频", None
        if reference_audio is None:
            return None, "❌ 请上传参考音频", None
        
        # 参数检查
        token_mel_ratio = 4  # 根据代码中的设置
        min_mel_cache = block_size * token_mel_ratio
        
        warning_msg = ""
        if mel_cache_len < min_mel_cache:
            warning_msg = f"⚠️ 警告: mel_cache_len ({mel_cache_len}) < block_size * token_mel_ratio ({min_mel_cache})\n"
            print(warning_msg)
        
        # 处理输入音频（限制最大30秒）
        print("\n[STEP 1] 处理输入音频...")
        input_tensor, input_sr, input_duration = process_gradio_audio(input_audio, max_duration=30.0)
        temp_input = OUTPUT_DIR / "temp_input_stream.wav"
        torchaudio.save(str(temp_input), input_tensor, input_sr)
        print(f"[INFO] 临时输入文件已保存: {temp_input}")
        
        # 处理参考音频（限制最大10秒）
        print("\n[STEP 2] 处理参考音频...")
        ref_tensor, ref_sr, ref_duration = process_gradio_audio(reference_audio, max_duration=10.0)
        temp_reference = OUTPUT_DIR / "temp_reference_stream.wav"
        torchaudio.save(str(temp_reference), ref_tensor, ref_sr)
        print(f"[INFO] 临时参考文件已保存: {temp_reference}")
        
        # 编码输入音频
        print("\n[STEP 3] 正在编码音频...")
        audio_tokens = model.encode_token(str(temp_input))
        print(f"[INFO] ✅ 生成了 {len(audio_tokens)} 个 tokens")
        
        # 流式解码
        print(f"\n[STEP 4] 正在进行流式解码...")
        print(f"[INFO] 参数:")
        print(f"  - block_size: {block_size}")
        print(f"  - max_token_len: {max_token_len if max_token_len else 'None (无限制)'}")
        print(f"  - mel_cache_len: {mel_cache_len}")
        print(f"  - use_spk_embedding: {use_spk_embedding}")
        print(f"  - use_prompt_speech: {use_prompt_speech}")
        
        result = model.decode_streaming(
            [audio_tokens],
            prompt_speech=str(temp_reference),
            use_spk_embedding=use_spk_embedding,
            use_prompt_speech=use_prompt_speech,
            block_size=block_size,
            max_token_len=max_token_len
        )
        
        # 保存输出
        print(f"\n[STEP 5] 保存输出音频...")
        output_audio = result['syn_wav_list'][0]
        print(f"[INFO] 输出音频形状: {output_audio.shape}")
        print(f"[INFO] 输出音频范围: [{output_audio.min():.6f}, {output_audio.max():.6f}]")
        
        output_path = save_audio_for_gradio(
            output_audio,
            sample_rate=24000,
            prefix="streaming_output"
        )
                
        if output_path is None:
            return None, "❌ 保存音频失败", None
        
        info = warning_msg
        info += f"✅ 流式解码完成\n"
        info += f"输入音频时长: {input_duration:.2f}秒\n"
        info += f"参考音频时长: {ref_duration:.2f}秒\n"
        info += f"Token 数量: {len(audio_tokens)}\n"
        info += f"Block size: {block_size}\n"
        info += f"Max token len: {max_token_len if max_token_len else 'None (无限制)'}\n"
        info += f"Mel cache length: {mel_cache_len}\n"
        info += f"使用说话人嵌入: {use_spk_embedding}\n"
        info += f"使用提示语音: {use_prompt_speech}\n"
        info += f"输出文件: {output_path}"
        
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
        stream_state = gr.State(value=None)
        
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
                
                # gr.Markdown("### ⚙️ 共同参数")
                
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
                
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🌊 流式解码")
                
                with gr.Row():
                    block_size = gr.Slider(
                        minimum=2,
                        maximum=10,
                        step=1,
                        value=5,
                        label="Block Size",
                        info="推理时每一步会推出的token数"
                    )
                    
                    max_token_len = gr.Slider(
                        minimum=10,
                        maximum=50,
                        step=5,
                        value=15,
                        label="Max Token Length",
                        info="每步送入flow模型的最大token数，设为0表示不限制"
                    )
                
                with gr.Row():
                    stream_button = gr.Button(
                        "⚡ 流式解码",
                        variant="primary",
                        size="lg"
                    )
                
                stream_output = gr.HTML(label="流式输出")
                stream_reload = gr.Button("🔁 重新加载流式音频")
                stream_info = gr.Textbox(
                    label="处理信息",
                    lines=8,
                    interactive=False
                )
        
        # 绑定事件
        nonstream_button.click(
            fn=process_audio_nonstreaming,
            inputs=[input_audio, reference_audio],
            outputs=[nonstream_output, nonstream_info, nonstream_state]
        )

        stream_button.click(
            fn=process_audio_streaming,
            inputs=[input_audio, reference_audio, block_size, max_token_len],
            outputs=[stream_output, stream_info, stream_state]
        )

        nonstream_reload.click(
            fn=reload_audio,
            inputs=[nonstream_state],
            outputs=[nonstream_output]
        )
        stream_reload.click(
            fn=reload_audio,
            inputs=[stream_state],
            outputs=[stream_output]
        )
        
        # 使用提示
        gr.Markdown("### 💡 使用提示")
        gr.Markdown(f"""
        1. **输入音频**: 你想要转换的原始音频
        2. **参考音频**: 目标音色的参考音频
        3. **Mel Cache Length**: Vocoder解码时overlap的长度，建议 mel_cache_len / 4 ≤ block_size
        4. **非流式解码**: 一次性处理整个音频，质量更好
        5. **流式解码**: 分块处理音频，延迟更低，适合实时场景
        6. **Block Size**: 控制流式处理的块大小 (2-10)
        7. **Max Token Length**: 限制每次推理的最大token数量 (10-50)，必须 ≥ block_size + pre_lookahead_len(3) ，且最好多留有一些余量
        8. **输出目录**: `{OUTPUT_DIR}`
        
        **参数建议**:
        - mel_cache_len = 8, block_size = 5, max_token_len = 15 (默认配置)
        
        **重要提示**: 
        - mel_cache_len / 4 应该 ≤ block_size
        - max_token_len 必须 ≥ block_size + pre_lookahead_len (通常 pre_lookahead_len ≈ 5)
        """)
    
    return demo


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Streaming Codec Gradio Demo")
    parser.add_argument("--output_dir", type=str, default=None, help="指定音频输出目录")
    parser.add_argument("--port", type=int, default=7860, help="指定服务端口")
    args = parser.parse_args()

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
    
    
    
    
