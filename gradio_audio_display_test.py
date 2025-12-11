import gradio as gr
import os
import base64

def load_audio_to_html(path):
    """读取音频文件并转换为 HTML Audio 标签 (Base64 嵌入模式)"""
    if not path:
        return "<div>路径为空</div>"
    
    if not os.path.exists(path):
        return f"<div>文件不存在: {path}</div>"
    
    try:
        print(f"正在读取: {path}")
        with open(path, "rb") as f:
            audio_data = f.read()
            b64_data = base64.b64encode(audio_data).decode('utf-8')
            
        # 生成 HTML 音频标签，直接嵌入数据
        html = f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 20px; background: #f5f5f5; border-radius: 10px;">
            <p style="margin: 0; font-weight: bold;">{os.path.basename(path)}</p>
            <audio controls style="width: 100%">
                <source src="data:audio/wav;base64,{b64_data}" type="audio/wav">
                您的浏览器不支持 audio 标签。
            </audio>
            <a href="data:audio/wav;base64,{b64_data}" download="{os.path.basename(path)}" style="color: #2196F3; text-decoration: none;">⬇️ 下载音频</a>
        </div>
        """
        return html
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<div>读取失败: {str(e)}</div>"

with gr.Blocks() as demo:
    gr.Markdown("### 🔊 音频加载测试 (Base64 HTML 嵌入模式)")
    gr.Markdown("此模式将音频直接编码进 HTML，彻底解决文件权限和 WebSocket 传输问题。")
    
    default_path = "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/gradio_outputs/nonstreaming_output_1764834616691.wav"
    
    path_input = gr.Textbox(label="音频绝对路径", value=default_path)
    load_btn = gr.Button("加载音频", variant="primary")
    
    # 关键：使用 HTML 组件显示音频
    audio_html = gr.HTML(label="播放器")
    
    load_btn.click(load_audio_to_html, inputs=path_input, outputs=audio_html)

if __name__ == "__main__":
    print("启动测试服务...")
    demo.launch(server_name="0.0.0.0", server_port=7861)