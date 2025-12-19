#!/bin/bash
set -e

# ========== 注意！！！可修改配置区域开始 ==========
# 1. 指定输出音频的缓存目录 (绝对路径)
OUTPUT_DIR="/inspire/hdd/project/embodied-multimodality/public/kwchen/Streaming_Codec/.gradio_output"

# 2. 指定服务端口
PORT=7864

# 3. 指定 Python 脚本路径
SCRIPT_PATH="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/gradio_voice_converter_unstreaming.py"
# ========== 注意！！！可修改配置区域结束 ==========

# ========== 环境设置 ==========
# 激活 conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate moss-speech2

# ========== 启动服务 ==========
echo "=================================================="
echo "🚀 正在启动 Streaming Codec Gradio Demo..."
echo "📂 输出目录: ${OUTPUT_DIR}"
echo "🔌 端口: ${PORT}"
echo "📜 脚本路径: ${SCRIPT_PATH}"
echo "=================================================="

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

# 启动 Python 脚本，传入参数
python "${SCRIPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --port "${PORT}" \
    --tokenizer_path "/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt" \