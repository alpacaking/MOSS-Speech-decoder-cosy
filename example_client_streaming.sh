#!/bin/bash
# Streaming Codec 客户端快速示例

# 设置路径
SCRIPT_DIR="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec"
CLIENT="$SCRIPT_DIR/client_streaming.sh"
API_URL="http://127.0.0.1:7864"

# 示例音频路径
INPUT_AUDIO="/inspire/hdd/project/embodied-multimodality/public/datasets/haitianruisheng_3/segment_00_000/00460fd0bd40915bd31e79976176c9b5_001.mp3"
REF_AUDIO="/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-100/103/1241/103_1241_000027_000007.wav"
OUTPUT_AUDIO="example_output.wav"

# 调用客户端
$CLIENT "$API_URL" "$INPUT_AUDIO" "$REF_AUDIO" "$OUTPUT_AUDIO"
