#!/bin/bash

# ===================================================================
# Streaming Codec 非流式解码 API 客户端
# 用法: ./client_streaming.sh <API_URL> <INPUT_AUDIO> <REF_AUDIO> <OUTPUT_FILE>
# ===================================================================

set -e

# 检查参数
if [ $# -ne 4 ]; then
    echo "用法: $0 <API_URL> <INPUT_AUDIO> <REF_AUDIO> <OUTPUT_FILE>"
    echo "示例: $0 http://127.0.0.1:7864 input.mp3 ref.wav output.wav"
    exit 1
fi

API_URL="$1"
INPUT="$2"
REF="$3"
OUTPUT="$4"

# 检查文件是否存在
if [ ! -f "$INPUT" ]; then
    echo "❌ 错误: 输入音频文件不存在: $INPUT"
    exit 1
fi

if [ ! -f "$REF" ]; then
    echo "❌ 错误: 参考音频文件不存在: $REF"
    exit 1
fi

# 检查依赖
for cmd in curl jq base64; do
    if ! command -v $cmd &> /dev/null; then
        echo "❌ 错误: 缺少依赖工具 $cmd"
        exit 1
    fi
done

# echo "==================================================="
# echo "🚀 Streaming Codec API 客户端"
# echo "==================================================="
# echo "API URL: $API_URL"
# echo "输入音频: $INPUT"
# echo "参考音频: $REF"
# echo "输出文件: $OUTPUT"
# echo "==================================================="

# 1. 上传音频文件
# echo "[1/3] 上传音频文件..."
UPLOAD_RESP=$(curl -s -X POST "$API_URL/gradio_api/upload" \
    -F "files=@$INPUT" \
    -F "files=@$REF")

INPUT_PATH=$(echo "$UPLOAD_RESP" | jq -r '.[0]')
REF_PATH=$(echo "$UPLOAD_RESP" | jq -r '.[1]')

if [ "$INPUT_PATH" == "null" ] || [ -z "$INPUT_PATH" ]; then
    echo "❌ 上传失败，请检查API服务是否运行"
    exit 1
fi

# echo "   ✅ 输入音频: $INPUT_PATH"
# echo "   ✅ 参考音频: $REF_PATH"

# 2. 发起处理请求
# echo "[2/3] 发起处理请求..."
PREDICT_RESP=$(curl -s -X POST "$API_URL/gradio_api/call/process_audio_nonstreaming" \
    -H "Content-Type: application/json" \
    -d "{\"data\":[{\"path\":\"$INPUT_PATH\",\"meta\":{\"_type\":\"gradio.FileData\"}},{\"path\":\"$REF_PATH\",\"meta\":{\"_type\":\"gradio.FileData\"}},0.8]}")

EVENT_ID=$(echo "$PREDICT_RESP" | jq -r '.event_id')

if [ "$EVENT_ID" == "null" ] || [ -z "$EVENT_ID" ]; then
    echo "❌ 请求失败: $PREDICT_RESP"
    exit 1
fi

# echo "   ✅ Event ID: $EVENT_ID"

# 3. 实时轮询SSE直到完成
# echo "[3/3] 等待处理完成（实时轮询）..."
START_TIME=$(date +%s)
SUCCESS=0

curl -N -s "$API_URL/gradio_api/call/process_audio_nonstreaming/$EVENT_ID" | while IFS= read -r line; do
    # 显示进度（heartbeat事件）
    if [[ "$line" =~ ^event:\ heartbeat ]]; then
        ELAPSED=$(($(date +%s) - START_TIME))
        # echo -ne "   ⏳ 处理中... ${ELAPSED}s\r"
    fi
    
    # 检查是否完成（data行包含JSON数组）
    if [[ "$line" =~ ^data:\ \[.+\]$ ]]; then
        # 提取data:后的内容
        DATA="${line#data: }"
        
        # 使用jq解析JSON数组的第一个元素（HTML字符串）
        HTML=$(echo "$DATA" | jq -r '.[0]' 2>/dev/null)
        
        # 从HTML中提取Base64数据
        if [[ "$HTML" =~ data:audio/wav\;base64\,([A-Za-z0-9+/=]+) ]]; then
            BASE64_DATA="${BASH_REMATCH[1]}"
            
            # echo ""
            # echo "   ✅ 检测到完成事件，正在保存..."
            
            # 解码Base64到输出文件（tr删除所有换行和空格）
            echo "$BASE64_DATA" | tr -d '\n\r ' | base64 -d > "$OUTPUT"
            
            if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
                FILE_SIZE=$(ls -lh "$OUTPUT" | awk '{print $5}')
                TOTAL_TIME=$(($(date +%s) - START_TIME))
                # echo "==================================================="
                # echo "🎉 处理完成！"
                # echo "   输出文件: $OUTPUT"
                # echo "   文件大小: $FILE_SIZE"
                # echo "   总耗时: ${TOTAL_TIME}s"
                # echo "==================================================="
                SUCCESS=1
                # 终止curl进程
                pkill -P $$ curl 2>/dev/null || true
                exit 0
            else
                echo "❌ Base64解码失败"
                exit 1
            fi
        fi
    fi
    
    # 检查错误事件
    if [[ "$line" =~ ^event:\ error ]]; then
        echo ""
        echo "❌ 服务器返回错误事件"
        exit 1
    fi
done

# 检查是否成功（通过检查输出文件）
if [ ! -f "$OUTPUT" ] || [ ! -s "$OUTPUT" ]; then
    echo ""
    echo "❌ 未能获取处理结果"
    exit 1
fi
