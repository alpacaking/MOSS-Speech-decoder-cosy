#!/bin/bash
set -euo pipefail

# ========== 激活环境 ==========
export HOME=/inspire/hdd/project/embodied-multimodality/public/kwchen
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /inspire/hdd/project/embodied-multimodality/public/kwchen/.conda/envs/moss-speech2

# ========== 参数设置 ==========
TOKENIZER_PATH="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt"
BENCHMARK_DIR="/inspire/hdd/project/embodied-multimodality/public/ytgong/TTS_evaluation/benchmark"
OUTPUT_DIR="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/eval_results/moss_decoder_streaming_fix_b5_mcl8_mtl40"
BLOCK_SIZE=5
mel_cache_len=8
max_token_len=40

# ========== 检测GPU数量 ==========
NGPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected ${NGPUS} GPUs"

# ========== 随机端口 ==========
MASTER_PORT=$(( ( RANDOM % 10000 )  + 20000 ))
echo "MASTER_PORT=${MASTER_PORT}"

# ========== 切换工作目录 ==========
cd /inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec || {
    echo "❌ Cannot change to directory"
    exit 1
}

# ========== 推理英文 ==========
echo "=========================================="
echo "Running inference for English..."
echo "=========================================="

torchrun --nproc-per-node=${NGPUS} \
    --master_port ${MASTER_PORT} \
    benchmark_moss_decoder.py \
    --benchmark_dir ${BENCHMARK_DIR} \
    --save_dir ${OUTPUT_DIR} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --lang en \
    --block_size ${BLOCK_SIZE} \
    --mel_cache_len ${mel_cache_len} \
    --max_token_len ${max_token_len} \
    --device cuda

echo "✅ English inference completed"

# ========== 推理中文 ==========
echo "=========================================="
echo "Running inference for Chinese..."
echo "=========================================="

MASTER_PORT=$(( ( RANDOM % 10000 )  + 20000 ))
echo "MASTER_PORT=${MASTER_PORT}"

torchrun --nproc-per-node=${NGPUS} \
    --master_port ${MASTER_PORT} \
    benchmark_moss_decoder.py \
    --benchmark_dir ${BENCHMARK_DIR} \
    --save_dir ${OUTPUT_DIR} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --lang zh \
    --block_size ${BLOCK_SIZE} \
    --mel_cache_len ${mel_cache_len} \
    --max_token_len ${max_token_len} \
    --device cuda

echo "✅ Chinese inference completed"

# ========== 运行评测 ==========
echo "=========================================="
echo "Running evaluation..."
echo "=========================================="

bash /inspire/hdd/project/embodied-multimodality/public/ytgong/TTS_evaluation/seed-tts-eval/benchmark.sh ${OUTPUT_DIR}

echo "=========================================="
echo "✅ All done! Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

# 打印结果
if [ -f "${OUTPUT_DIR}/result.json" ]; then
    echo "Evaluation results:"
    cat ${OUTPUT_DIR}/result.json
fi



# conda activate /inspire/hdd/project/embodied-multimodality/public/kwchen/.conda/envs/f5-tts
# cd /inspire/hdd/project/embodied-multimodality/public/kwchen/F5-TTS
# python run_command_every_6_hours.py 