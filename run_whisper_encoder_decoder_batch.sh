#!/bin/bash
set -euo pipefail

# ========== 配置参数 ==========
# INPUT_AUDIO_DIR="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/GLM_modules/input_wavs"
# INPUT_AUDIO_DIR="/inspire/hdd/project/embodied-multimodality/public/datasets/Amphion___Emilia/raw/ZH/ZH_B00007/ZH_B00007_S00006/mp3"
# # PROMPT_AUDIO="/inspire/hdd/project/embodied-multimodality/public/datasets/LibriTTS/LibriTTS/train-clean-360/98/121658/98_121658_000008_000000.wav"
# PROMPT_AUDIO="/inspire/hdd/project/embodied-multimodality/public/datasets/Amphion___Emilia/raw/ZH/ZH_B00007/ZH_B00007_S00006/mp3/ZH_B00007_S00006_W000001.mp3"
# # PROMPT_AUDIO="/inspire/hdd/project/embodied-multimodality/public/datasets/Amphion___Emilia/raw/ZH/ZH_B00024/ZH_B00024_S00000/mp3/ZH_B00024_S00000_W000001.mp3"
# OUTPUT_DIR="./inference_outputs"
# MODEL_PATH="/inspire/hdd/project/embodied-multimodality/public/kwchen/MOSS-Speech-decoder/MOSS-Speech-decoder/output/train_semantic_to_acoustic_emilia_1.7b_ltrmlayers8_heads32_rvq8_inputtrm_partial_loss_delay5/checkpoint-170000"
# GLM4_CHECKPOINT="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt"
# TAC_CONFIG="/inspire/hdd/project/embodied-multimodality/public/ytgong/SpeechTokenizer3_v4.26/exp/v26.2.a/with_disc_mpd_mscstftd_fix_encoder_gen_lr_1e-5_disc_lr_1e-4_from_v26.1.a_24khz_emilia_blbl_raw_data_12.5hz_trm_ratio_240_2_2_2_dim_512_512_512_512_layers_8_8_8_8_rvq32_quantizer_dropout_1.0_no_llm_no_disc_torch_ddp_760000_steps/config.yaml"
# TAC_CHECKPOINT="/inspire/hdd/project/embodied-multimodality/public/ytgong/SpeechTokenizer3_v4.26/exp/v26.2.a/with_disc_mpd_mscstftd_fix_encoder_gen_lr_1e-5_disc_lr_1e-4_from_v26.1.a_24khz_emilia_blbl_raw_data_12.5hz_trm_ratio_240_2_2_2_dim_512_512_512_512_layers_8_8_8_8_rvq32_quantizer_dropout_1.0_no_llm_no_disc_torch_ddp_760000_steps_accelerator/SpeechTokenizerTrainer_02100000/generator_ckpt"
# MAX_FILES=20  # 限制处理文件数量，设为空则处理全部

INPUT_DIR="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/GLM_modules/GLM-4-Voice/audios/role_play_prompt/input"
OUTPUT_DIR="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/batch_test_results_b5_mcl8_mtl40_5"
# PROMPT_SPEECH="/inspire/hdd/project/embodied-multimodality/public/datasets/Amphion___Emilia/raw/ZH/ZH_B00007/ZH_B00007_S00006/mp3/ZH_B00007_S00006_W000001.mp3"
PROMPT_SPEECH="/inspire/hdd/project/embodied-multimodality/public/kwchen/datasets_seed_tts_eval/seedtts_testset/zh/prompt-wavs/00000309-00000300.wav"
TOKENIZER_PATH="/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt"


# ========== 环境激活 ==========
export HOME=/inspire/hdd/project/embodied-multimodality/public/kwchen
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /inspire/hdd/project/embodied-multimodality/public/kwchen/.conda/envs/moss-speech2

# ========== 切换工作目录 ==========
cd /inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec || { 
    echo "❌ 无法切换到目录"; 
    exit 1; 
}

# ========== 运行推理 ==========
echo "=========================================="
echo "开始推理..."
echo "输入目录: ${INPUT_DIR}"
echo "Prompt 音频: ${PROMPT_SPEECH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
#     # --max_token_len 40 \
python whisper_encoder_decoder_batch.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_speech ${PROMPT_SPEECH} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --block_size 5 \
    --mel_cache_len 8 \
    --max_token_len 40 \
    --device cuda \
    --seed 42 \
    --use_spk_embedding \
    --use_prompt_speech \
    --extensions .wav .mp3 .flac

echo "=========================================="
echo "✅ 推理完成！"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
