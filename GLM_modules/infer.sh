#!/usr/bin/env bash

export HOME=/inspire/hdd/project/embodied-multimodality/public/ytgong

source /opt/conda/etc/profile.d/conda.sh
conda activate /inspire/hdd/project/embodied-multimodality/public/ytgong/conda_envs/glm-4-voice
which python

work_dir=/inspire/hdd/project/embodied-multimodality/public/ytgong/GLM-4-Voice
cd ${work_dir}
export PYTHONPATH="${work_dir}"

input_dir="input_wavs"
output_dir="output_wavs"

if [ "$1" == "debug" ]; then
    debug_cmd="--debug_ip localhost --debug_port 32431 --debug 1"    
else
    debug_cmd=""
fi

cmd="python infer_glm4_tokenizer.py \
--input_dir ${input_dir} \
--output_dir ${output_dir} \
${debug_cmd}"

echo "Executing ${cmd}"

eval ${cmd}