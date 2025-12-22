#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import onnxruntime
import numpy as np
import torchaudio
import whisper
import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/..'.format(ROOT_DIR))
# sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec')
from whisper_encoder import GLM4Encoder

def single_job(utt):
    audio, sample_rate = torchaudio.load(utt2wav[utt], backend='soundfile')
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    # Convert audio to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if audio.shape[1] / 16000 > 30:
        logging.warning('do not support extract speech token for audio longer than 30s')
        speech_token = []
    else:
        speech_token=encoder.encode_batch_token([utt2wav[utt]])[0]
    return utt, speech_token


def main(args):
    all_task = [executor.submit(single_job, utt) for utt in utt2wav.keys()]
    utt2speech_token = {}
    for future in tqdm(as_completed(all_task)):
        utt, speech_token = future.result()
        utt2speech_token[utt] = speech_token
    torch.save(utt2speech_token, '{}/utt2speech_token.pt'.format(args.dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--device", type=str,default='cuda')
    parser.add_argument("--num_thread", type=int, default=32)
    args = parser.parse_args()

    utt2wav = {}
    with open('{}/wav.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]

    encoder=GLM4Encoder(tokenizer_path='/inspire/hdd/project/embodied-multimodality/public/lzjjin/Streaming-Codec/SpeechTokenizerTrainer_final/generator_ckpt').to(args.device)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)
