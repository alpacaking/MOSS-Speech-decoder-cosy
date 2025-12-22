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
import json # Import json module for handling JSONL files

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Global variables for ONNX session and executor (initialized in __main__)
ort_session = None
executor = None

# Modify single_job to accept an index and the data entry, then return index and tokens
def single_job(index, data_entry):
    """
    Processes a single audio entry to extract speech tokens.

    Args:
        index (int): The original index of the data_entry in the input list.
        data_entry (dict): A dictionary containing 'audio_path' and other fields.

    Returns:
        tuple: (index, speech_token_list) where speech_token_list is [[tokens...]].
               Returns [[ ]] if processing fails or audio is too long.
    """
    audio_path = data_entry["audio_path"]
    
    try:
        audio, sample_rate = torchaudio.load(audio_path, backend='soundfile')
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        
        # Convert audio to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Check audio length
        if audio.shape[1] / 16000 > 30:
            logging.warning(f'Audio {audio_path} is longer than 30s, skipping token extraction. Returning empty tokens.')
            speech_token = []
        else:
            # Ensure audio is on CPU for whisper.log_mel_spectrogram if it was on GPU (unlikely for torchaudio.load)
            # and ensure it's a 2D tensor (batch, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0) # Add batch dimension if missing

            feat = whisper.log_mel_spectrogram(audio, n_mels=128)
            
            # Ensure feat is numpy array and its type matches ONNX input expectation (typically float32)
            feat_np = feat.detach().cpu().numpy().astype(np.float32)
            
            # The input shape to ONNX session expects [batch, channels, time_steps] for mel,
            # but whisper.log_mel_spectrogram returns [mels, time_steps].
            # For ONNX, we generally need [batch_size, num_mels, num_frames]
            # Assuming the ONNX model expects input shape [1, 128, N]
            if feat_np.ndim == 2:
                feat_np = np.expand_dims(feat_np, axis=0) # Add batch dimension
            
            # The length input is also expected to be [batch_size]
            input_length_np = np.array([feat_np.shape[2]], dtype=np.int32)

            speech_token_raw = ort_session.run(
                None, 
                {
                    ort_session.get_inputs()[0].name: feat_np,
                    ort_session.get_inputs()[1].name: input_length_np
                }
            )[0].flatten().tolist()
            speech_token = speech_token_raw
            
    except Exception as e:
        logging.error(f"Error processing audio_path {audio_path}: {e}")
        speech_token = [] # Return empty tokens on error
        
    # Wrap the tokens in an outer list as requested: [[token1, token2, ...]]
    return index, [speech_token]


def main(args):
    """
    Main function to orchestrate loading data, processing, and saving results.
    """
    # 1. Load input JSONL data
    data_entries = []
    try:
        with open(args.input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                data_entries.append(json.loads(line.strip()))
        logging.info(f"Loaded {len(data_entries)} entries from {args.input_jsonl}")
    except Exception as e:
        logging.error(f"Failed to load input JSONL file {args.input_jsonl}: {e}")
        return

    # 2. Submit tasks to the thread pool executor
    futures = []
    for i, entry in enumerate(data_entries):
        futures.append(executor.submit(single_job, i, entry))

    # 3. Collect results and update data entries
    results_map = {} # Map index to speech tokens
    for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting speech tokens"):
        index, speech_token = future.result()
        results_map[index] = speech_token

    output_data = []
    for i, entry in enumerate(data_entries):
        # Create a copy to avoid modifying the original list item during iteration
        modified_entry = entry.copy() 
        # Add the 'speech_tokens' key with the extracted tokens
        modified_entry["speech_tokens"] = results_map.get(i, [[]]) # Use .get with default for safety
        output_data.append(modified_entry)
    logging.info("All speech tokens extracted and combined with original data.")

    # 4. Write output JSONL file
    try:
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            for entry in output_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        logging.info(f"Results saved to {args.output_jsonl}")
    except Exception as e:
        logging.error(f"Failed to write output JSONL file {args.output_jsonl}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract speech tokens from audio paths in a JSONL file.")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Path to the input JSONL file containing audio_path and text.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path to the output JSONL file where results will be saved.")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="Path to the ONNX model for speech token extraction.")
    parser.add_argument("--num_thread", type=int, default=16,
                        help="Number of threads for concurrent audio processing.")
    args = parser.parse_args()

    # Setup ONNX runtime session (globally accessible)
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1 # Each ONNX session will use 1 thread internally
    
    # Try CUDAExecutionProvider first, fall back to CPU if not available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] 
    
    try:
        ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)
        logging.info(f"ONNX session initialized with providers: {ort_session.get_providers()}")
    except Exception as e:
        logging.error(f"Failed to initialize ONNX session: {e}")
        logging.error("Please check --onnx_path and ONNX runtime installation.")
        exit(1)

    # Setup ThreadPoolExecutor (globally accessible)
    executor = ThreadPoolExecutor(max_workers=args.num_thread)
    logging.info(f"ThreadPoolExecutor initialized with {args.num_thread} workers.")

    # Run the main processing logic
    main(args)

    # Shut down the executor
    executor.shutdown(wait=True)
    logging.info("Executor shut down.")

