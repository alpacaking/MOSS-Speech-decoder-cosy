import os
import argparse
import uuid
import torch
import torchaudio
import logging

from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
from tqdm import tqdm

from utils.helpers import set_logging, waiting_for_debug, load_audio, save_audio, find_audio_files

def main():
    set_logging()

    parser = argparse.ArgumentParser(description="Process audio files to reconstruct waveforms using GLM-4 Voice in chunk-wise manner.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save reconstructed audio files")
    parser.add_argument("--flow_path", type=str, default="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/glm-4-voice-decoder/", help="Path to flow model directory")
    parser.add_argument("--tokenizer_path", type=str, default="/inspire/hdd/project/embodied-multimodality/public/downloaded_ckpts/glm-4-voice-tokenizer/", help="Path to tokenizer model")
    parser.add_argument('--debug_ip', default='localhost', type=str)
    parser.add_argument('--debug_port', default=32431, type=int)
    parser.add_argument('--debug', default=0, type=int)
    
    args = parser.parse_args()
   
    if args.debug == 1:
        waiting_for_debug(args.debug_ip, args.debug_port)

    # Initialize paths
    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, "flow.pt")
    hift_checkpoint = os.path.join(args.flow_path, "hift.pt")
    device = "cuda"

    # Initialize models
    audio_decoder = AudioDecoder(
        config_path=flow_config,
        flow_ckpt_path=flow_checkpoint,
        hift_ckpt_path=hift_checkpoint,
        device=device
    )
    whisper_model = WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Supported audio extensions
    audio_extensions = ('.wav', '.mp3', '.flac', '.ogg')

    # Process each audio file in input_dir
    for filename in tqdm(os.listdir(args.input_dir)):
        if not filename.lower().endswith(audio_extensions):
            continue

        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        # Extract speech tokens
        audio_tokens = extract_speech_token(whisper_model, feature_extractor, [input_path])[0]
        if len(audio_tokens) == 0:
            print(f"Warning: No audio tokens extracted for {filename}")
            continue

        # Chunk-wise decoding
        block_size_list = [25, 50, 100, 150, 200]
        block_size_idx = 0
        block_size = block_size_list[block_size_idx]
        tts_speechs = []
        tts_mels = []
        prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
        this_uuid = str(uuid.uuid4())
        audio_tokens_processed = 0

        logging.info("start new inference")
        with torch.no_grad():
            while audio_tokens_processed < len(audio_tokens):
                # Get current chunk of tokens
                chunk_tokens = audio_tokens[audio_tokens_processed:audio_tokens_processed + block_size]
                audio_tokens_processed += len(chunk_tokens)
                
                # Update block size for next iteration
                if audio_tokens_processed < len(audio_tokens) and block_size_idx < len(block_size_list) - 1:
                    block_size_idx += 1
                    block_size = block_size_list[block_size_idx]

                # Convert chunk to tensor
                tts_token = torch.tensor(chunk_tokens, device=device).unsqueeze(0)

                # Determine if this is the final chunk
                is_finalize = audio_tokens_processed >= len(audio_tokens)
                logging.info(f"{tts_token.shape = }, {flow_prompt_speech_token.shape = }, {prompt_speech_feat.shape = }")
                # Reconstruct waveform for the chunk
                tts_speech, tts_mel = audio_decoder.token2wav(
                    tts_token, # is_final (1, 43), not final (1, 25)
                    uuid=this_uuid,
                    prompt_token=flow_prompt_speech_token.to(device), # is_final (1, 25) not final (1, 0)
                    prompt_feat=prompt_speech_feat.to(device), # is_final (1, 138, 80) not final (1, 0, 80)
                    finalize=is_finalize
                )

                # Accumulate speech and mel
                tts_speechs.append(tts_speech.squeeze())
                tts_mels.append(tts_mel)

                # Update prompt for next chunk
                flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                if tts_mel is not None:
                    prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)

        # Concatenate all chunks
        tts_speech = torch.cat(tts_speechs, dim=-1).cpu()

        # Save reconstructed waveform
        torchaudio.save(output_path, tts_speech.unsqueeze(0), 22050)
        print(f"Processed {filename} and saved to {output_path}")

if __name__ == "__main__":
    main()