import torch
import torchaudio
import numpy as np
import re
import uuid

from torch import nn
from hyperpyyaml import load_hyperpyyaml
from collections import defaultdict
import onnxruntime


import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")

    print(f"Setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 禁用cudnn的自动优化，可能影响性能
    else:
        torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 影响Python哈希函数的随机性
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1' # TensorFlow相关的CUDNN确定性设置


def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    if fade_in_mel.device == torch.device('cpu'):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
        fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


tts_speech_prev=None
tts_mel_prev=None
class AudioDecoder(nn.Module):
    def __init__(self, config_path, flow_ckpt_path, hift_ckpt_path,campplus_model,mel_cache_len=8, device="cuda"):
        super().__init__()
        self.device = device

        with open(config_path, 'r') as f:
            print(f"{config_path = }")
            self.scratch_configs = load_hyperpyyaml(f)

        # Load models
        self.flow = self.scratch_configs['flow']
        self.flow.load_state_dict(torch.load(flow_ckpt_path, map_location=self.device),strict=False)
        self.hift = self.scratch_configs['hift']
        self.hift.load_state_dict(torch.load(hift_ckpt_path, map_location=self.device))
        self.hift = self.hift.eval()
        self.sample_rate=self.scratch_configs['sample_rate']
        self.feat_extractor=self.scratch_configs['feat_extractor']
        # Move models to the appropriate device
        self.flow.to(self.device)
        self.hift.to(self.device)
        
        # dict used to store session related variable
        self.mel_overlap_dict = defaultdict(lambda: None)
        self.hift_cache_dict = defaultdict(lambda: None)
        self.llm_end_dict = defaultdict(lambda: None)
        self.tts_speech_token_dict = defaultdict(lambda: None)
        self.flow_cache_dict = defaultdict(lambda: None)
        
        self.token_overlap_len = 3.5
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 24000 / 480/2) # 7
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
   
        # self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = mel_cache_len
        # self.mel_cache_len = 4  # mel cache 帧数
        self.source_cache_len = int(self.mel_cache_len * 480) #  24000 / ( 12.5 * 4 ) = 480 
        # speech fade in out
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # self.stream_scale_factor = 1
        # assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'

    def token2wav(self, token, uuid, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192)):
        tts_mel = self.flow.inference(token=token.to(self.device),
                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                      prompt_token=prompt_token.to(self.device),
                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      prompt_feat=prompt_feat.to(self.device),
                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                                          self.device),
                                      embedding=embedding.to(self.device),streaming=False,finalize=True)
        # mel overlap fade in out
        tts_mel=tts_mel[0]
        if self.mel_overlap_dict[uuid] is not None:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # _tts_mel=tts_mel.contiguous()
        # keep overlap mel and hift cache
        
        tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
        del self.hift_cache_dict[uuid]
        del self.mel_overlap_dict[uuid]
        # if uuid in self.hift_cache_dict.keys() and self.hift_cache_dict[uuid] is not None:
        #     tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech, tts_mel

    def offline_inference(self, token):
        this_uuid = str(uuid.uuid1())
        tts_speech, tts_mel = self.token2wav(token, uuid=this_uuid, finalize=True)
        return tts_speech.cpu()

    def token2wav_streaming(self, token, prompt_token, prompt_feat, token_offset, uuid, embedding=torch.zeros(1, 192), finalize=False,stream=False, speed=1.0):
        # with torch.cuda.amp.autocast(self.fp16):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_token=prompt_token.to(self.device),
                                            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_feat=prompt_feat.to(self.device),
                                            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=embedding.to(self.device),
                                            streaming=stream,
                                            finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech
    
    def stream_inference(self, token, prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(0, 192),block_size=8, max_token_len=None):
        token = token.to(self.device)
        # print(f"stream_inference_fix2 token shape: {token.shape}")
        this_uuid = str(uuid.uuid1())
        self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = token.squeeze(0).cpu().tolist(), True
        self.hift_cache_dict[this_uuid] = None
        # self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
        # self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)

        # token_hop_len = self.token_min_hop_len
        # token_hop_len = self.token_hop_len
        token_hop_len = block_size
        # Prepare other necessary input tensors
        embedding = embedding.to(self.device)
        prompt_speech_feat = prompt_feat.to(self.device)
        flow_prompt_speech_token = prompt_token.to(self.device)
        this_tts_speechs = []
        
        
        token_offset = 0
        prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / token_hop_len) * token_hop_len - flow_prompt_speech_token.shape[1])
        while True:
            # time.sleep(0.1)
            this_token_hop_len = token_hop_len + prompt_token_pad if token_offset == 0 else token_hop_len
            if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
                # this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                
                if max_token_len is not None:
                    # 取最新的 max_token_len 长度的 token
                    start_idx = max(0, token_offset + this_token_hop_len + self.flow.pre_lookahead_len - max_token_len)
                    chunk_tokens = self.tts_speech_token_dict[this_uuid][start_idx : token_offset + this_token_hop_len + self.flow.pre_lookahead_len]
                    # 计算实际的 token_offset (相对于截取的 chunk)
                    actual_token_offset = token_offset - start_idx if start_idx > 0 else token_offset
                else:# 不设置max_token_len，默认每次推理将前面所有生成token重推一遍
                    chunk_tokens = self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]
                    actual_token_offset = token_offset

                this_tts_speech_token = torch.tensor(chunk_tokens).unsqueeze(dim=0)
                
                this_tts_speech = self.token2wav_streaming(token=this_tts_speech_token,
                                                    prompt_token=flow_prompt_speech_token,
                                                    prompt_feat=prompt_speech_feat,
                                                    embedding=embedding,
                                                    token_offset=actual_token_offset,
                                                    uuid=this_uuid,
                                                    stream=True,
                                                    finalize=False)
                token_offset += this_token_hop_len
                this_tts_speechs.append(this_tts_speech.cpu())
                # yield {'tts_speech': this_tts_speech.cpu()}
            if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
                break
        # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
        # this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        if max_token_len is not None:
            remaining_tokens = self.tts_speech_token_dict[this_uuid][-max_token_len:]
            start_idx = max(0, len(self.tts_speech_token_dict[this_uuid]) - max_token_len)
            actual_token_offset = token_offset - start_idx
        else:
            remaining_tokens = self.tts_speech_token_dict[this_uuid]
            actual_token_offset = token_offset
            
        this_tts_speech_token = torch.tensor(remaining_tokens).unsqueeze(dim=0)
        this_tts_speech = self.token2wav_streaming(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=embedding,
                                            token_offset=actual_token_offset,
                                            uuid=this_uuid,
                                            finalize=True)
        this_tts_speechs.append(this_tts_speech.cpu())
        # yield {'tts_speech': this_tts_speech.cpu()}
        
        # Convert Mel spectrogram to audio using HiFi-GAN
        this_tts_speech = torch.cat(this_tts_speechs, dim=-1).cpu()

        return this_tts_speech.cpu()
    
    def streaming_inference(self, token,prompt_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_feat=torch.zeros(1, 0, 80), embedding=torch.zeros(1, 192),uuid=None, prev_mel=None,prev_token=None, is_finalize=True):
        token.to(self.device)
        if uuid is None:
            this_uuid = str(uuid.uuid1())
        else:
            this_uuid=uuid
        # Prepare other necessary input tensors
        llm_embedding = embedding.to(self.device)
        prompt_speech_feat = prompt_feat.to(self.device)
        flow_prompt_speech_token = prompt_token.to(self.device)

        block_size = block_size

        tts_token = token

        print(tts_token.size())

        if prev_mel is not None:
            prompt_speech_feat = prev_mel
        if prev_token is not None:
            flow_prompt_speech_token = prev_token
        
        tts_speech, tts_mel = self.token2wav(tts_token, uuid=this_uuid,
                                                prompt_token=flow_prompt_speech_token.to(self.device),
                                                prompt_feat=prompt_speech_feat.to(self.device), finalize=is_finalize)

        prev_mel = torch.cat([prev_mel,tts_mel],dim=1)
        prev_token = torch.cat([prev_token,token],dim=-1)

        # Convert Mel spectrogram to audio using HiFi-GAN
        tts_speech = tts_speech.cpu()

        return tts_speech.cpu(),prev_mel,prev_token

