import logging
import torch
import sys

try:
    from speechtokenizer.model import SpeechTokenizer
    from speechtokenizer.model_for_semantic_evaluation_pre_quantizer_layers_sum import SpeechTokenizer as SpeechTokenizer_For_Semantic_Evaluation_Pre_Quantizer_Layers_Sum 
    from speechtokenizer.hf_modules.hf_pretrained_model import Hf_pretrained_model
except Exception as e:
    print(f"Import spt model failed")

def load_and_fix_speechtokenizer(config_path, ckpt_path, device=torch.device("cuda")):
    speechtokenizer = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    speechtokenizer = speechtokenizer.to(device)
    speechtokenizer.eval()
    
    for param in speechtokenizer.parameters():
        param.requires_grad = False
    
    logging.info(f"Load and fix speechtokenizer of config: {config_path} from checkpoint: {ckpt_path} success")
    
    return speechtokenizer

def load_and_fix_speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum(config_path, ckpt_path, device=torch.device("cuda")):
    speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum = SpeechTokenizer_For_Semantic_Evaluation_Pre_Quantizer_Layers_Sum.load_from_checkpoint(config_path, ckpt_path)
    speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum = speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum.to(device)
    speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum.eval()
    
    for param in speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum.parameters():
        param.requires_grad = False
    
    logging.info(f"Load and fix speechtokenizer of config: {config_path} from checkpoint: {ckpt_path} success")
    
    return speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum
    
def load_and_fix_hf_pretrained_models(config_path, device=torch.device("cuda")):
    hf_model = Hf_pretrained_model.load_from_checkpoint(config_path)
    hf_model = hf_model.to(device)
    hf_model.eval()
    
    for param in hf_model.parameters():
        param.requires_grad = False
    
    logging.info(f"Load and fix hf_model of config: {config_path} success")
    
    return hf_model

def load_and_fix_codec_model(args):
    """
        支持 Load 不同 codec model
        目前只支持 Load SpeechTokenizer, 
        如果需要支持其他，model 在这边实现即可，注意满足 CodecForCodec 中的需求
    """
    if args.model_type == "SpeechTokenizer":
        codec_model = load_and_fix_speechtokenizer(args.config, args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == "SpeechTokenizer_For_Semantic_Evaluation_Pre_Quantizer_Layers_Sum":
        codec_model = load_and_fix_speechtokenizer_for_semantic_evaluation_pre_quantizer_layers_sum(args.config, args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == "x-codec2":
        sys.path.append("/remote-home1/ytgong/X-Codec-2.0")
        from model_x_codec import X_Codec2
        codec_model = X_Codec2(semantic_position=args.config, ckpt_path=args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == 'huggingface_model':
        codec_model = load_and_fix_hf_pretrained_models(config_path=args.config)
        target_frame_rate_before_ctc = codec_model.sampling_rate // codec_model.downsample_rate
    elif args.model_type == 'BigCodec':
        sys.path.append("/remote-home1/ytgong/BigCodec")
        from model_bigcodec import BigCodec
        codec_model = BigCodec(ckpt_path=args.codec_ckpt)
        target_frame_rate_before_ctc = 80
    elif args.model_type == 'focalcodec':
        sys.path.append("/remote-home1/ytgong/focalcodec")
        import focalcodec
        codec_model = focalcodec.FocalCodec.from_pretrained()
        target_frame_rate_before_ctc = 50
    elif args.model_type == 'Data2VecPytorch':
        sys.path.append("/remote-home1/ytgong/data2vec-pytorch")
        from model_data2vec_pytorch import Data2VecPytorch
        codec_model = Data2VecPytorch(config=args.config, ckpt_path=args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == 'SpeechTokenizer1_release':
        sys.path.append("/remote-home1/ytgong/SpeechTokenizer")
        from speechtokenizer_release.model import SpeechTokenizer as SpeechTokenizer1_Release
        codec_model = SpeechTokenizer1_Release.load_from_checkpoint(config_path=args.config, ckpt_path=args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    elif args.model_type == 'StableCodec':
        sys.path.append("/remote-home1/ytgong/stable-codec")
        from modeling_stable_codec import Modeling_StableCodec
        codec_model = Modeling_StableCodec(model_config_path=args.config, ckpt_path=args.codec_ckpt)
        target_frame_rate_before_ctc = 50
    else:
        assert False, f'model type {args.model_type} not support !'
    
    for param in codec_model.parameters():
        param.requires_grad = False
    
    assert target_frame_rate_before_ctc >= 50
    
    return codec_model, target_frame_rate_before_ctc