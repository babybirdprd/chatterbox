#!/usr/bin/env python3
"""
dump_intermediates.py - Dumps intermediate tensors at each pipeline phase
for parity testing between Python and Rust implementations.
"""

import argparse
import os
import gc
from pathlib import Path
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2Tokenizer
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.models.s3tokenizer import S3_SR, S3Tokenizer
from chatterbox.models.s3gen import S3GEN_SR, S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.voice_encoder.melspec import melspectrogram
from chatterbox.models.voice_encoder.config import VoiceEncConfig
from chatterbox.models.s3gen.utils.mel import mel_spectrogram as s3gen_mel_spectrogram
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3gen.const import S3GEN_SIL


def save_tensor(tensor, path, name):
    """Save a tensor or numpy array to a .npy file."""
    filepath = path / f"{name}.npy"
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    np.save(filepath, arr)
    print(f"  Saved {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def punc_norm(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("…", ", "), (":", ","), ("—", "-"), ("–", "-"),
        (" ,", ","), ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text


def main():
    parser = argparse.ArgumentParser(description="Dump intermediate tensors for parity testing")
    parser.add_argument("--ref-audio", type=str, required=True, help="Path to reference audio WAV")
    parser.add_argument("--text", type=str, default="Hello world this is a test", help="Text to synthesize")
    parser.add_argument("--output-dir", type=str, default="parity_data", help="Output directory for .npy files")
    parser.add_argument("--ckpt-dir", type=str, default="models", help="Path to model checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    print("\n=== Loading Models ===")
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"Models directory {ckpt_dir} not found. Downloading from HF Hub...")
        ckpt_dir = Path(snapshot_download(
            repo_id="ResembleAI/chatterbox-turbo",
            token=os.getenv("HF_TOKEN") or True,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
        ))
    print(f"Models directory: {ckpt_dir}")

    # --- Phase 1: Audio ---
    print("\n=== Phase 1: Audio Loading & Resampling ===")
    ref_24k, _ = librosa.load(args.ref_audio, sr=S3GEN_SR)
    save_tensor(ref_24k, output_dir, "ref_24k")
    ref_16k = librosa.resample(ref_24k, orig_sr=S3GEN_SR, target_sr=S3_SR)
    save_tensor(ref_16k, output_dir, "ref_16k")

    # --- Phase 2: Mels ---
    print("\n=== Phase 2: Mel Spectrograms ===")
    ve_config = VoiceEncConfig()
    mel_ve = melspectrogram(ref_16k, ve_config)
    save_tensor(mel_ve, output_dir, "mel_ve")

    print("  Computing S3Tokenizer Mel...")
    s3tok_model = S3Tokenizer()
    s3tok_model.to(device).eval()
    ref_16k_tensor = torch.from_numpy(ref_16k).unsqueeze(0).to(device)
    mel_s3tok = s3tok_model.log_mel_spectrogram(ref_16k_tensor)
    save_tensor(mel_s3tok, output_dir, "mel_s3tok")
    
    # Delete to free memory - we're done with S3Tokenizer
    del s3tok_model
    del mel_s3tok
    del ref_16k_tensor
    gc.collect()

    print("  Computing S3Gen Mel...")
    ref_24k_tensor = torch.from_numpy(ref_24k).float()
    mel_s3gen_raw = s3gen_mel_spectrogram(ref_24k_tensor)
    save_tensor(mel_s3gen_raw, output_dir, "mel_s3gen")

    # --- Phase 3: VoiceEncoder ---
    print("\n=== Phase 3: VoiceEncoder ===")
    ve = VoiceEncoder()
    ve_path = ckpt_dir / "ve.safetensors"
    print(f"  Loading {ve_path}...")
    ve.load_state_dict(load_file(ve_path))
    ve.to(device).eval()
    with torch.inference_mode():
        spk_emb_ve = torch.from_numpy(ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR))
        spk_emb_ve = spk_emb_ve.mean(axis=0, keepdim=True)
    save_tensor(spk_emb_ve, output_dir, "spk_emb_ve")
    del ve
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Phase 4: S3Gen & CAMPPlus ---
    print("\n=== Phase 4: S3Gen & CAMPPlus ===")
    s3gen = S3Gen(meanflow=True)
    s3gen_path = ckpt_dir / "s3gen_meanflow.safetensors"
    print(f"  Loading {s3gen_path}...")
    s3gen.load_state_dict(load_file(s3gen_path), strict=True)
    s3gen.to(device).eval()
    
    DEC_COND_LEN = 10 * S3GEN_SR
    ref_24k_cond = ref_24k[:DEC_COND_LEN]
    with torch.inference_mode():
        print("  Extracting reference dict...")
        ref_dict = s3gen.embed_ref(ref_24k_cond, S3GEN_SR, device=device)
    
    save_tensor(ref_dict["embedding"], output_dir, "spk_emb_camp_full")
    save_tensor(ref_dict["embedding"][:, :80], output_dir, "spk_emb_camp")
    save_tensor(ref_dict["prompt_token"], output_dir, "prompt_tokens")
    save_tensor(ref_dict["prompt_feat"], output_dir, "prompt_feat")

    # Also dump CAMPPlus Mel input for parity debugging
    from src.chatterbox.models.s3gen.xvector import extract_feature
    ref_16k_torch = torch.from_numpy(ref_16k).float().to(device)
    with torch.inference_mode():
        mel_camp, _, _ = extract_feature(ref_16k_torch.unsqueeze(0))
    # mel_camp shape is [1, T, 80]
    save_tensor(mel_camp.transpose(1, 2).contiguous(), output_dir, "mel_camp")

    # PRE-CALCULATE T3 Conditioning while s3gen is still here
    print("  Pre-calculating T3 conditioning...")
    ENC_COND_LEN = 15 * S3_SR
    ref_16k_cond = ref_16k[:ENC_COND_LEN]
    # We need t3_config params here, let's hardcode or assume defaults
    t3_cond_prompt_tokens, _ = s3gen.tokenizer([ref_16k_cond], max_len=375)
    t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).cpu()
    save_tensor(t3_cond_prompt_tokens, output_dir, "t3_cond_prompt_tokens")

    # Store ref_dict for later but clear model memory
    ref_dict_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in ref_dict.items()}
    spk_emb_ve_cpu = spk_emb_ve.cpu()
    del s3gen
    del ref_dict
    del spk_emb_ve
    del ref_16k_torch
    del mel_camp
    gc.collect()
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Phase 5: Text Tokenization ---
    print("\n=== Phase 5: Text Tokenization ===")
    try:
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        print(f"  AutoTokenizer failed: {e}. Trying GPT2Tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(ckpt_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    text = punc_norm(args.text)
    print(f"  Normalized text: {text}")
    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_token_ids = text_tokens.input_ids.to(device)
    save_tensor(text_token_ids, output_dir, "text_tokens")

    # --- Phase 6: T3 Generation ---
    print("\n=== Phase 6: T3 Generation ===")
    try:
        print("  Initializing T3...")
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False
        
        # Use torch's meta device to create an empty model shell (no memory)
        # Then load weights directly
        print("  Creating T3 model (this may take a moment)...")
        t3 = T3(hp)
        t3_path = ckpt_dir / "t3_turbo_v1.safetensors"
        print(f"  Loading weights from {t3_path}...")
        
        # Direct load_file is more memory efficient than safe_open + dict comprehension
        t3_state = load_file(t3_path, device="cpu")
        
        print(f"  Loaded {len(t3_state)} tensors, applying state dict...")
        missing, unexpected = t3.load_state_dict(t3_state, strict=False)
        print(f"  State dict loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        del t3_state 
        gc.collect()
        
        if hasattr(t3.tfmr, 'wte'):
            print("  Deleting tfmr.wte to save memory...")
            del t3.tfmr.wte
            
        t3.to(device).eval()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        print("  Preparing T3 conditioning...")
        with torch.inference_mode():
            print("  Creating T3Cond...")
            t3_cond = T3Cond(
                speaker_emb=spk_emb_ve_cpu.to(device),
                cond_prompt_speech_tokens=t3_cond_prompt_tokens.to(device),
                emotion_adv=0.0 * torch.ones(1, 1, 1).to(device),
            ).to(device=device)
            
            print("  Running T3 inference_turbo...")
            torch.manual_seed(args.seed)
            speech_tokens = t3.inference_turbo(
                t3_cond=t3_cond,
                text_tokens=text_token_ids,
                temperature=0.8,
                top_k=1000,
                top_p=0.95,
                repetition_penalty=1.2,
            )
        
        print(f"  Generated {speech_tokens.size(1)} speech tokens.")
        del t3
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        speech_tokens = speech_tokens[speech_tokens < 6561]
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(device)
        speech_tokens = torch.cat([speech_tokens, silence])
        save_tensor(speech_tokens, output_dir, "gen_tokens")
        
        # --- Phase 7: S3Gen Flow ---
        print("\n=== Phase 7: S3Gen Flow (Token -> Mel) ===")
        # Re-initialize S3Gen
        s3gen = S3Gen(meanflow=True)
        s3gen_path = ckpt_dir / "s3gen_meanflow.safetensors"
        s3gen.load_state_dict(load_file(s3gen_path), strict=True)
        s3gen.to(device).eval()
        
        ref_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in ref_dict_cpu.items()}
        with torch.inference_mode():
            print("  Running flow_inference...")
            output_mels = s3gen.flow_inference(
                speech_tokens=speech_tokens,
                ref_dict=ref_dict,
                n_cfm_timesteps=2,
                finalize=True,
            )
        save_tensor(output_mels, output_dir, "flow_output_mel")
        
        # --- Phase 8: HiFiGAN ---
        print("\n=== Phase 8: HiFiGAN Vocoder ===")
        with torch.inference_mode():
            print("  Running hift_inference...")
            output_wavs, output_sources = s3gen.hift_inference(output_mels, None)
            
            print("  Applying trim_fade...")
            n_trim = S3GEN_SR // 50
            trim_fade = torch.zeros(2 * n_trim, device=device)
            trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim, device=device)) + 1) / 2
            output_wavs[:, :len(trim_fade)] *= trim_fade
        
        save_tensor(output_wavs, output_dir, "final_audio")
        save_tensor(output_sources, output_dir, "f0_source")
        
        import scipy.io.wavfile as wav
        final_audio_np = output_wavs.squeeze(0).cpu().numpy()
        wav_path = output_dir / "output_python.wav"
        wav.write(str(wav_path), S3GEN_SR, (final_audio_np * 32767).astype(np.int16))
        print(f"  Saved {wav_path}")

    except Exception as e:
        print(f"\nERROR in Phase 6-8: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Summary ===")
    for f in sorted(output_dir.glob("*.npy")):
        try:
            arr = np.load(f)
            print(f"  {f.name}: shape={arr.shape}")
        except Exception:
            print(f"  {f.name}: error loading")


if __name__ == "__main__":
    main()
