// candle/examples/split_generate.rs
use candle::audio::{AudioProcessor, MelConfig};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    // ... (args parsing same as before) ...
    let args: Vec<String> = std::env::args().collect();
    let text = args
        .iter()
        .position(|a| a == "--text")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("Hello world");
    let ref_audio = args
        .iter()
        .position(|a| a == "--ref-audio")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("reference.wav");
    let output = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("output.wav");
    let use_fp16 = args.iter().any(|a| a == "--fp16");
    let use_cuda = args.iter().any(|a| a == "--cuda");
    let device = if use_cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let dtype = if use_fp16 { DType::F16 } else { DType::F32 };

    println!("Downloading models...");
    let api = Api::new()?;
    let turbo_repo = api.model("ResembleAI/chatterbox-turbo".to_string());
    let t3_path = turbo_repo.get("t3_turbo_v1.safetensors")?;
    let s3gen_path = turbo_repo.get("s3gen_meanflow.safetensors")?;
    let ve_path = turbo_repo.get("ve.safetensors")?;
    let s3tokenizer_path = api
        .model("ResembleAI/s3tokenizer-v2".to_string())
        .get("model.safetensors")?;
    let tokenizer_path = api
        .model("ResembleAI/chatterbox".to_string())
        .get("tokenizer.json")?;

    // 1. Load Audio
    let (ref_samples, ref_sr) =
        candle::audio::load_wav(ref_audio).map_err(|e| anyhow::anyhow!(e))?;

    // 2. Resample
    let ref_16k =
        candle::audio::resample(&ref_samples, ref_sr, 16000).map_err(|e| anyhow::anyhow!(e))?;
    let ref_24k =
        candle::audio::resample(&ref_samples, ref_sr, 24000).map_err(|e| anyhow::anyhow!(e))?;

    // 3. Compute Mels using Specific Configs

    // A. Voice Encoder: 16k, 40 mels, Power, Linear
    let cfg_ve = MelConfig::for_voice_encoder();
    let mel_ve = AudioProcessor::compute_mel_spectrogram(&ref_16k, &device, &cfg_ve)?;
    // VoiceEncoder expects (B, T, 40)
    let mel_ve_t = mel_ve.transpose(1, 2)?.to_dtype(dtype)?;

    // B. S3Tokenizer: 16k, 128 mels, Power, Log10
    let cfg_s3tok = MelConfig::for_s3tokenizer();
    let mel_s3tok = AudioProcessor::compute_mel_spectrogram(&ref_16k, &device, &cfg_s3tok)?;
    let mel_s3tok_log = AudioProcessor::log_process(&mel_s3tok, &cfg_s3tok)?;

    // Post-process for S3Tokenizer: max normalization and scaling
    // Python: log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    let max_val = mel_s3tok_log.max_all()?.to_scalar::<f32>()?;
    let mel_s3tok_norm = mel_s3tok_log.maximum(max_val - 8.0)?;
    let mel_s3tok_final = ((mel_s3tok_norm + 4.0)? / 4.0)?.to_dtype(dtype)?;

    // C. S3Gen/CAMPPlus: 24k, 80 mels, Magnitude, Ln
    // NOTE: CAMPPlus in Python uses 16k audio but S3Gen uses 24k.
    // However, the provided `s3gen_meanflow.safetensors` contains `speaker_encoder` (CAMPPlus).
    // In `s3gen.py`, `self.speaker_encoder.inference(ref_wav_16)` is called.
    // So CAMPPlus needs 16k input.
    // But S3Gen Flow/Decoder needs 24k Mel.

    // C1. S3Gen Conditioning (24k)
    let cfg_s3gen = MelConfig::for_s3gen();
    let mel_s3gen = AudioProcessor::compute_mel_spectrogram(&ref_24k, &device, &cfg_s3gen)?;
    let mel_s3gen_log = AudioProcessor::log_process(&mel_s3gen, &cfg_s3gen)?.to_dtype(dtype)?;

    // C2. CAMPPlus (16k)
    // Using Kaldi-like config via MelConfig::for_campplus()
    let cfg_camp = MelConfig::for_campplus();
    let mel_camp = AudioProcessor::compute_mel_spectrogram(&ref_16k, &device, &cfg_camp)?;
    // CAMPPlus expects Log (Natural)
    let mel_camp_log = mel_camp.clamp(1e-5, f32::MAX)?.log()?;
    let mean = mel_camp_log.mean_keepdim(2)?;
    let mel_camp_norm = mel_camp_log.broadcast_sub(&mean)?.to_dtype(dtype)?;

    // 4. Run Models

    // Voice Encoder
    println!("Running VoiceEncoder...");
    let vb_ve = unsafe { VarBuilder::from_mmaped_safetensors(&[&ve_path], dtype, &device)? };
    let ve = candle::voice_encoder::VoiceEncoder::new(
        candle::voice_encoder::VoiceEncoderConfig::default(),
        vb_ve,
    )?;
    let spk_emb_ve = ve.forward(&mel_ve_t)?;

    // S3Tokenizer
    println!("Running S3Tokenizer...");
    let vb_tok =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&s3tokenizer_path], dtype, &device)? };
    let s3tok = candle::s3tokenizer::S3TokenizerV2::new(
        &candle::s3tokenizer::ModelConfig::default(),
        vb_tok,
    )?;
    let prompt_tokens = s3tok.encode(&mel_s3tok_final)?;

    // CAMPPlus
    println!("Running CAMPPlus...");
    let vb_s3 = unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], dtype, &device)? };
    let campplus = candle::campplus::CAMPPlus::new(80, 192, vb_s3.pp("speaker_encoder"))?;
    let spk_emb_camp = campplus.forward(&mel_camp_norm)?.narrow(1, 0, 80)?;

    // T3 Generation
    println!("Running T3...");
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
    let text_ids = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?
        .get_ids()
        .to_vec();
    let text_tensor = Tensor::from_vec(
        text_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
        (1, text_ids.len()),
        &device,
    )?;

    let vb_t3 = unsafe { VarBuilder::from_mmaped_safetensors(&[&t3_path], dtype, &device)? };
    let t3_config = candle::t3_model::T3Config::default(); // Uses defaults from lib
    let t3 = candle::t3_model::T3::new(t3_config, vb_t3)?;

    let generated_tokens = t3.generate(
        &text_tensor,
        &spk_emb_ve,
        Some(&prompt_tokens),
        None,
        500,
        0.8,
        0.95,
        50,
        1.2,
        42,
    )?;

    // S3Gen Synthesis
    println!("Running S3Gen...");
    let s3gen = candle::s3gen::S3Gen::new(vb_s3, true)?;

    // Filter tokens
    let gen_vec = generated_tokens.to_vec2::<u32>()?[0].clone();
    let valid_gen: Vec<u32> = gen_vec.into_iter().filter(|&t| t < 6561).collect();
    let valid_gen_tensor = Tensor::from_vec(valid_gen.clone(), (1, valid_gen.len()), &device)?;

    let input_tokens = Tensor::cat(&[&prompt_tokens, &valid_gen_tensor], 1)?;

    // Prepare Conditioning (Prompt Mel + Zero Pad)
    // Target length is 2x tokens
    let target_len = input_tokens.dim(1)? * 2;
    let (b, c, prompt_len) = mel_s3gen_log.dims3()?;
    let cond = if prompt_len < target_len {
        let pad = Tensor::zeros((b, c, target_len - prompt_len), dtype, &device)?;
        Tensor::cat(&[&mel_s3gen_log, &pad], 2)?
    } else {
        mel_s3gen_log.narrow(2, 0, target_len)?
    };

    let audio = s3gen.forward(&input_tokens, Some(&spk_emb_camp), Some(&cond))?;

    let audio_vec = audio.flatten_all()?.to_vec1::<f32>()?;
    candle::audio::save_wav(output, &audio_vec, 24000).map_err(|e| anyhow::anyhow!(e))?;

    println!("Done!");
    Ok(())
}
