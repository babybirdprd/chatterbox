use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

fn process_mel_for_s3tokenizer(mel: &Tensor) -> anyhow::Result<Tensor> {
    // Python: log10(clamp(min=1e-10)) -> max(x, x.max() - 8.0) -> (x + 4.0) / 4.0
    let mel = mel.clamp(1e-10, f32::MAX)?;
    let log_mel = ((mel.log()? / 10.0f64.ln())? * 2.0)?; // ln(x) / ln(10) = log10(x) * 2.0 for Power Spectrogram

    // Dynamic range compression
    let max_val = log_mel.max_all()?.to_scalar::<f32>()?;
    let log_mel = log_mel.maximum(max_val - 8.0)?;

    let result = ((log_mel + 4.0)? / 4.0)?;
    Ok(result)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let text = args
        .iter()
        .position(|a| a == "--text")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("Hello, world.");
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
    let use_cuda = args.iter().any(|a| a == "--cuda");
    let use_t3_cuda = args.iter().any(|a| a == "--t3-cuda");
    let use_fp16 = args.iter().any(|a| a == "--fp16");
    let device = if use_cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let t3_device = if use_t3_cuda {
        Device::new_cuda(0)?
    } else {
        device.clone()
    };
    let dtype = if use_fp16 { DType::F16 } else { DType::F32 };

    println!("Downloading model files...");
    let api = Api::new().map_err(|e| anyhow::anyhow!("{e}"))?;
    let turbo_repo = api.model("ResembleAI/chatterbox-turbo".to_string());
    let t3_path = turbo_repo.get("t3_turbo_v1.safetensors")?;
    println!("[DEBUG] T3 Path: {:?}", t3_path);
    {
        let file = std::fs::File::open(&t3_path).unwrap();
        let mem = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
        let safetensors = safetensors::SafeTensors::deserialize(&mem).unwrap();
        let mut names: Vec<_> = safetensors.names().into_iter().collect();
        names.sort();
        println!("[DEBUG] Keys in t3_turbo_v1.safetensors:");
        for name in names {
            println!("  {}", name);
        }
    }
    let s3gen_path = turbo_repo.get("s3gen_meanflow.safetensors")?;
    let ve_path = turbo_repo.get("ve.safetensors")?;
    let s3tokenizer_path = api
        .model("ResembleAI/s3tokenizer-v2".to_string())
        .get("model.safetensors")?;
    let tokenizer_path = api
        .model("ResembleAI/chatterbox".to_string())
        .get("tokenizer.json")?;

    // STEP 1: Text
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let text_tokens: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let text_tokens_tensor =
        Tensor::from_vec(text_tokens.clone(), (1, text_tokens.len()), &device)?;

    // STEP 2: Reference Audio Processing
    let (ref_samples, ref_sr) =
        candle::audio::load_wav(ref_audio).map_err(|e| anyhow::anyhow!("{e}"))?;
    let ref_samples_16k = if ref_sr != candle::audio::S3_SR {
        candle::audio::resample(&ref_samples, ref_sr, candle::audio::S3_SR)
            .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        ref_samples.clone()
    };

    let ref_samples_24k = if ref_sr != candle::audio::S3GEN_SR {
        candle::audio::resample(&ref_samples, ref_sr, candle::audio::S3GEN_SR)
            .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        ref_samples.clone()
    };

    // NOTE: compute_mel_spectrogram now returns LINEAR MAGNITUDE (unlogged)

    // 1. Config for VoiceEncoder (16kHz, 40 mels, n_fft=400)
    let config_ve = candle::audio::MelConfig {
        n_fft: 400,
        hop_length: 160,
        win_length: 400,
        n_mels: 40,
        fmax: 8000.0,
    };

    // 2. Config for S3Tokenizer (16kHz, 128 mels, n_fft=400)
    let config_s3tok = candle::audio::MelConfig {
        n_fft: 400,
        hop_length: 160,
        win_length: 400,
        n_mels: 128,
        fmax: 8000.0,
    };

    let mel_40_linear = candle::audio::compute_mel_spectrogram(
        &ref_samples_16k,
        candle::audio::S3_SR,
        &device,
        &config_ve,
    )?;

    // Used for S3Tokenizer (no longer using the default 16k config which was 80 mels)
    let mel_128_linear = candle::audio::compute_mel_spectrogram(
        &ref_samples_16k,
        candle::audio::S3_SR,
        &device,
        &config_s3tok,
    )?;

    // S3Gen/CAMPPlus (24kHz, 80 mels, n_fft=1920) - This was already correct
    let mel_80_24k_linear = candle::audio::compute_mel_spectrogram(
        &ref_samples_24k,
        candle::audio::S3GEN_SR,
        &device,
        &candle::audio::MelConfig::for_24k(80),
    )?;

    // --- STEP 3: S3 Prompt Tokens ---
    // Use mel_128_linear directly (no padding needed anymore)
    let mel_for_tokenizer = process_mel_for_s3tokenizer(&mel_128_linear)?;
    let speech_prompt_tokens = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&s3tokenizer_path], DType::F32, &device)?
        };
        let s3tok = candle::s3tokenizer::S3TokenizerV2::new(
            &candle::s3tokenizer::ModelConfig::default(),
            vb,
        )?;

        // Input is already (B, 128, T), so just encode directly
        s3tok.encode(&mel_for_tokenizer)?
    };

    // --- STEP 4: T3 Embedding (Voice Encoder) ---
    println!("\n[Step 4/6] Extracting speaker embedding with VoiceEncoder...");
    let spk_emb_256 = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&ve_path], DType::F32, &device)? };
        let ve = candle::voice_encoder::VoiceEncoder::new(
            candle::voice_encoder::VoiceEncoderConfig::default(),
            vb,
        )?;
        // VoiceEncoder expects Power Mel Spectrogram (amp^2)
        let mel_40_power = mel_40_linear.sqr()?;
        // VoiceEncoder expects (B, T, 40)
        ve.forward(&mel_40_power.transpose(1, 2)?)?
    };

    // --- STEP 5: S3Gen Embedding (CAMPPlus) ---
    println!("[Step 5/6] Extracting synthesis embedding with CAMPPlus...");
    // CAMPPlus expects Mean-Normalized Log-Mel
    let mel_80_log = mel_80_24k_linear.clamp(1e-5, f32::MAX)?.log()?;
    let mean = mel_80_log.mean_keepdim(2)?;
    let mel_for_campplus = mel_80_log.broadcast_sub(&mean)?;

    let spk_emb_80 = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], DType::F32, &device)? };
        let campplus = candle::campplus::CAMPPlus::new(80, 192, vb.pp("speaker_encoder"))?;
        campplus.forward(&mel_for_campplus)?.narrow(1, 0, 80)?
    };

    // STEP 6: T3 Generation
    println!("\n[Step 6/6] Generating tokens with T3...");
    let speech_tokens = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&t3_path], dtype, &t3_device)? };
        let t3_config = candle::t3_model::T3Config {
            text_tokens_dict_size: 50276,
            speech_tokens_dict_size: 6563,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            vocab_size: 50276,
            speaker_embed_size: 256,
            start_speech_token: 6561,
            stop_speech_token: 6562,
            speech_cond_prompt_len: Some(375),
            use_perceiver_resampler: false,
            emotion_adv: false,
            n_positions: 8196,
        };
        let t3 = candle::t3_model::T3::new(t3_config, vb)?;
        t3.generate(
            &text_tokens_tensor.to_device(&t3_device)?,
            &spk_emb_256.to_device(&t3_device)?.to_dtype(dtype)?,
            Some(&speech_prompt_tokens.to_device(&t3_device)?),
            None,
            500,
            0.8,
            0.95,
            50,
            1.2,
            42,
        )?
    };

    // STEP 7: Synthesis
    println!("\n[Step 7/6] Synthesizing audio...");

    // 1. Prepare the full sequence of tokens: [Prompt Tokens, Generated Tokens]
    let speech_tokens_filtered = {
        let tokens = speech_tokens.to_vec2::<u32>()?[0].clone();
        let filtered: Vec<u32> = tokens
            .into_iter()
            .filter(|&t| t != 6561 && t != 6562)
            .collect();
        Tensor::from_vec(filtered.clone(), (1, filtered.len()), &device)?
    };

    // Concatenate prompt tokens + generated tokens
    // Note: speech_prompt_tokens was calculated in Step 3.
    // We assume speech_prompt_tokens is (1, N_PROMPT).
    let s3_input_tokens = Tensor::cat(&[&speech_prompt_tokens, &speech_tokens_filtered], 1)?;
    println!("  S3Gen Input Tokens Shape: {:?}", s3_input_tokens.dims());

    let audio_tensor = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], DType::F32, &device)? };
        let s3gen = candle::s3gen::S3Gen::new(vb, true)?;

        // 2. Prepare Reference Mel for Conditioning
        // S3Gen expects Natural Log of Linear Magnitude Mel (not Power, not Log10)
        // mel_80_24k_linear is (1, 80, T_ref)
        let ref_mel_log = mel_80_24k_linear.clamp(1e-5, f32::MAX)?.log()?;

        // 3. Construct the Conditioning Tensor [Ref Mel | Zeros]
        // The S3Gen Encoder upsamples tokens by 2x.
        // So output length = input_tokens_len * 2.
        let (_, n_toks) = s3_input_tokens.dims2()?;
        let target_len = n_toks * 2;
        let (b, c, ref_len) = ref_mel_log.dims3()?;

        // Ensure we don't overflow if ref is somehow longer than target (unlikely if logic is correct)
        let safe_ref_len = ref_len.min(target_len);
        let ref_mel_log_trimmed = ref_mel_log.narrow(2, 0, safe_ref_len)?;

        // Create padding
        let padding_len = target_len - safe_ref_len;
        let cond = if padding_len > 0 {
            let padding = Tensor::zeros((b, c, padding_len), DType::F32, &device)?;
            Tensor::cat(&[&ref_mel_log_trimmed, &padding], 2)?
        } else {
            ref_mel_log_trimmed
        };

        println!("  Conditioning Tensor Shape: {:?}", cond.dims());

        // 4. Forward with corrected inputs
        s3gen.forward(
            &s3_input_tokens,
            Some(&spk_emb_80),
            Some(&cond), // Pass the time-aligned conditioning, not the mean!
        )?
    };

    let samples: Vec<f32> = audio_tensor.flatten_all()?.to_vec1()?;
    candle::audio::save_wav(output, &samples, candle::audio::S3GEN_SR)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("SUCCESS! Audio saved to: {}", output);

    Ok(())
}
