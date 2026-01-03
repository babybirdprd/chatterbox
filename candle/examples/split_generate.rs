use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

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

    // VoiceEncoder (T3) needs 40 Mel bins
    let mel_40 = candle::audio::compute_mel_spectrogram(
        &ref_samples_16k,
        candle::audio::S3_SR,
        &device,
        40,
    )?;
    // CAMPPlus & S3Tokenizer need 80 Mel bins
    let mel_80 = candle::audio::compute_mel_spectrogram(
        &ref_samples_16k,
        candle::audio::S3_SR,
        &device,
        80,
    )?;

    // STEP 3: S3 Prompt Tokens (Needs 80 mel padded to 128)
    let speech_prompt_tokens = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&s3tokenizer_path], DType::F32, &device)?
        };
        let s3tok = candle::s3tokenizer::S3TokenizerV2::new(
            &candle::s3tokenizer::ModelConfig::default(),
            vb,
        )?;
        let mel_t = mel_80.transpose(1, 2)?;
        let (b, c, t) = mel_t.dims3()?;
        let padding = Tensor::zeros((b, 128 - c, t), DType::F32, &device)?;
        s3tok.encode(&Tensor::cat(&[&mel_t, &padding], 1)?)?
    };

    // STEP 4: T3 Embedding (Needs 40 mel)
    println!("\n[Step 4/6] Extracting speaker embedding with VoiceEncoder...");
    let spk_emb_256 = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&ve_path], DType::F32, &device)? };
        let ve = candle::voice_encoder::VoiceEncoder::new(
            candle::voice_encoder::VoiceEncoderConfig::default(),
            vb,
        )?;
        ve.forward(&mel_40)?
    };

    // STEP 5: S3Gen Embedding (Needs 80 mel)
    println!("[Step 5/6] Extracting synthesis embedding with CAMPPlus...");
    let spk_emb_80 = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], DType::F32, &device)? };
        let campplus = candle::campplus::CAMPPlus::new(80, 192, vb.pp("speaker_encoder"))?;
        campplus.forward(&mel_80)?.narrow(1, 0, 80)?
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
    let speech_tokens_filtered = {
        let tokens = speech_tokens.to_vec2::<u32>()?[0].clone();
        let filtered: Vec<u32> = tokens.into_iter().filter(|&t| t < 6561).collect();
        Tensor::from_vec(filtered.clone(), (1, filtered.len()), &device)?
    };

    let audio_tensor = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], DType::F32, &device)? };
        let s3gen = candle::s3gen::S3Gen::new(vb, true)?;
        s3gen.forward(&speech_tokens_filtered, Some(&spk_emb_80))?
    };

    let samples: Vec<f32> = audio_tensor.flatten_all()?.to_vec1()?;
    candle::audio::save_wav(output, &samples, candle::audio::S3GEN_SR)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("SUCCESS! Audio saved to: {}", output);

    Ok(())
}
