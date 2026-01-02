//! Example: Split model loading to reduce peak memory usage
//!
//! This example loads and processes each model step sequentially,
//! dropping intermediate results to minimize memory footprint.
//!
//! Usage:
//! ```bash
//! cargo run --release --example split_generate -- --text "Hello world" --ref-audio reference.wav
//! ```

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use std::path::Path;
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();

    let text = args
        .iter()
        .position(|a| a == "--text")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("Hello, this is a test of the Chatterbox text to speech system.");

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

    let device = if use_cuda {
        println!("Using CUDA...");
        Device::new_cuda(0)?
    } else {
        println!("Using CPU...");
        Device::Cpu
    };

    // Check reference audio exists
    if !Path::new(ref_audio).exists() {
        anyhow::bail!("Reference audio not found: {}", ref_audio);
    }

    // Download all model files first (cached by hf_hub)
    println!("Downloading model files...");
    let api = Api::new().map_err(|e| anyhow::anyhow!("{e}"))?;

    let turbo_repo = api.model("ResembleAI/chatterbox-turbo".to_string());
    let t3_path = turbo_repo
        .get("t3_turbo_v1.safetensors")
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let s3gen_path = turbo_repo
        .get("s3gen_meanflow.safetensors")
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let s3tok_repo = api.model("ResembleAI/s3tokenizer-v2".to_string());
    let s3tokenizer_path = s3tok_repo
        .get("model.safetensors")
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let base_repo = api.model("ResembleAI/chatterbox".to_string());
    let tokenizer_path = base_repo
        .get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("Model files ready.\n");

    // ========== STEP 1: Tokenize text ==========
    println!("[Step 1/6] Tokenizing text: \"{}\"", text);
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let text_tokens: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let text_tokens_tensor =
        Tensor::from_vec(text_tokens.clone(), (1, text_tokens.len()), &device)?;
    println!("  → {} text tokens", text_tokens.len());
    drop(tokenizer);

    // ========== STEP 2: Process reference audio ==========
    println!("\n[Step 2/6] Processing reference audio...");
    let (ref_samples, ref_sr) =
        candle::audio::load_wav(ref_audio).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("  → {} samples @ {}Hz", ref_samples.len(), ref_sr);

    // Resample to 16kHz for S3Tokenizer and CAMPPlus
    let ref_samples_16k = if ref_sr != candle::audio::S3_SR {
        candle::audio::resample(&ref_samples, ref_sr, candle::audio::S3_SR)
            .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        ref_samples.clone()
    };
    drop(ref_samples);

    // Compute mel spectrogram (80 mels for CAMPPlus)
    let mel_80 =
        candle::audio::compute_mel_spectrogram(&ref_samples_16k, candle::audio::S3_SR, &device)?;
    println!("  → Mel spectrogram: {:?}", mel_80.shape());

    // ========== STEP 3: Get speech prompt tokens (S3Tokenizer) ==========
    println!("\n[Step 3/6] Encoding reference audio with S3Tokenizer...");
    let speech_prompt_tokens = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&s3tokenizer_path], DType::F32, &device)?
        };
        let config = candle::s3tokenizer::ModelConfig::default();
        let s3tok = candle::s3tokenizer::S3TokenizerV2::new(&config, vb)?;

        // S3Tokenizer expects (B, 128, T) - pad 80 mels to 128
        let mel_t = mel_80.transpose(1, 2)?; // (1, 80, T)
        let (b, c, t) = mel_t.dims3()?;
        let padding = Tensor::zeros((b, 128 - c, t), DType::F32, &device)?;
        let mel_128 = Tensor::cat(&[&mel_t, &padding], 1)?;

        let tokens = s3tok.encode(&mel_128)?;
        println!("  → {} speech prompt tokens", tokens.dim(1)?);
        tokens
    };

    // ========== STEP 4: Get speaker embedding (CAMPPlus) ==========
    println!("\n[Step 4/6] Extracting speaker embedding with CAMPPlus...");
    let speaker_embedding = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], DType::F32, &device)? };
        let campplus = candle::campplus::CAMPPlus::new(80, 192, vb.pp("speaker_encoder"))?;
        let spk_emb = campplus.forward(&mel_80)?;
        println!("  → Speaker embedding: {:?}", spk_emb.shape());
        spk_emb
    };
    drop(ref_samples_16k);

    // Project speaker embedding to 80 dims for S3Gen
    let spk_emb_80 = speaker_embedding.narrow(1, 0, 80)?;
    println!("  → Projected to {:?}", spk_emb_80.shape());

    // ========== STEP 5: Generate speech tokens (T3) ==========
    println!("\n[Step 5/6] Generating speech tokens with T3...");
    let speech_tokens = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&t3_path], DType::F32, &device)? };

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
        println!("  → T3 model loaded, generating...");

        // Generate speech tokens
        let tokens = t3.generate(
            &text_tokens_tensor,
            &speaker_embedding, // Use full 192-dim embedding
            Some(&speech_prompt_tokens),
            None, // emotion_adv
            500,  // max_gen_len
            0.8,  // temperature
            0.95, // top_p
            50,   // top_k
            1.0,  // repetition_penalty
            42,   // seed
        )?;

        println!("  → Generated {} speech tokens", tokens.dim(1)?);
        tokens
    };

    // Filter valid tokens (< 6561)
    let speech_tokens_filtered = {
        let tokens = speech_tokens.to_vec2::<i64>()?[0].clone();
        let filtered: Vec<i64> = tokens.into_iter().filter(|&t| t >= 0 && t < 6561).collect();
        println!("  → {} valid tokens after filtering", filtered.len());
        let len = filtered.len();
        if len == 0 {
            anyhow::bail!("No valid speech tokens generated!");
        }
        Tensor::from_vec(filtered, (1, len), &device)?
    };

    // ========== STEP 6: Generate audio (S3Gen) ==========
    println!("\n[Step 6/6] Synthesizing audio with S3Gen...");
    let audio_tensor = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], DType::F32, &device)? };
        let s3gen = candle::s3gen::S3Gen::new(vb, true)?;
        println!("  → S3Gen loaded, running synthesis...");

        let audio = s3gen.forward(&speech_tokens_filtered, Some(&spk_emb_80))?;
        println!("  → Audio tensor: {:?}", audio.shape());
        audio
    };

    // Convert to samples and save
    let samples: Vec<f32> = audio_tensor.flatten_all()?.to_vec1()?;
    println!("\n=== Saving audio ===");
    println!(
        "  → {} samples @ {}Hz",
        samples.len(),
        candle::audio::S3GEN_SR
    );

    candle::audio::save_wav(output, &samples, candle::audio::S3GEN_SR)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    println!("  → Saved to: {}", output);
    println!("\n🎉 SUCCESS! Listen to your generated audio: {}", output);

    Ok(())
}
