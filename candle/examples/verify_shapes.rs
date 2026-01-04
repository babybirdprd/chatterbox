use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

/// Comprehensive shape verification test for all pipeline components
/// Tests each component's forward pass with dummy tensors to catch shape mismatches
/// before running full inference.
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model_dir = "C:/Users/Steve Business/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd";

    println!("=== COMPREHENSIVE SHAPE VERIFICATION TEST ===\n");

    // Sample dimensions
    let batch = 1;
    let time_frames = 100; // ~4 seconds at 25fps token rate
    let num_mels_40 = 40;
    let num_mels_80 = 80;

    // =====================================================
    // TEST 1: VoiceEncoder (ve.safetensors)
    // =====================================================
    println!("--- TEST 1: VoiceEncoder ---");
    let ve_path = format!("{}/ve.safetensors", model_dir);
    let vb_ve = unsafe { VarBuilder::from_mmaped_safetensors(&[&ve_path], DType::F32, &device)? };
    let ve = candle::voice_encoder::VoiceEncoder::new(
        candle::voice_encoder::VoiceEncoderConfig::default(),
        vb_ve,
    )?;

    // VoiceEncoder expects: (B, T, 40) - time-major format
    let mel_40_input = Tensor::randn(0f32, 1f32, (batch, time_frames, num_mels_40), &device)?;
    println!("  Input: {:?}", mel_40_input.dims());

    match ve.forward(&mel_40_input) {
        Ok(out) => println!("  ✓ Output: {:?} (expected [1, 256])", out.dims()),
        Err(e) => println!("  ✗ Error: {}", e),
    }

    // =====================================================
    // TEST 2: CAMPPlus (speaker_encoder in s3gen_meanflow.safetensors)
    // =====================================================
    println!("\n--- TEST 2: CAMPPlus ---");
    let s3gen_path = format!("{}/s3gen_meanflow.safetensors", model_dir);
    let vb_s3 =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&s3gen_path], DType::F32, &device)? };
    let campplus = candle::campplus::CAMPPlus::new(80, 192, vb_s3.pp("speaker_encoder"))?;

    // CAMPPlus expects: (B, C, T) = (B, 80, T) - channel-first format
    let mel_80_input = Tensor::randn(0f32, 1f32, (batch, num_mels_80, time_frames), &device)?;
    println!("  Input: {:?}", mel_80_input.dims());

    match campplus.forward(&mel_80_input) {
        Ok(out) => println!("  ✓ Output: {:?} (expected [1, 192])", out.dims()),
        Err(e) => println!("  ✗ Error: {}", e),
    }

    // =====================================================
    // TEST 3: S3Tokenizer (from local s3tokenizer-v2-model)
    // =====================================================
    println!("\n--- TEST 3: S3TokenizerV2 ---");
    let s3tok_path = "D:/chatterbox-rs/s3tokenizer-v2-model/model.safetensors";
    if std::path::Path::new(s3tok_path).exists() {
        let vb_s3tok =
            unsafe { VarBuilder::from_mmaped_safetensors(&[s3tok_path], DType::F32, &device)? };
        let s3tokenizer = candle::s3tokenizer::S3TokenizerV2::new(
            &candle::s3tokenizer::ModelConfig::default(),
            vb_s3tok,
        )?;

        // S3Tokenizer expects: (B, 128, T) mel spectrogram (padded 80->128)
        let mel_128_input = Tensor::randn(0f32, 1f32, (batch, 128, time_frames), &device)?;
        println!("  Input: {:?}", mel_128_input.dims());

        match s3tokenizer.encode(&mel_128_input) {
            Ok(out) => println!("  ✓ Output tokens: {:?}", out.dims()),
            Err(e) => println!("  ✗ Error: {}", e),
        }
    } else {
        println!("  (Skipped - model not found at {})", s3tok_path);
    }

    // =====================================================
    // TEST 4: S3Gen Flow (flow.* in s3gen_meanflow.safetensors)
    // =====================================================
    println!("\n--- TEST 4: S3Gen ---");
    let s3gen = candle::s3gen::S3Gen::new(vb_s3.clone(), true)?;

    // S3Gen.forward expects speech_tokens: (B, T) and spks: Option<(B, 80)>
    let speech_tokens = Tensor::from_vec(vec![100u32; 20], (batch, 20), &device)?;
    let spk_emb = Tensor::randn(0f32, 1f32, (batch, 80), &device)?;
    println!("  Speech tokens: {:?}", speech_tokens.dims());
    println!("  Speaker embedding: {:?}", spk_emb.dims());

    let dummy_mel_80 = Tensor::randn(0f32, 1f32, (batch, 80, 20), &device)?;
    match s3gen.forward(&speech_tokens, Some(&spk_emb), Some(&dummy_mel_80)) {
        Ok(out) => println!("  ✓ Output: {:?}", out.dims()),
        Err(e) => println!("  ✗ Error: {}", e),
    }

    // =====================================================
    // TEST 5: HiFiGAN vocoder (mel2wav.* in s3gen_meanflow.safetensors)
    // =====================================================
    println!("\n--- TEST 5: HiFiGAN ---");
    let hifigan_config = candle::hifigan::HiFTConfig {
        in_channels: 80,
        base_channels: 512,
        nb_harmonics: 8,
        sampling_rate: 24000,
        upsample_rates: vec![8, 5, 3],
        upsample_kernel_sizes: vec![16, 11, 7],
        resblock_kernel_sizes: vec![3, 7, 11],
        resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        n_fft: 16,
        hop_len: 4,
    };
    let hifigan = candle::hifigan::HiFTGenerator::new(hifigan_config, vb_s3.pp("mel2wav"))?;

    // HiFiGAN expects: (B, 80, T) mel spectrogram
    let mel_for_vocoder = Tensor::randn(0f32, 1f32, (batch, 80, 40), &device)?;
    println!("  Input mel: {:?}", mel_for_vocoder.dims());

    match hifigan.inference(&mel_for_vocoder) {
        Ok(out) => println!("  ✓ Output audio: {:?}", out.dims()),
        Err(e) => println!("  ✗ Error: {}", e),
    }

    // =====================================================
    // TEST 6: T3 model (t3_turbo_v1.safetensors)
    // =====================================================
    println!("\n--- TEST 6: T3 ---");
    let t3_path = format!("{}/t3_turbo_v1.safetensors", model_dir);
    let vb_t3 = unsafe { VarBuilder::from_mmaped_safetensors(&[&t3_path], DType::F32, &device)? };
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
    let t3 = candle::t3_model::T3::new(t3_config, vb_t3)?;

    // T3 inputs
    let text_tokens = Tensor::from_vec(vec![100u32; 10], (batch, 10), &device)?;
    let speaker_emb = Tensor::randn(0f32, 1f32, (batch, 256), &device)?;
    let prompt_tokens = Tensor::from_vec(vec![100u32; 50], (batch, 50), &device)?;
    println!("  Text tokens: {:?}", text_tokens.dims());
    println!("  Speaker embedding: {:?}", speaker_emb.dims());
    println!("  Prompt tokens: {:?}", prompt_tokens.dims());

    match t3.generate(
        &text_tokens,
        &speaker_emb,
        Some(&prompt_tokens),
        None,
        10, // max_new_tokens (small for test)
        0.8,
        0.95,
        50,
        1.2,
        42,
    ) {
        Ok(out) => println!("  ✓ Generated tokens: {:?}", out.dims()),
        Err(e) => println!("  ✗ Error: {}", e),
    }

    println!("\n=== SHAPE VERIFICATION COMPLETE ===");
    Ok(())
}
