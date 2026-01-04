use candle_core::{DType, Device};
use candle_nn::VarBuilder;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // 1. Load debug tensors
    let debug_path = "debug_tensors.safetensors";
    if !std::path::Path::new(debug_path).exists() {
        anyhow::bail!("debug_tensors.safetensors not found. Run extract_debug.py first.");
    }

    let tensors = candle_core::safetensors::load(debug_path, &device)?;
    let ref_mel = tensors
        .get("ref_mel")
        .ok_or_else(|| anyhow::anyhow!("ref_mel not found"))?
        .clone();
    let ref_audio = tensors
        .get("ref_audio")
        .ok_or_else(|| anyhow::anyhow!("ref_audio not found"))?
        .clone();

    println!("Loaded ground truth mel: {:?}", ref_mel.dims());

    // 2. Load HiFiGAN
    let model_path = "C:/Users/Steve Business/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd/s3gen_meanflow.safetensors";
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    let config = candle::hifigan::HiFTConfig {
        in_channels: 80,
        base_channels: 512,
        nb_harmonics: 8,
        sampling_rate: 24000,
        upsample_rates: vec![8, 5, 3],
        upsample_kernel_sizes: vec![16, 11, 7],
        resblock_kernel_sizes: vec![3, 7, 11],
        resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        n_fft: 16,
        hop_len: 2,
    };

    let vocoder = candle::hifigan::HiFTGenerator::new(config, vb.pp("mel2wav"))?;
    println!("Vocoder loaded.");

    // 3. Run inference (with B=1)
    let mel_input = ref_mel.unsqueeze(0)?;
    println!("Running vocoder inference on mel {:?}...", mel_input.dims());
    let audio = vocoder.inference(&mel_input)?;
    println!("Inference complete: {:?}", audio.dims());

    // 4. Save output
    let samples = audio.flatten_all()?.to_vec1::<f32>()?;
    candle::audio::save_wav("test_vocoder_isolated.wav", &samples, 24000)
        .map_err(|e| anyhow::anyhow!(e))?;
    println!("Saved test_vocoder_isolated.wav");

    // 5. Verify Mel Extraction (Rust)
    let ref_audio_vec = ref_audio.flatten_all()?.to_vec1::<f32>()?;
    let rust_mel = candle::audio::compute_mel_spectrogram(
        &ref_audio_vec,
        24000,
        &device,
        &candle::audio::MelConfig::for_24k(80),
    )?;

    println!("Rust mel shape: {:?}", rust_mel.dims());

    // Compare first few values: ref_mel is [80, T], rust_mel is [1, 80, T]
    let gt_val = ref_mel.get(0)?.get(0)?.to_vec0::<f32>()?;
    let rust_val = rust_mel.get(0)?.get(0)?.get(0)?.to_vec0::<f32>()?;
    println!(
        "Comparison mel[0,0]: GT={:.6}, Rust={:.6}",
        gt_val, rust_val
    );

    Ok(())
}
