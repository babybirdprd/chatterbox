//! Example: Generate speech from text using Chatterbox TTS
//!
//! Usage:
//! ```bash
//! cargo run --example generate -- --text "Hello world" --ref-audio reference.wav
//! ```

use candle::{audio, chatterbox::ChatterboxTurboTTS, GenerateConfig};
use candle_core::Device;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // Parse command line args (simplified)
    let args: Vec<String> = std::env::args().collect();

    let text = args
        .iter()
        .position(|a| a == "--text")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("Hello, this is a test of the Chatterbox TTS system.");

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

    // Select device
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

    // Load model
    println!("Loading Chatterbox Turbo model...");
    let model = ChatterboxTurboTTS::from_pretrained(device)?;
    println!("Model loaded.");

    // Generate speech
    println!("Generating speech for: \"{}\"", text);
    let config = GenerateConfig::default();
    let (samples, sample_rate) = model.generate_speech(text, Path::new(ref_audio), config)?;

    // Save output
    audio::save_wav(output, &samples, sample_rate).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Audio saved to: {} ({} samples @ {}Hz)",
        output,
        samples.len(),
        sample_rate
    );

    Ok(())
}
