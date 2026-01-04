//! Chatterbox TTS CLI - Command-line interface for text-to-speech generation.
//!
//! Mirrors the functionality of gradio_tts_turbo_app.py

use candle::{audio, chatterbox::ChatterboxTurboTTS, GenerateConfig};
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "chatterbox",
    author,
    version,
    about = "Chatterbox TTS - High-quality text-to-speech synthesis",
    long_about = "Generate speech from text using the Chatterbox Turbo model.\n\n\
                  Requires a reference audio file (5+ seconds) for voice cloning."
)]
struct Args {
    /// Text to synthesize
    #[arg(short, long)]
    text: String,

    /// Reference audio file for voice cloning (WAV, 5+ seconds)
    #[arg(short, long)]
    ref_audio: PathBuf,

    /// Output audio file path
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Device to run on
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "cuda"])]
    device: String,

    /// Sampling temperature (0.05-2.0)
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-p nucleus sampling (0.0-1.0)
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Top-k sampling (0-1000)
    #[arg(long, default_value_t = 1000)]
    top_k: usize,

    /// Repetition penalty (1.0-2.0)
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f32,

    /// Random seed (0 for random)
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Normalize output loudness to -27 LUFS
    #[arg(long, default_value_t = true)]
    normalize: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Select device
    let device = match args.device.as_str() {
        "cuda" => {
            println!("Using CUDA device...");
            Device::new_cuda(0)?
        }
        _ => {
            println!("Using CPU device...");
            Device::Cpu
        }
    };

    // Validate reference audio exists
    if !args.ref_audio.exists() {
        anyhow::bail!(
            "Reference audio file not found: {}",
            args.ref_audio.display()
        );
    }

    println!("Loading Chatterbox Turbo model...");
    let model = ChatterboxTurboTTS::from_pretrained(device.clone())?;
    println!("Model loaded successfully.");

    // Set random seed if specified
    if args.seed != 0 {
        // Note: Candle doesn't have global seed setting, handled per-operation
        println!("Using seed: {}", args.seed);
    }

    // Create generation config
    let config = GenerateConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        normalize_loudness: args.normalize,
        ..Default::default()
    };

    println!("Generating speech for: \"{}\"", args.text);
    println!("Reference audio: {}", args.ref_audio.display());

    let (samples, sample_rate) = model.generate_speech(&args.text, &args.ref_audio, config)?;

    // Save output
    audio::save_wav(&args.output, &samples, sample_rate).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!(
        "Audio saved to: {} ({} samples @ {}Hz)",
        args.output.display(),
        samples.len(),
        sample_rate
    );

    Ok(())
}
