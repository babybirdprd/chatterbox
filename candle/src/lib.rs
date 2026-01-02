//! Chatterbox TTS - Rust port of the Chatterbox text-to-speech system.
//!
//! This crate provides a clean API for text-to-speech synthesis using the
//! Chatterbox Turbo model.
//!
//! # Example
//!
//! ```no_run
//! use candle::chatterbox::ChatterboxTurboTTS;
//! use candle::audio;
//! use candle_core::Device;
//! use std::path::Path;
//!
//! fn main() -> anyhow::Result<()> {
//!     // Load model (downloads from HuggingFace if needed)
//!     let model = ChatterboxTurboTTS::from_pretrained(Device::Cpu)?;
//!
//!     // Generate speech
//!     let (samples, sample_rate) = model.generate_speech(
//!         "Hello, world!",
//!         Path::new("reference.wav"),
//!         Default::default(),
//!     )?;
//!
//!     // Save to file
//!     audio::save_wav("output.wav", &samples, sample_rate)?;
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - `cuda` - Enable CUDA GPU acceleration

pub mod audio;
pub mod chatterbox;
pub mod gpt2;
pub mod hifigan;
pub mod modules;
pub mod s3gen;
pub mod t3_model;
pub mod voice_encoder;

// Re-export main types for convenience
pub use audio::{load_wav, resample, save_wav, S3GEN_SR, S3_SR};
pub use chatterbox::{ChatterboxTTS, ChatterboxTurboTTS};

/// Generation configuration options
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Sampling temperature (0.05-2.0, default 0.8)
    pub temperature: f32,
    /// Top-p nucleus sampling (0.0-1.0, default 0.95)
    pub top_p: f32,
    /// Top-k sampling (0-1000, default 1000)
    pub top_k: usize,
    /// Repetition penalty (1.0-2.0, default 1.2)
    pub repetition_penalty: f32,
    /// Random seed (0 for random)
    pub seed: u64,
    /// Normalize output loudness
    pub normalize_loudness: bool,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 1000,
            repetition_penalty: 1.2,
            seed: 0,
            normalize_loudness: true,
        }
    }
}
