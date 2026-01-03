pub mod audio;
pub mod campplus;
pub mod chatterbox;
pub mod gpt2;
pub mod hifigan;
pub mod modules;
pub mod s3gen;
pub mod s3tokenizer;
pub mod sampling;
pub mod t3_model;
pub mod voice_encoder;

pub use audio::{load_wav, resample, save_wav, S3GEN_SR, S3_SR};
pub use campplus::CAMPPlus;
pub use chatterbox::{ChatterboxTTS, ChatterboxTurboTTS};
pub use s3tokenizer::{ModelConfig as S3TokenizerConfig, S3TokenizerV2};
pub use sampling::LogitsProcessor;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: u64,
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
