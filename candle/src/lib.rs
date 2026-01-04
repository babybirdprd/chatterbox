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

pub use audio::{load_wav, resample, save_wav, AudioProcessor, MelConfig, S3GEN_SR, S3_SR};
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
    /// Minimum probability threshold (0.0 to disable)
    pub min_p: f32,
    /// Emotion exaggeration factor (maps to emotion_adv in T3)
    pub exaggeration: f32,
    /// Classifier-Free Guidance weight (0.0 to disable)
    pub cfg_weight: f32,
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
            min_p: 0.0,
            exaggeration: 0.0,
            cfg_weight: 0.0,
        }
    }
}
