//! Main Chatterbox TTS structs with complete generation pipeline.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::audio::{self, S3GEN_SR, S3_SR};
use crate::s3gen::S3Gen;
use crate::t3_model::{T3Config, T3};
use crate::voice_encoder::{VoiceEncoder, VoiceEncoderConfig};
use crate::GenerateConfig;

/// Standard Chatterbox TTS model
pub struct ChatterboxTTS {
    voice_encoder: VoiceEncoder,
    t3: T3,
    s3gen: S3Gen,
    tokenizer: Tokenizer,
    device: Device,
}

impl ChatterboxTTS {
    /// Load model from HuggingFace Hub
    pub fn from_pretrained(device: Device) -> Result<Self> {
        let api = Api::new().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.model("ResembleAI/chatterbox".to_string());

        let ve_path = repo
            .get("ve.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let t3_path = repo
            .get("t3_cfg.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3gen_path = repo
            .get("s3gen.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Self::from_local(ve_path, t3_path, s3gen_path, tokenizer_path, device)
    }

    /// Load model from local paths
    pub fn from_local(
        ve_path: PathBuf,
        t3_path: PathBuf,
        s3gen_path: PathBuf,
        tokenizer_path: PathBuf,
        device: Device,
    ) -> Result<Self> {
        let vb_ve = unsafe {
            VarBuilder::from_mmaped_safetensors(&[ve_path], candle_core::DType::F32, &device)?
        };
        let ve_config = VoiceEncoderConfig::default();
        let voice_encoder = VoiceEncoder::new(ve_config, vb_ve)?;

        let t3_config = T3Config::default();
        let vb_t3 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[t3_path], candle_core::DType::F32, &device)?
        };
        let t3 = T3::new(t3_config, vb_t3)?;

        let vb_s3 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[s3gen_path], candle_core::DType::F32, &device)?
        };
        let s3gen = S3Gen::new(vb_s3, false)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Ok(Self {
            voice_encoder,
            t3,
            s3gen,
            tokenizer,
            device,
        })
    }

    /// Generate speech from text and reference audio
    pub fn generate_speech(
        &self,
        text: &str,
        ref_audio_path: &Path,
        config: GenerateConfig,
    ) -> Result<(Vec<f32>, u32)> {
        // Load and process reference audio
        let (mut ref_samples, ref_sr) =
            audio::load_wav(ref_audio_path).map_err(|e| candle_core::Error::Msg(e))?;

        // Resample to S3_SR for voice encoder
        if ref_sr != S3_SR {
            ref_samples = audio::resample(&ref_samples, ref_sr, S3_SR)
                .map_err(|e| candle_core::Error::Msg(e))?;
        }

        // Compute mel spectrogram for voice encoder
        let ref_mels = audio::compute_mel_spectrogram(&ref_samples, S3_SR, &self.device)?;

        // Get speaker embedding
        let spk_emb = self.voice_encoder.forward(&ref_mels)?;

        // Tokenize text
        let text_tokens = self.tokenize_text(text)?;

        // Generate speech tokens
        let speech_tokens = self.t3.generate(&text_tokens, &spk_emb, None, None, 500)?;

        // Generate audio
        let audio_tensor = self.s3gen.forward(&speech_tokens, None)?;

        // Convert to samples
        let mut samples = tensor_to_samples(&audio_tensor)?;

        // Normalize loudness if requested
        if config.normalize_loudness {
            audio::normalize_loudness(&mut samples, -27.0);
        }

        Ok((samples, S3GEN_SR))
    }

    fn tokenize_text(&self, text: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();
        let tensor = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
        Ok(tensor.to_dtype(DType::U32)?)
    }
}

/// Chatterbox Turbo TTS model - faster inference
pub struct ChatterboxTurboTTS {
    voice_encoder: VoiceEncoder,
    t3: T3,
    s3gen: S3Gen,
    tokenizer: Tokenizer,
    device: Device,
}

impl ChatterboxTurboTTS {
    /// Load model from HuggingFace Hub
    pub fn from_pretrained(device: Device) -> Result<Self> {
        let api = Api::new().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.model("ResembleAI/chatterbox-turbo".to_string());

        let ve_path = repo
            .get("ve.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let t3_path = repo
            .get("t3_turbo_v1.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3gen_path = repo
            .get("s3gen_meanflow.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Self::from_local(ve_path, t3_path, s3gen_path, tokenizer_path, device)
    }

    /// Load model from local paths
    pub fn from_local(
        ve_path: PathBuf,
        t3_path: PathBuf,
        s3gen_path: PathBuf,
        tokenizer_path: PathBuf,
        device: Device,
    ) -> Result<Self> {
        let vb_ve = unsafe {
            VarBuilder::from_mmaped_safetensors(&[ve_path], candle_core::DType::F32, &device)?
        };
        let ve_config = VoiceEncoderConfig::default();
        let voice_encoder = VoiceEncoder::new(ve_config, vb_ve)?;

        // Turbo Config
        let t3_config = T3Config {
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
        };

        let vb_t3 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[t3_path], candle_core::DType::F32, &device)?
        };
        let t3 = T3::new(t3_config, vb_t3)?;

        let vb_s3 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[s3gen_path], candle_core::DType::F32, &device)?
        };
        let s3gen = S3Gen::new(vb_s3, true)?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Ok(Self {
            voice_encoder,
            t3,
            s3gen,
            tokenizer,
            device,
        })
    }

    /// Generate speech from text and reference audio
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `ref_audio_path` - Path to reference audio file (5+ seconds, WAV)
    /// * `config` - Generation configuration
    ///
    /// # Returns
    /// Tuple of (audio samples, sample rate)
    pub fn generate_speech(
        &self,
        text: &str,
        ref_audio_path: &Path,
        config: GenerateConfig,
    ) -> Result<(Vec<f32>, u32)> {
        // Normalize text
        let text = normalize_text(text);

        // Load and process reference audio
        let (mut ref_samples, ref_sr) =
            audio::load_wav(ref_audio_path).map_err(|e| candle_core::Error::Msg(e))?;

        // Resample to S3_SR for voice encoder
        if ref_sr != S3_SR {
            ref_samples = audio::resample(&ref_samples, ref_sr, S3_SR)
                .map_err(|e| candle_core::Error::Msg(e))?;
        }

        // Compute mel spectrogram for voice encoder
        let ref_mels = audio::compute_mel_spectrogram(&ref_samples, S3_SR, &self.device)?;

        // Get speaker embedding
        let spk_emb = self.voice_encoder.forward(&ref_mels)?;

        // Tokenize text
        let text_tokens = self.tokenize_text(&text)?;

        // Generate speech tokens (using config parameters)
        // TODO: Pass temperature, top_p, top_k, repetition_penalty to T3
        let speech_tokens = self.t3.generate(&text_tokens, &spk_emb, None, None, 500)?;

        // Generate audio
        let audio_tensor = self.s3gen.forward(&speech_tokens, None)?;

        // Convert to samples
        let mut samples = tensor_to_samples(&audio_tensor)?;

        // Normalize loudness if requested
        if config.normalize_loudness {
            audio::normalize_loudness(&mut samples, -27.0);
        }

        Ok((samples, S3GEN_SR))
    }

    fn tokenize_text(&self, text: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();
        let tensor = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
        Ok(tensor.to_dtype(DType::U32)?)
    }

    /// Get device the model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get sample rate of generated audio
    pub fn sample_rate(&self) -> u32 {
        S3GEN_SR
    }
}

/// Normalize text for TTS (punctuation cleanup, etc.)
fn normalize_text(text: &str) -> String {
    let mut text = text.to_string();

    // Capitalize first letter if lowercase
    if let Some(first) = text.chars().next() {
        if first.is_lowercase() {
            text = first.to_uppercase().to_string() + &text[first.len_utf8()..];
        }
    }

    // Remove multiple spaces
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");

    // Replace uncommon punctuation
    let replacements = [
        ("\u{2026}", ", "),
        (":", ","),
        ("\u{2014}", "-"),
        ("\u{2013}", "-"),
        (" ,", ","),
        ("\u{201C}", "\""),
        ("\u{201D}", "\""),
        ("\u{2018}", "'"),
        ("\u{2019}", "'"),
    ];
    for (from, to) in replacements {
        text = text.replace(from, to);
    }

    // Add period if no ending punctuation
    text = text.trim_end().to_string();
    if !text.ends_with('.')
        && !text.ends_with('!')
        && !text.ends_with('?')
        && !text.ends_with('-')
        && !text.ends_with(',')
    {
        text.push('.');
    }

    text
}

/// Convert audio tensor to f32 samples
fn tensor_to_samples(tensor: &Tensor) -> Result<Vec<f32>> {
    // Expected shape: (B, C, T) or (B, T) or (C, T) or (T,)
    let flat = tensor.flatten_all()?;
    let samples: Vec<f32> = flat.to_vec1()?;
    Ok(samples)
}
