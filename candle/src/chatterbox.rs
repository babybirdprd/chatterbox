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

pub struct ChatterboxTTS {
    t3: T3,
    s3gen: S3Gen,
    s3tokenizer: crate::s3tokenizer::S3TokenizerV2,
    voice_encoder: VoiceEncoder,
    tokenizer: Tokenizer,
    device: Device,
}

impl ChatterboxTTS {
    pub fn from_pretrained(device: Device) -> Result<Self> {
        let api = Api::new().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.model("ResembleAI/chatterbox".to_string());
        let t3_path = repo
            .get("t3_cfg.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3gen_path = repo
            .get("s3gen.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let ve_path = repo
            .get("ve.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3tokenizer_path = api
            .model("ResembleAI/s3tokenizer-v2".to_string())
            .get("model.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Self::from_local(
            t3_path,
            s3gen_path,
            ve_path,
            s3tokenizer_path,
            tokenizer_path,
            device,
        )
    }

    pub fn from_local(
        t3_path: PathBuf,
        s3gen_path: PathBuf,
        ve_path: PathBuf,
        s3tokenizer_path: PathBuf,
        tokenizer_path: PathBuf,
        device: Device,
    ) -> Result<Self> {
        let vb_t3 =
            unsafe { VarBuilder::from_mmaped_safetensors(&[t3_path], DType::F32, &device)? };
        let t3 = T3::new(T3Config::default(), vb_t3)?;
        let vb_s3 =
            unsafe { VarBuilder::from_mmaped_safetensors(&[s3gen_path], DType::F32, &device)? };
        let s3gen = S3Gen::new(vb_s3, false)?;
        let vb_ve =
            unsafe { VarBuilder::from_mmaped_safetensors(&[ve_path], DType::F32, &device)? };
        let voice_encoder = VoiceEncoder::new(VoiceEncoderConfig::default(), vb_ve)?;
        let vb_s3tok = unsafe {
            VarBuilder::from_mmaped_safetensors(&[s3tokenizer_path], DType::F32, &device)?
        };
        let s3tokenizer = crate::s3tokenizer::S3TokenizerV2::new(
            &crate::s3tokenizer::ModelConfig::default(),
            vb_s3tok,
        )?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(Self {
            t3,
            s3gen,
            s3tokenizer,
            voice_encoder,
            tokenizer,
            device,
        })
    }

    pub fn generate_speech(
        &self,
        text: &str,
        ref_audio_path: &Path,
        config: GenerateConfig,
    ) -> Result<(Vec<f32>, u32)> {
        let (ref_samples, ref_sr) =
            audio::load_wav(ref_audio_path).map_err(|e| candle_core::Error::Msg(e))?;
        let ref_samples_16k = if ref_sr != S3_SR {
            audio::resample(&ref_samples, ref_sr, S3_SR).map_err(|e| candle_core::Error::Msg(e))?
        } else {
            ref_samples.clone()
        };
        let ref_samples_24k = if ref_sr != S3GEN_SR {
            audio::resample(&ref_samples, ref_sr, S3GEN_SR)
                .map_err(|e| candle_core::Error::Msg(e))?
        } else {
            ref_samples
        };

        // VoiceEncoder expects 16kHz mel (40 channels, n_fft=400)
        let config_ve = audio::MelConfig {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 40,
            fmax: 8000.0,
        };
        // S3Tokenizer expects 16kHz mel (128 channels, n_fft=400)
        let config_s3tok = audio::MelConfig {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 128,
            fmax: 8000.0,
        };

        let mel_40 =
            audio::compute_mel_spectrogram(&ref_samples_16k, S3_SR, &self.device, &config_ve)?;

        let mel_128 =
            audio::compute_mel_spectrogram(&ref_samples_16k, S3_SR, &self.device, &config_s3tok)?;

        let mel_80_24k = audio::compute_mel_spectrogram(
            &ref_samples_24k,
            S3GEN_SR,
            &self.device,
            &audio::MelConfig::for_24k(80),
        )?;

        // VoiceEncoder expects Power Mel Spectrogram (amp^2), not dB, not Magnitude
        let mel_40_power = mel_40.sqr()?;
        let mel_40_t = mel_40_power.transpose(1, 2)?; // (B, T, 40)
        let spk_emb_256 = self.voice_encoder.forward(&mel_40_t)?;

        // CAMPPlus expects Mean-Normalized Log-Mel
        let mel_80_log = mel_80_24k.clamp(1e-5, f32::MAX)?.log()?;
        let mean = mel_80_log.mean_keepdim(2)?;
        let mel_80_norm = mel_80_log.broadcast_sub(&mean)?;
        let spk_emb_80 = self
            .s3gen
            .campplus
            .forward(&mel_80_norm)?
            .narrow(1, 0, 80)?;

        // S3Tokenizer expects (B, 128, T) mel - now computed directly
        let prompt_tokens = self.s3tokenizer.encode(&mel_128)?;
        let text_tokens = self.tokenize_text(text)?;

        let speech_tokens = self.t3.generate(
            &text_tokens,
            &spk_emb_256,
            Some(&prompt_tokens),
            None,
            500,
            config.temperature,
            config.top_p,
            config.top_k,
            config.repetition_penalty,
            config.seed,
        )?;
        let audio_tensor = self
            .s3gen
            .forward(&speech_tokens, Some(&spk_emb_80), None)?;

        let mut samples = audio_tensor.flatten_all()?.to_vec1::<f32>()?;
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
        Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)
    }
}

pub struct ChatterboxTurboTTS {
    t3: T3,
    s3gen: S3Gen,
    s3tokenizer: crate::s3tokenizer::S3TokenizerV2,
    voice_encoder: VoiceEncoder,
    tokenizer: Tokenizer,
    device: Device,
}

impl ChatterboxTurboTTS {
    pub fn from_pretrained(device: Device) -> Result<Self> {
        let api = Api::new().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let turbo_repo = api.model("ResembleAI/chatterbox-turbo".to_string());
        let t3_path = turbo_repo
            .get("t3_turbo_v1.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3gen_path = turbo_repo
            .get("s3gen_meanflow.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let ve_path = turbo_repo
            .get("ve.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3tokenizer_path = api
            .model("ResembleAI/s3tokenizer-v2".to_string())
            .get("model.safetensors")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let base_repo = api.model("ResembleAI/chatterbox".to_string());
        let tokenizer_path = base_repo
            .get("tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Self::from_local(
            t3_path,
            s3gen_path,
            ve_path,
            s3tokenizer_path,
            tokenizer_path,
            device,
        )
    }

    pub fn from_local(
        t3_path: PathBuf,
        s3gen_path: PathBuf,
        ve_path: PathBuf,
        s3tokenizer_path: PathBuf,
        tokenizer_path: PathBuf,
        device: Device,
    ) -> Result<Self> {
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
            n_positions: 8196,
        };
        let vb_t3 =
            unsafe { VarBuilder::from_mmaped_safetensors(&[t3_path], DType::F32, &device)? };
        let t3 = T3::new(t3_config, vb_t3)?;
        let vb_s3 =
            unsafe { VarBuilder::from_mmaped_safetensors(&[s3gen_path], DType::F32, &device)? };
        let s3gen = S3Gen::new(vb_s3, true)?;
        let vb_ve =
            unsafe { VarBuilder::from_mmaped_safetensors(&[ve_path], DType::F32, &device)? };
        let voice_encoder = VoiceEncoder::new(VoiceEncoderConfig::default(), vb_ve)?;
        let vb_s3tok = unsafe {
            VarBuilder::from_mmaped_safetensors(&[s3tokenizer_path], DType::F32, &device)?
        };
        let s3tokenizer = crate::s3tokenizer::S3TokenizerV2::new(
            &crate::s3tokenizer::ModelConfig::default(),
            vb_s3tok,
        )?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(Self {
            t3,
            s3gen,
            s3tokenizer,
            voice_encoder,
            tokenizer,
            device,
        })
    }

    pub fn generate_speech(
        &self,
        text: &str,
        ref_audio_path: &Path,
        config: GenerateConfig,
    ) -> Result<(Vec<f32>, u32)> {
        let text = normalize_text(text);
        eprintln!("[generate_speech] Loading reference audio...");
        let (ref_samples_orig, ref_sr) =
            audio::load_wav(ref_audio_path).map_err(|e| candle_core::Error::Msg(e))?;
        eprintln!(
            "[generate_speech] ref_samples_orig len: {}, sr: {}",
            ref_samples_orig.len(),
            ref_sr
        );

        let ref_samples_24k = if ref_sr != S3GEN_SR {
            audio::resample(&ref_samples_orig, ref_sr, S3GEN_SR)
                .map_err(|e| candle_core::Error::Msg(e))?
        } else {
            ref_samples_orig.clone()
        };

        let ref_samples_16k = if ref_sr != S3_SR {
            audio::resample(&ref_samples_orig, ref_sr, S3_SR)
                .map_err(|e| candle_core::Error::Msg(e))?
        } else {
            ref_samples_orig
        };

        eprintln!("[generate_speech] Computing mel spectrograms...");
        use std::io::Write;
        std::io::stderr().flush().unwrap();

        // VoiceEncoder expects 16kHz mel (40 channels)
        // VoiceEncoder expects 16kHz mel (40 channels)
        // VoiceEncoder expects 16kHz mel (40 channels, n_fft=400)
        let config_ve = audio::MelConfig {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 40,
            fmax: 8000.0,
        };
        // S3Tokenizer expects 16kHz mel (128 channels, n_fft=400)
        let config_s3tok = audio::MelConfig {
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 128,
            fmax: 8000.0,
        };

        let mel_40 =
            audio::compute_mel_spectrogram(&ref_samples_16k, S3_SR, &self.device, &config_ve)?;
        eprintln!("[generate_speech] mel_40 (16k): {:?}", mel_40.dims());

        // Tokenizer expects 16kHz mel (128 channels)
        let mel_128 =
            audio::compute_mel_spectrogram(&ref_samples_16k, S3_SR, &self.device, &config_s3tok)?;
        eprintln!("[generate_speech] mel_128 (16k): {:?}", mel_128.dims());

        // S3Gen expects 24kHz mel (80 channels)
        let mel_80_24k = audio::compute_mel_spectrogram(
            &ref_samples_24k,
            S3GEN_SR,
            &self.device,
            &audio::MelConfig::for_24k(80),
        )?;
        eprintln!("[generate_speech] mel_80 (24k): {:?}", mel_80_24k.dims());

        // VoiceEncoder expects Power Mel Spectrogram (amp^2), not dB, not Magnitude
        let mel_40_power = mel_40.sqr()?;
        let mel_40_t = mel_40_power.transpose(1, 2)?; // (B, T, 40)
        let spk_emb_256 = self.voice_encoder.forward(&mel_40_t)?;

        eprintln!("[generate_speech] spk_emb_256: {:?}", spk_emb_256.dims());
        eprintln!("[generate_speech] Running CAMPPlus...");
        // CAMPPlus expects Mean-Normalized Log-Mel
        let mel_80_log = mel_80_24k.clamp(1e-5, f32::MAX)?.log()?;
        let mean = mel_80_log.mean_keepdim(2)?;
        let mel_80_norm = mel_80_log.broadcast_sub(&mean)?;
        let spk_emb_80 = self
            .s3gen
            .campplus
            .forward(&mel_80_norm)?
            .narrow(1, 0, 80)?;
        eprintln!("[generate_speech] spk_emb_80: {:?}", spk_emb_80.dims());

        // S3Tokenizer expects (B, 128, T) mel - now computed directly
        eprintln!("[generate_speech] Running S3Tokenizer...");
        let prompt_tokens = self.s3tokenizer.encode(&mel_128)?;

        eprintln!(
            "[generate_speech] prompt_tokens: {:?}",
            prompt_tokens.dims()
        );
        let text_tokens = self.tokenize_text(&text)?;
        eprintln!("[generate_speech] text_tokens: {:?}", text_tokens.dims());

        eprintln!("[generate_speech] Running T3.generate...");
        let speech_tokens = self.t3.generate(
            &text_tokens,
            &spk_emb_256,
            Some(&prompt_tokens),
            None,
            500,
            config.temperature,
            config.top_p,
            config.top_k,
            config.repetition_penalty,
            config.seed,
        )?;
        eprintln!(
            "[generate_speech] speech_tokens: {:?}",
            speech_tokens.dims()
        );
        let speech_tokens_filtered = {
            let tokens = speech_tokens.to_vec2::<u32>()?[0].clone();
            let original_len = tokens.len();
            let filtered: Vec<u32> = tokens.into_iter().filter(|&t| t < 6561).collect();
            eprintln!(
                "[generate_speech] filtered tokens: {} (from {})",
                filtered.len(),
                original_len
            );
            Tensor::from_vec(filtered.clone(), (1, filtered.len()), &self.device)?
        };

        eprintln!("[generate_speech] Running S3Gen.forward...");
        let audio_tensor = self
            .s3gen
            .forward(&speech_tokens_filtered, Some(&spk_emb_80), None)?;
        eprintln!("[generate_speech] audio_tensor: {:?}", audio_tensor.dims());
        let mut samples = audio_tensor.flatten_all()?.to_vec1::<f32>()?;
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
        Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)
    }
}

fn normalize_text(text: &str) -> String {
    let mut text = text.to_string();
    if let Some(first) = text.chars().next() {
        if first.is_lowercase() {
            text = first.to_uppercase().to_string() + &text[first.len_utf8()..];
        }
    }
    text = text.split_whitespace().collect::<Vec<_>>().join(" ");
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
