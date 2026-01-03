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
        let (mut ref_samples, ref_sr) =
            audio::load_wav(ref_audio_path).map_err(|e| candle_core::Error::Msg(e))?;
        if ref_sr != S3_SR {
            ref_samples = audio::resample(&ref_samples, ref_sr, S3_SR)
                .map_err(|e| candle_core::Error::Msg(e))?;
        }

        let mel_40 = audio::compute_mel_spectrogram(&ref_samples, S3_SR, &self.device, 40)?;
        let mel_80 = audio::compute_mel_spectrogram(&ref_samples, S3_SR, &self.device, 80)?;

        let spk_emb_256 = self.voice_encoder.forward(&mel_40)?;
        let spk_emb_80 = self.s3gen.campplus.forward(&mel_80)?.narrow(1, 0, 80)?;

        // S3Tokenizer expects (B, 128, T) mel - transpose and pad from (B, T, 80)
        let prompt_tokens = {
            let mel_t = mel_80.transpose(1, 2)?; // (B, 80, T)
            let (b, c, t) = mel_t.dims3()?;
            let padding = Tensor::zeros((b, 128 - c, t), DType::F32, &self.device)?;
            let mel_padded = Tensor::cat(&[&mel_t, &padding], 1)?; // (B, 128, T)
            self.s3tokenizer.encode(&mel_padded)?
        };
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
        let audio_tensor = self.s3gen.forward(&speech_tokens, Some(&spk_emb_80))?;
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
        let (mut ref_samples, ref_sr) =
            audio::load_wav(ref_audio_path).map_err(|e| candle_core::Error::Msg(e))?;
        if ref_sr != S3_SR {
            ref_samples = audio::resample(&ref_samples, ref_sr, S3_SR)
                .map_err(|e| candle_core::Error::Msg(e))?;
        }

        let mel_40 = audio::compute_mel_spectrogram(&ref_samples, S3_SR, &self.device, 40)?;
        let mel_80 = audio::compute_mel_spectrogram(&ref_samples, S3_SR, &self.device, 80)?;

        let spk_emb_256 = self.voice_encoder.forward(&mel_40)?;
        let spk_emb_80 = self.s3gen.campplus.forward(&mel_80)?.narrow(1, 0, 80)?;

        // S3Tokenizer expects (B, 128, T) mel - transpose and pad from (B, T, 80)
        let prompt_tokens = {
            let mel_t = mel_80.transpose(1, 2)?; // (B, 80, T)
            let (b, c, t) = mel_t.dims3()?;
            let padding = Tensor::zeros((b, 128 - c, t), DType::F32, &self.device)?;
            let mel_padded = Tensor::cat(&[&mel_t, &padding], 1)?; // (B, 128, T)
            self.s3tokenizer.encode(&mel_padded)?
        };
        let text_tokens = self.tokenize_text(&text)?;

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
        let speech_tokens_filtered = {
            let tokens = speech_tokens.to_vec2::<u32>()?[0].clone();
            let filtered: Vec<u32> = tokens.into_iter().filter(|&t| t < 6561).collect();
            Tensor::from_vec(filtered.clone(), (1, filtered.len()), &self.device)?
        };
        let audio_tensor = self
            .s3gen
            .forward(&speech_tokens_filtered, Some(&spk_emb_80))?;
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
