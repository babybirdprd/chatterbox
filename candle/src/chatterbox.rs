use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use crate::audio::{self, AudioProcessor, MelConfig, S3GEN_SR, S3_SR};
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

        // 1. Voice Encoder: 16k, 40 mels, Power, Linear
        let cfg_ve = MelConfig::for_voice_encoder();
        let mel_ve =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_16k, &self.device, &cfg_ve)?;
        let mel_ve_t = mel_ve.transpose(1, 2)?;
        let spk_emb_256 = self
            .voice_encoder
            .inference(&mel_ve_t, 0.5, Some(1.3), 0.8)?;

        // 2. S3Tokenizer: 16k, 128 mels, Power, Log10 + Norm
        let cfg_s3tok = MelConfig::for_s3tokenizer();
        let mel_s3tok =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_16k, &self.device, &cfg_s3tok)?;
        let mel_s3tok_log = AudioProcessor::log_process(&mel_s3tok, &cfg_s3tok)?;

        let max_val = mel_s3tok_log.max_all()?.to_scalar::<f32>()?;
        let mel_s3tok_norm = mel_s3tok_log.maximum(max_val - 8.0)?;
        let mel_s3tok_final = ((mel_s3tok_norm + 4.0)? / 4.0)?;

        let prompt_tokens = self.s3tokenizer.encode(&mel_s3tok_final)?;

        // 3. CAMPPlus: 16k, 80 mels, Power, Ln + Mean Norm
        let cfg_camp = MelConfig::for_campplus();
        let mel_camp =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_16k, &self.device, &cfg_camp)?;
        let mel_camp_log = mel_camp.clamp(1e-5, f32::MAX)?.log()?;
        let mean = mel_camp_log.mean_keepdim(2)?;
        let mel_camp_norm = mel_camp_log.broadcast_sub(&mean)?;

        let spk_emb_80 = self
            .s3gen
            .campplus
            .forward(&mel_camp_norm)?
            .narrow(1, 0, 80)?;

        // 4. S3Gen Conditioning: 24k, 80 mels, Magnitude, Ln
        let cfg_s3gen = MelConfig::for_s3gen();
        let mel_s3gen =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_24k, &self.device, &cfg_s3gen)?;
        let mel_s3gen_log = AudioProcessor::log_process(&mel_s3gen, &cfg_s3gen)?;

        // Generate
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
        // Filter tokens and append silence tokens (S3GEN_SIL = 4299)
        let speech_tokens_filtered = {
            let tokens = speech_tokens.to_vec2::<u32>()?[0].clone();
            let mut filtered: Vec<u32> = tokens.into_iter().filter(|&t| t < 6561).collect();
            filtered.extend_from_slice(&[4299, 4299, 4299]);
            Tensor::from_vec(filtered.clone(), (1, filtered.len()), &self.device)?
        };

        // S3Gen Forward
        // Prepare S3Gen Conditioning (Prompt Mel + Zero Pad)
        let token_len = speech_tokens_filtered.dim(1)?;
        let target_len = token_len * 2;
        let (b, c, prompt_len) = mel_s3gen_log.dims3()?;
        let cond = if prompt_len < target_len {
            let pad = Tensor::zeros((b, c, target_len - prompt_len), DType::F32, &self.device)?;
            Tensor::cat(&[&mel_s3gen_log, &pad], 2)?
        } else {
            mel_s3gen_log.narrow(2, 0, target_len)?
        };

        let audio_tensor =
            self.s3gen
                .forward(&speech_tokens_filtered, Some(&spk_emb_80), Some(&cond))?;

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

        eprintln!("[generate_speech] Computing mel spectrograms...");

        // 1. Voice Encoder
        let cfg_ve = MelConfig::for_voice_encoder();
        let mel_ve =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_16k, &self.device, &cfg_ve)?;
        let mel_ve_t = mel_ve.transpose(1, 2)?;
        let spk_emb_256 = self
            .voice_encoder
            .inference(&mel_ve_t, 0.5, Some(1.3), 0.8)?;

        // 2. S3Tokenizer
        let cfg_s3tok = MelConfig::for_s3tokenizer();
        let mel_s3tok =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_16k, &self.device, &cfg_s3tok)?;
        let mel_s3tok_log = AudioProcessor::log_process(&mel_s3tok, &cfg_s3tok)?;

        let max_val = mel_s3tok_log.max_all()?.to_scalar::<f32>()?;
        let mel_s3tok_norm = mel_s3tok_log.maximum(max_val - 8.0)?;
        let mel_s3tok_final = ((mel_s3tok_norm + 4.0)? / 4.0)?;

        let prompt_tokens = self.s3tokenizer.encode(&mel_s3tok_final)?;

        // 3. CAMPPlus
        let cfg_camp = MelConfig::for_campplus();
        let mel_camp =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_16k, &self.device, &cfg_camp)?;
        let mel_camp_log = mel_camp.clamp(1e-5, f32::MAX)?.log()?;
        let mean = mel_camp_log.mean_keepdim(2)?;
        let mel_camp_norm = mel_camp_log.broadcast_sub(&mean)?;

        let spk_emb_80 = self
            .s3gen
            .campplus
            .forward(&mel_camp_norm)?
            .narrow(1, 0, 80)?;

        // 4. S3Gen Conditioning
        let cfg_s3gen = MelConfig::for_s3gen();
        let mel_s3gen =
            AudioProcessor::compute_mel_spectrogram(&ref_samples_24k, &self.device, &cfg_s3gen)?;
        let mel_s3gen_log = AudioProcessor::log_process(&mel_s3gen, &cfg_s3gen)?;

        // Generate
        eprintln!("[generate_speech] Running T3.generate...");
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

        // Filter tokens and append silence tokens (S3GEN_SIL = 4299)
        // Python: speech_tokens = speech_tokens[speech_tokens < 6561]
        //         silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL])
        //         speech_tokens = torch.cat([speech_tokens, silence])
        let speech_tokens_filtered = {
            let tokens = speech_tokens.to_vec2::<u32>()?[0].clone();
            let mut filtered: Vec<u32> = tokens.into_iter().filter(|&t| t < 6561).collect();
            // Append silence tokens to prevent audio cutoff
            filtered.extend_from_slice(&[4299, 4299, 4299]);
            Tensor::from_vec(filtered.clone(), (1, filtered.len()), &self.device)?
        };

        // S3Gen Forward with Conditioning
        eprintln!("[generate_speech] Running S3Gen.forward...");

        // Target length is 2x tokens (upsampling)
        let token_len = speech_tokens_filtered.dim(1)?;
        let target_len = token_len * 2;
        let (b, c, prompt_len) = mel_s3gen_log.dims3()?;
        let cond = if prompt_len < target_len {
            let pad = Tensor::zeros((b, c, target_len - prompt_len), DType::F32, &self.device)?;
            Tensor::cat(&[&mel_s3gen_log, &pad], 2)?
        } else {
            mel_s3gen_log.narrow(2, 0, target_len)?
        };

        let audio_tensor =
            self.s3gen
                .forward(&speech_tokens_filtered, Some(&spk_emb_80), Some(&cond))?;

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
