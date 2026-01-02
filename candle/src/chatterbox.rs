use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use crate::voice_encoder::{VoiceEncoder, VoiceEncoderConfig};
use crate::t3_model::{T3, T3Config};
use crate::s3gen::S3Gen;
use std::path::PathBuf;

pub struct ChatterboxTTS {
    voice_encoder: VoiceEncoder,
    t3: T3,
    s3gen: S3Gen,
    device: Device,
}

impl ChatterboxTTS {
    pub fn from_pretrained(device: Device) -> Result<Self> {
        let api = Api::new().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.model("ResembleAI/chatterbox".to_string());

        let ve_path = repo.get("ve.safetensors").map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let t3_path = repo.get("t3_cfg.safetensors").map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3gen_path = repo.get("s3gen.safetensors").map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Self::from_local(ve_path, t3_path, s3gen_path, device)
    }

    pub fn from_local(ve_path: PathBuf, t3_path: PathBuf, s3gen_path: PathBuf, device: Device) -> Result<Self> {
        let vb_ve = unsafe { VarBuilder::from_mmaped_safetensors(&[ve_path], candle_core::DType::F32, &device)? };
        let ve_config = VoiceEncoderConfig::default();
        let voice_encoder = VoiceEncoder::new(ve_config, vb_ve)?;

        // T3 config for Standard Chatterbox
        let t3_config = T3Config::default();

        let vb_t3 = unsafe { VarBuilder::from_mmaped_safetensors(&[t3_path], candle_core::DType::F32, &device)? };
        let t3 = T3::new(t3_config, vb_t3)?;

        let vb_s3 = unsafe { VarBuilder::from_mmaped_safetensors(&[s3gen_path], candle_core::DType::F32, &device)? };
        let s3gen = S3Gen::new(vb_s3, false)?; // meanflow=False

        Ok(Self {
            voice_encoder,
            t3,
            s3gen,
            device,
        })
    }

    pub fn generate(&self, text_tokens: &Tensor, ref_mels: &Tensor, max_len: usize) -> Result<Tensor> {
        let spk_emb = self.voice_encoder.forward(ref_mels)?; // (B, E)

        // Generate speech tokens
        let speech_tokens = self.t3.generate(text_tokens, &spk_emb, None, None, max_len)?;

        // S3Gen Inference with dummy spks
        let wav = self.s3gen.forward(&speech_tokens, None)?; // spks=None, will use zeros inside S3Gen

        Ok(wav)
    }
}

pub struct ChatterboxTurboTTS {
    voice_encoder: VoiceEncoder,
    t3: T3,
    s3gen: S3Gen,
    device: Device,
}

impl ChatterboxTurboTTS {
    pub fn from_pretrained(device: Device) -> Result<Self> {
        let api = Api::new().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let repo = api.model("ResembleAI/chatterbox-turbo".to_string());

        let ve_path = repo.get("ve.safetensors").map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let t3_path = repo.get("t3_turbo_v1.safetensors").map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let s3gen_path = repo.get("s3gen_meanflow.safetensors").map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        Self::from_local(ve_path, t3_path, s3gen_path, device)
    }

    pub fn from_local(ve_path: PathBuf, t3_path: PathBuf, s3gen_path: PathBuf, device: Device) -> Result<Self> {
        let vb_ve = unsafe { VarBuilder::from_mmaped_safetensors(&[ve_path], candle_core::DType::F32, &device)? };
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

        let vb_t3 = unsafe { VarBuilder::from_mmaped_safetensors(&[t3_path], candle_core::DType::F32, &device)? };
        let t3 = T3::new(t3_config, vb_t3)?;

        let vb_s3 = unsafe { VarBuilder::from_mmaped_safetensors(&[s3gen_path], candle_core::DType::F32, &device)? };
        let s3gen = S3Gen::new(vb_s3, true)?; // meanflow=True

        Ok(Self {
            voice_encoder,
            t3,
            s3gen,
            device,
        })
    }

    pub fn generate(&self, text_tokens: &Tensor, ref_mels: &Tensor, max_len: usize) -> Result<Tensor> {
        let spk_emb = self.voice_encoder.forward(ref_mels)?;

        let speech_tokens = self.t3.generate(text_tokens, &spk_emb, None, None, max_len)?;

        let wav = self.s3gen.forward(&speech_tokens, None)?;
        Ok(wav)
    }
}
