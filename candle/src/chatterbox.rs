use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;
use crate::voice_encoder::{VoiceEncoder, VoiceEncoderConfig};
use crate::t3_model::{T3, T3Config};
use crate::s3gen::S3Gen;

pub struct ChatterboxTurboTTS {
    voice_encoder: VoiceEncoder,
    t3: T3,
    s3gen: S3Gen,
    device: Device,
}

impl ChatterboxTurboTTS {
    pub fn new(vb: VarBuilder, device: Device) -> Result<Self> {
        let ve_config = VoiceEncoderConfig::default();
        let voice_encoder = VoiceEncoder::new(ve_config, vb.pp("ve"))?;

        let t3_config = T3Config::default();
        let t3 = T3::new(t3_config, vb.pp("t3"))?;

        let s3gen = S3Gen::new(vb.pp("s3gen"))?;

        Ok(Self {
            voice_encoder,
            t3,
            s3gen,
            device,
        })
    }

    pub fn generate(&self, text_tokens: &Tensor, ref_mels: &Tensor) -> Result<Tensor> {
        // 1. Voice Encoder
        // ref_mels: (B, T, M)
        let spk_emb = self.voice_encoder.forward(ref_mels)?; // (B, E)

        // 2. T3 (Text -> Speech Tokens)
        // Need to implement T3 inference loop.

        let speech_tokens = self.t3.generate(text_tokens, &spk_emb, 10)?;

        // 3. S3Gen (Speech Tokens -> Wav)
        // S3Gen needs speech tokens and speaker embedding (or ref wav embedding).
        // For now, S3Gen only takes speech tokens in current signature.
        // We should likely pass spk_emb or ref_mels to S3Gen as well if it supports it,
        // but looking at s3gen.rs, `forward` only takes `speech_tokens`.
        // The original python s3gen takes ref_dict (embedding).
        // I will stick to current s3gen.rs interface which takes just speech_tokens,
        // but note that real implementation probably needs conditioning.

        let wav = self.s3gen.forward(&speech_tokens)?;

        Ok(wav)
    }
}
