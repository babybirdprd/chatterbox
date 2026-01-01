pub mod voice_encoder;
pub mod gpt2;
pub mod t3_model;
pub mod modules;
pub mod s3gen;
pub mod chatterbox;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use chatterbox::ChatterboxTurboTTS;

fn main() -> anyhow::Result<()> {
    println!("Initializing Chatterbox Turbo...");

    let device = Device::Cpu;

    // Create random weights for testing
    let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

    let model = ChatterboxTurboTTS::new(vb, device.clone())?;

    println!("Model initialized successfully.");

    // Mock input
    let text_tokens = Tensor::zeros((1, 50), candle_core::DType::U32, &device)?;
    // ref_mels: (B, T, M) -> (1, 200, 80)
    let ref_mels = Tensor::randn(0f32, 1f32, (1, 200, 80), &device)?;

    println!("Running generation...");
    let audio = model.generate(&text_tokens, &ref_mels)?;

    println!("Generated audio shape: {:?}", audio.dims());
    println!("Success!");

    Ok(())
}
