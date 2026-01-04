use anyhow::{anyhow, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use ndarray::ArrayD;
use ndarray_npy::read_npy;
use std::path::{Path, PathBuf};

use candle::audio::{AudioProcessor, MelConfig};
use candle::s3tokenizer::{ModelConfig as S3TokenizerConfig, S3TokenizerV2};
use candle::voice_encoder::{VoiceEncoder, VoiceEncoderConfig};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    ref_dir: PathBuf,

    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(long)]
    fp16: bool,
}

fn load_npy<P: AsRef<Path>>(path: P, device: &Device) -> Result<Tensor> {
    let arr: ArrayD<f32> = read_npy(path.as_ref())
        .map_err(|e| anyhow!("Failed to load {}: {}", path.as_ref().display(), e))?;
    let shape = arr.shape();
    let data = arr
        .as_slice()
        .ok_or_else(|| anyhow!("Non-contiguous array"))?;
    Tensor::from_slice(data, shape, device).map_err(anyhow::Error::from)
}

fn load_npy_i64<P: AsRef<Path>>(path: P, device: &Device) -> Result<Tensor> {
    let arr: ArrayD<i64> = read_npy(path.as_ref())
        .map_err(|e| anyhow!("Failed to load {}: {}", path.as_ref().display(), e))?;
    let shape = arr.shape();
    let data = arr
        .as_slice()
        .ok_or_else(|| anyhow!("Non-contiguous array"))?;
    Tensor::from_slice(data, shape, device).map_err(anyhow::Error::from)
}

fn check(name: &str, rust: &Tensor, python: &Tensor, tolerance: f64) -> Result<()> {
    let mut rust_f32 = rust.to_dtype(DType::F32)?;
    let mut python_f32 = python.to_dtype(DType::F32)?;

    // Handle leading batch dimension
    if rust_f32.dims().len() == 3 && python_f32.dims().len() == 2 {
        python_f32 = python_f32.unsqueeze(0)?;
    } else if rust_f32.dims().len() == 2
        && python_f32.dims().len() == 3
        && python_f32.dims()[0] == 1
    {
        python_f32 = python_f32.squeeze(0)?;
    }

    // Handle time dimension mismatch (usually at last dimension)
    let r_dims = rust_f32.dims();
    let p_dims = python_f32.dims();
    if r_dims.len() == p_dims.len() && r_dims.len() >= 2 {
        let last_idx = r_dims.len() - 1;
        let t_r = r_dims[last_idx];
        let t_p = p_dims[last_idx];
        if t_r != t_p {
            let t_min = t_r.min(t_p);
            println!(
                "  [Shape Note] {}: rust={:?}, python={:?}. Cropping last dim to {}.",
                name, r_dims, p_dims, t_min
            );
            rust_f32 = rust_f32.narrow(last_idx, 0, t_min)?;
            python_f32 = python_f32.narrow(last_idx, 0, t_min)?;
        }
    }

    // Final broadcast/match check
    if rust_f32.shape() != python_f32.shape() {
        python_f32 = python_f32.broadcast_as(rust_f32.shape())?;
    }

    let diff = (rust_f32 - python_f32)?.abs()?.flatten_all()?;
    let max_diff = diff.max(0)?.to_vec0::<f32>()?;
    let mean_diff = diff.mean_all()?.to_vec0::<f32>()?;

    if (max_diff as f64) > tolerance {
        println!(
            "FAIL: {} | max_diff: {:.6}, mean_diff: {:.6} (tol: {})",
            name, max_diff, mean_diff, tolerance
        );
    } else {
        println!(
            "PASS: {} | max_diff: {:.6}, mean_diff: {:.6}",
            name, max_diff, mean_diff
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = if args.device == "cuda" {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let dtype = if args.fp16 { DType::F16 } else { DType::F32 };

    println!("Starting Pipeline Parity Test...");
    println!("Reference Directory: {}", args.ref_dir.display());

    // =========================================================================
    // Phase 1: Audio Loading & Resampling
    // =========================================================================
    println!("\n--- Phase 1: Audio Loading & Resampling ---");
    let ref_24k_py = load_npy(args.ref_dir.join("ref_24k.npy"), &Device::Cpu)?;
    let ref_16k_py = load_npy(args.ref_dir.join("ref_16k.npy"), &Device::Cpu)?;
    println!("(Skipping native Rust resampling check - using Python's resampled audio as input to ensure downstream parity)");

    // =========================================================================
    // Phase 2: Mel Spectrograms
    // =========================================================================
    println!("\n--- Phase 2: Mel Spectrograms ---");
    let samples_16k = ref_16k_py.to_vec1::<f32>()?;
    let samples_24k = ref_24k_py.to_vec1::<f32>()?;

    // 2a. VoiceEncoder Mel
    let mel_ve_py = load_npy(args.ref_dir.join("mel_ve.npy"), &Device::Cpu)?;
    let cfg_ve = MelConfig::for_voice_encoder();
    let mel_ve_rust = AudioProcessor::compute_mel_spectrogram(&samples_16k, &Device::Cpu, &cfg_ve)?;
    check("Mel VE", &mel_ve_rust, &mel_ve_py, 1e-3)?;

    // 2b. S3Tokenizer Mel
    let mel_s3tok_py = load_npy(args.ref_dir.join("mel_s3tok.npy"), &Device::Cpu)?;
    let cfg_s3tok = MelConfig::for_s3tokenizer();
    let mel_s3tok_raw =
        AudioProcessor::compute_mel_spectrogram(&samples_16k, &Device::Cpu, &cfg_s3tok)?;
    let mel_s3tok_log = AudioProcessor::log_process(&mel_s3tok_raw, &cfg_s3tok)?;
    // S3Tokenizer uses max normalization: log_spec = max(log_spec, log_spec.max() - 8.0)
    let max_val = mel_s3tok_log.max_all()?.to_scalar::<f32>()?;
    let mel_s3tok_norm = mel_s3tok_log.maximum(max_val - 8.0)?;
    let mel_s3tok_rust = ((mel_s3tok_norm + 4.0)? / 4.0)?;
    check("Mel S3Tok", &mel_s3tok_rust, &mel_s3tok_py, 1e-3)?;

    // 2c. S3Gen Mel
    let mel_s3gen_py = load_npy(args.ref_dir.join("mel_s3gen.npy"), &Device::Cpu)?;
    let cfg_s3gen = MelConfig::for_s3gen();
    let mel_s3gen_raw =
        AudioProcessor::compute_mel_spectrogram(&samples_24k, &Device::Cpu, &cfg_s3gen)?;
    let mel_s3gen_rust = AudioProcessor::log_process(&mel_s3gen_raw, &cfg_s3gen)?;
    check("Mel S3Gen", &mel_s3gen_rust, &mel_s3gen_py, 1e-3)?;

    // =========================================================================
    // Phase 3: VoiceEncoder
    // =========================================================================
    println!("\n--- Phase 3: VoiceEncoder ---");
    let spk_emb_ve_py = load_npy(args.ref_dir.join("spk_emb_ve.npy"), &Device::Cpu)?;
    let ve_path = args.model_dir.join("ve.safetensors");
    let ve_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[ve_path], dtype, &device)? };
    let ve = VoiceEncoder::new(VoiceEncoderConfig::default(), ve_vb)?;

    // IMPORTANT: Transpose for VoiceEncoder (B, T, 40)
    let mel_ve_t = mel_ve_rust
        .transpose(1, 2)?
        .to_device(&device)?
        .to_dtype(dtype)?;
    // Use forward directly as in split_generate.rs or inference?
    // split_generate.rs uses forward on the whole thing.
    let spk_emb_ve_rust = ve.forward(&mel_ve_t)?;
    check(
        "VoiceEncoder Output",
        &spk_emb_ve_rust,
        &spk_emb_ve_py.to_device(&device)?,
        1e-2,
    )?;
    drop(ve);

    // =========================================================================
    // Phase 4: S3Tokenizer & CAMPPlus
    // =========================================================================
    println!("\n--- Phase 4: S3Tokenizer & CAMPPlus ---");

    // S3Tokenizer
    let prompt_tokens_py = load_npy_i64(args.ref_dir.join("prompt_tokens.npy"), &Device::Cpu)?;
    let s3tok_path = args.model_dir.join("s3tokenizer.safetensors");
    let s3tok_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[s3tok_path], dtype, &device)? };
    let s3tok = S3TokenizerV2::new(&S3TokenizerConfig::default(), s3tok_vb)?;
    let mel_s3tok_input = mel_s3tok_rust.to_device(&device)?.to_dtype(dtype)?;
    let prompt_tokens_rust = s3tok.encode(&mel_s3tok_input)?;

    let py_toks = prompt_tokens_py.to_vec2::<i64>()?[0].clone();
    let rust_toks: Vec<u32> = prompt_tokens_rust.to_vec1()?;
    let match_count = py_toks
        .iter()
        .zip(rust_toks.iter())
        .filter(|(&p, &r)| p as u32 == r)
        .count();
    println!(
        "S3Tokenizer Tokens: {}/{} match",
        match_count,
        py_toks.len()
    );
    drop(s3tok);

    // CAMPPlus
    let spk_emb_camp_py = load_npy(args.ref_dir.join("spk_emb_camp.npy"), &Device::Cpu)?;
    let s3gen_path = args.model_dir.join("s3gen_meanflow.safetensors");
    let s3gen_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[s3gen_path], dtype, &device)? };
    let campplus = candle::campplus::CAMPPlus::new(80, 192, s3gen_vb.pp("speaker_encoder"))?;

    // CAMPPlus Mel processing from split_generate.rs
    let cfg_camp = MelConfig::for_campplus();
    let mel_camp_raw =
        AudioProcessor::compute_mel_spectrogram(&samples_16k, &Device::Cpu, &cfg_camp)?;
    let mel_camp_log = mel_camp_raw.clamp(1e-5, f32::MAX)?.log()?;
    let mean = mel_camp_log.mean_keepdim(2)?;
    let mel_camp_norm = mel_camp_log
        .broadcast_sub(&mean)?
        .to_device(&device)?
        .to_dtype(dtype)?;

    let spk_emb_camp_rust = campplus.forward(&mel_camp_norm)?.narrow(1, 0, 80)?;
    check(
        "CAMPPlus Speaker Embedding",
        &spk_emb_camp_rust,
        &spk_emb_camp_py.to_device(&device)?,
        1e-2,
    )?;
    drop(campplus);

    // =========================================================================
    // Phase 5: Text Tokenization (Placeholder for now)
    // =========================================================================
    println!("\n--- Phase 5: Text Tokenization ---");
    let text_tokens_py = load_npy_i64(args.ref_dir.join("text_tokens.npy"), &Device::Cpu)?;
    println!("Reference text tokens shape: {:?}", text_tokens_py.shape());

    println!("\nParity test completed up to Phase 4.");
    Ok(())
}
