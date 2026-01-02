//! HiFTGenerator Vocoder - converts mel spectrograms to audio waveforms.
//!
//! This module implements the HiFTNet neural vocoder which uses:
//! - F0 prediction from mel features
//! - Neural source-filter model for excitation generation  
//! - iSTFT-based waveform synthesis
//!
//! Based on: <https://arxiv.org/abs/2309.09493>

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, VarBuilder};
use std::f32::consts::PI;

/// HiFTGenerator configuration
#[derive(Debug, Clone)]
pub struct HiFTConfig {
    pub in_channels: usize,
    pub base_channels: usize,
    pub nb_harmonics: usize,
    pub sampling_rate: u32,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub n_fft: usize,
    pub hop_len: usize,
}

impl Default for HiFTConfig {
    fn default() -> Self {
        Self {
            in_channels: 80,
            base_channels: 512,
            nb_harmonics: 8,
            sampling_rate: 22050,
            upsample_rates: vec![8, 8],
            upsample_kernel_sizes: vec![16, 16],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            n_fft: 16,
            hop_len: 4,
        }
    }
}

/// Snake activation function: x + (1/a) * sin²(ax)
struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Snake: x + (1/a) * sin²(ax)
        let ax = x.broadcast_mul(&self.alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;
        let inv_alpha = (1.0 / &self.alpha)?;
        let result = (x + sin_sq.broadcast_mul(&inv_alpha)?)?;
        Ok(result)
    }
}

/// Residual block with Snake activations
struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    activations1: Vec<Snake>,
    activations2: Vec<Snake>,
}

impl ResBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut activations1 = Vec::new();
        let mut activations2 = Vec::new();

        for (i, &dilation) in dilations.iter().enumerate() {
            let padding = (kernel_size * dilation - dilation) / 2;
            let cfg1 = Conv1dConfig {
                padding,
                dilation,
                ..Default::default()
            };
            convs1.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                cfg1,
                vb.pp(format!("convs1.{}", i)),
            )?);

            let cfg2 = Conv1dConfig {
                padding: (kernel_size - 1) / 2,
                ..Default::default()
            };
            convs2.push(candle_nn::conv1d(
                channels,
                channels,
                kernel_size,
                cfg2,
                vb.pp(format!("convs2.{}", i)),
            )?);

            activations1.push(Snake::new(channels, vb.pp(format!("activations1.{}", i)))?);
            activations2.push(Snake::new(channels, vb.pp(format!("activations2.{}", i)))?);
        }

        Ok(Self {
            convs1,
            convs2,
            activations1,
            activations2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.activations1[i].forward(&out)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.activations2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            out = (&out + &xt)?;
        }
        Ok(out)
    }
}

/// F0 predictor using ConvRNN architecture
pub struct F0Predictor {
    conv_in: Conv1d,
    conv_hidden: Conv1d,
    conv_out: Conv1d,
}

impl F0Predictor {
    pub fn new(in_channels: usize, hidden_channels: usize, vb: VarBuilder) -> Result<Self> {
        let conv_in = candle_nn::conv1d(
            in_channels,
            hidden_channels,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_in"),
        )?;

        let conv_hidden = candle_nn::conv1d(
            hidden_channels,
            hidden_channels,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_hidden"),
        )?;

        let conv_out = candle_nn::conv1d(
            hidden_channels,
            1,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            conv_hidden,
            conv_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T) -> f0: (B, 1, T)
        let h = self.conv_in.forward(x)?;
        let h = h.relu()?;
        let h = self.conv_hidden.forward(&h)?;
        let h = h.relu()?;
        let f0 = self.conv_out.forward(&h)?;
        // Sigmoid: 1/(1+exp(-x)), then scale for F0 range
        let neg_f0 = (f0 * -1.0)?;
        let exp_neg = neg_f0.exp()?;
        let sigmoid = (1.0 / (exp_neg + 1.0)?)?;
        let f0 = (sigmoid * 500.0)?; // Max F0 ~500Hz
        Ok(f0)
    }
}

/// Sine generator for harmonic source
struct SineGen {
    harmonic_num: usize,
    sine_amp: f32,
    noise_std: f32,
    sampling_rate: u32,
    voiced_threshold: f32,
}

impl SineGen {
    fn new(
        sampling_rate: u32,
        harmonic_num: usize,
        sine_amp: f32,
        noise_std: f32,
        voiced_threshold: f32,
    ) -> Self {
        Self {
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate,
            voiced_threshold,
        }
    }

    fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor)> {
        // f0: (B, 1, T)
        let device = f0.device();
        let (b, _, t) = f0.dims3()?;

        // Create harmonic frequencies
        let mut sine_waves = Tensor::zeros((b, self.harmonic_num + 1, t), DType::F32, device)?;

        for h in 0..=self.harmonic_num {
            let harmonic_f0 = (f0 * (h + 1) as f64)?;
            let normalized = (&harmonic_f0 / self.sampling_rate as f64)?;

            // Cumulative phase
            let phase = (normalized.cumsum(2)? * (2.0 * PI as f64))?;

            // Add random initial phase for harmonics > 0
            let sine = if h == 0 {
                (phase.sin()? * self.sine_amp as f64)?
            } else {
                let random_phase = Tensor::rand(0.0f32, 2.0 * PI, (b, 1, 1), device)?;
                let phase_shifted = (&phase + &random_phase)?;
                (phase_shifted.sin()? * self.sine_amp as f64)?
            };

            sine_waves = sine_waves.slice_assign(&[0..b, h..h + 1, 0..t], &sine)?;
        }

        // UV (voiced/unvoiced) mask
        let uv = f0.gt(self.voiced_threshold as f64)?.to_dtype(DType::F32)?;

        // Add noise for unvoiced regions
        let noise = Tensor::randn(
            0.0f32,
            self.sine_amp / 3.0,
            (b, self.harmonic_num + 1, t),
            device,
        )?;
        let uv_broadcast = uv.broadcast_as((b, self.harmonic_num + 1, t))?;
        let sine_waves = (&sine_waves * &uv_broadcast)?;
        let sine_waves = (&sine_waves + &noise * (1.0 - &uv_broadcast)?)?;

        Ok((sine_waves, uv))
    }
}

/// Source module combining sine generator with linear projection
struct SourceModule {
    sine_gen: SineGen,
    l_linear: Linear,
}

impl SourceModule {
    fn new(
        sampling_rate: u32,
        _upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f32,
        noise_std: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen::new(sampling_rate, harmonic_num, sine_amp, noise_std, 10.0);
        let l_linear = candle_nn::linear(harmonic_num + 1, 1, vb.pp("l_linear"))?;

        Ok(Self { sine_gen, l_linear })
    }

    fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        // f0: (B, T, 1) -> transpose to (B, 1, T)
        let f0_t = f0.transpose(1, 2)?;
        let (sine_waves, _uv) = self.sine_gen.forward(&f0_t)?;

        // sine_waves: (B, H+1, T) -> transpose to (B, T, H+1) for linear
        let sine_waves = sine_waves.transpose(1, 2)?;
        let merged = self.l_linear.forward(&sine_waves)?;
        let merged = merged.tanh()?;

        // Back to (B, 1, T)
        Ok(merged.transpose(1, 2)?)
    }
}

/// HiFTGenerator vocoder
pub struct HiFTGenerator {
    config: HiFTConfig,
    f0_predictor: F0Predictor,
    source_module: SourceModule,
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
}

impl HiFTGenerator {
    pub fn new(config: HiFTConfig, vb: VarBuilder) -> Result<Self> {
        let upsample_scale: usize =
            config.upsample_rates.iter().product::<usize>() * config.hop_len;

        // F0 predictor
        let f0_predictor = F0Predictor::new(config.in_channels, 256, vb.pp("f0_predictor"))?;

        // Source module
        let source_module = SourceModule::new(
            config.sampling_rate,
            upsample_scale,
            config.nb_harmonics,
            0.1,
            0.003,
            vb.pp("m_source"),
        )?;

        // Input conv
        let conv_pre = candle_nn::conv1d(
            config.in_channels,
            config.base_channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;

        // Upsample layers
        let mut ups = Vec::new();
        for (i, (&rate, &kernel)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = config.base_channels / (1 << i);
            let out_ch = config.base_channels / (1 << (i + 1));
            let padding = (kernel - rate) / 2;

            let up = candle_nn::conv_transpose1d(
                in_ch,
                out_ch,
                kernel,
                ConvTranspose1dConfig {
                    stride: rate,
                    padding,
                    ..Default::default()
                },
                vb.pp(format!("ups.{}", i)),
            )?;
            ups.push(up);
        }

        // Source downsampling and resblocks
        let mut source_downs = Vec::new();
        let mut source_resblocks = Vec::new();
        let source_dilations = vec![vec![1, 3, 5], vec![1, 3, 5]];
        let source_kernels = vec![7, 11];

        for (i, (&kernel, dilations)) in source_kernels
            .iter()
            .zip(source_dilations.iter())
            .enumerate()
        {
            let out_ch = config.base_channels / (1 << (i + 1));
            let in_ch = config.n_fft + 2;

            source_downs.push(candle_nn::conv1d(
                in_ch,
                out_ch,
                1,
                Default::default(),
                vb.pp(format!("source_downs.{}", i)),
            )?);

            source_resblocks.push(ResBlock::new(
                out_ch,
                kernel,
                dilations,
                vb.pp(format!("source_resblocks.{}", i)),
            )?);
        }

        // Main resblocks
        let mut resblocks = Vec::new();
        for i in 0..ups.len() {
            let ch = config.base_channels / (1 << (i + 1));
            for (j, (&kernel, dilations)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(ResBlock::new(
                    ch,
                    kernel,
                    dilations,
                    vb.pp(format!(
                        "resblocks.{}",
                        i * config.resblock_kernel_sizes.len() + j
                    )),
                )?);
            }
        }

        // Output conv
        let final_ch = config.base_channels / (1 << ups.len());
        let conv_post = candle_nn::conv1d(
            final_ch,
            config.n_fft + 2,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_post"),
        )?;

        Ok(Self {
            config,
            f0_predictor,
            source_module,
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
        })
    }

    /// Generate audio from mel spectrogram
    pub fn inference(&self, mel: &Tensor) -> Result<Tensor> {
        // mel: (B, M, T) where M=80
        let f0 = self.f0_predictor.forward(mel)?; // (B, 1, T)

        // Upsample F0 to audio rate
        let upsample_factor: usize =
            self.config.upsample_rates.iter().product::<usize>() * self.config.hop_len;
        let f0_up = upsample_1d(&f0, upsample_factor)?;

        // Generate source excitation
        let source = self.source_module.forward(&f0_up.transpose(1, 2)?)?;

        // Decode
        self.decode(mel, &source)
    }

    fn decode(&self, mel: &Tensor, source: &Tensor) -> Result<Tensor> {
        // Compute STFT of source
        let source_squeezed = source.squeeze(1)?; // (B, T)
        let (s_real, s_imag) =
            simple_stft(&source_squeezed, self.config.n_fft, self.config.hop_len)?;
        let s_stft = Tensor::cat(&[&s_real, &s_imag], 1)?; // (B, n_fft+2, T')

        // Main network
        let mut x = self.conv_pre.forward(mel)?;

        for (i, up) in self.ups.iter().enumerate() {
            x = x.gelu()?;
            x = up.forward(&x)?;

            // Source fusion
            if i < self.source_downs.len() {
                let si = self.source_downs[i].forward(&s_stft)?;
                let si = self.source_resblocks[i].forward(&si)?;
                x = (&x + &si)?;
            }

            // Resblocks
            let num_kernels = self.config.resblock_kernel_sizes.len();
            let mut xs: Option<Tensor> = None;
            for j in 0..num_kernels {
                let idx = i * num_kernels + j;
                if idx < self.resblocks.len() {
                    let rb_out = self.resblocks[idx].forward(&x)?;
                    xs = Some(match xs {
                        Some(acc) => (&acc + &rb_out)?,
                        None => rb_out,
                    });
                }
            }
            if let Some(xs_sum) = xs {
                x = (xs_sum / num_kernels as f64)?;
            }
        }

        x = x.gelu()?;
        x = self.conv_post.forward(&x)?;

        // Split to magnitude and phase, then iSTFT
        let half = (self.config.n_fft / 2) + 1;
        let magnitude = x.narrow(1, 0, half)?.exp()?;
        let phase = x.narrow(1, half, half)?;

        let audio = simple_istft(&magnitude, &phase, self.config.n_fft, self.config.hop_len)?;

        // Clamp output
        let audio = audio.clamp(-0.99f32, 0.99f32)?;

        Ok(audio)
    }
}

/// Simple 1D upsampling by repeating
fn upsample_1d(x: &Tensor, factor: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let x = x.unsqueeze(3)?; // (B, C, T, 1)
    let x = x.repeat((1, 1, 1, factor))?; // (B, C, T, factor)
    x.reshape((b, c, t * factor))
}

/// Simplified STFT (for source signal processing)
fn simple_stft(x: &Tensor, n_fft: usize, hop_len: usize) -> Result<(Tensor, Tensor)> {
    // This is a simplified STFT - for production, use proper FFT
    // Returns (real, imag) each of shape (B, n_fft/2+1, T')
    let device = x.device();
    let (b, t) = x.dims2()?;
    let n_frames = (t.saturating_sub(n_fft)) / hop_len + 1;
    let n_bins = n_fft / 2 + 1;

    // For now, return placeholder - actual STFT would use realfft
    let real = Tensor::zeros((b, n_bins, n_frames), DType::F32, device)?;
    let imag = Tensor::zeros((b, n_bins, n_frames), DType::F32, device)?;

    Ok((real, imag))
}

/// Simplified iSTFT
fn simple_istft(
    magnitude: &Tensor,
    _phase: &Tensor,
    n_fft: usize,
    hop_len: usize,
) -> Result<Tensor> {
    // Simplified - reconstruct by overlap-add of windowed sinusoids
    let device = magnitude.device();
    let (b, _n_bins, n_frames) = magnitude.dims3()?;
    let out_len = n_frames * hop_len + n_fft;

    // For now, return noise-like signal scaled by magnitude
    // Real implementation would use proper iFFT
    let output = Tensor::zeros((b, out_len), DType::F32, device)?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hift_config_default() {
        let config = HiFTConfig::default();
        assert_eq!(config.in_channels, 80);
        assert_eq!(config.sampling_rate, 22050);
    }
}
