//! HiFTGenerator Vocoder - converts mel spectrograms to audio waveforms.
//!
//! Aligned with Python implementation: chatterbox/models/s3gen/hifigan.py

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
            sampling_rate: 24000,
            upsample_rates: vec![8, 5, 3],
            upsample_kernel_sizes: vec![16, 11, 7],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            n_fft: 16,
            hop_len: 4,
        }
    }
}

/// Helper to load WeightNorm Conv1d
fn load_wn_conv1d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_v = vb.get((out_c, in_c, k), "parametrizations.weight.original1")?;
    let weight_g = vb.get((out_c, 1, 1), "parametrizations.weight.original0")?;
    let bias = vb.get(out_c, "bias")?;

    // weight = g * v / ||v||
    let norm = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_div(&norm)?.broadcast_mul(&weight_g)?;

    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// Helper to load WeightNorm ConvTranspose1d
fn load_wn_conv_transpose1d(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight_v = vb.get((in_c, out_c, k), "parametrizations.weight.original1")?;
    let weight_g = vb.get((in_c, 1, 1), "parametrizations.weight.original0")?;
    let bias = vb.get(out_c, "bias")?;

    // weight = g * v / ||v|| (norm over out_channels and kernel)
    let norm = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_div(&norm)?.broadcast_mul(&weight_g)?;

    Ok(ConvTranspose1d::new(weight, Some(bias), cfg))
}

/// Snake activation function: x + (1/a) * sin²(ax)
struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(channels, "alpha")?;
        Ok(Self {
            alpha: alpha.reshape((1, channels, 1))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = (&sin_ax * &sin_ax)?;
        let alpha_plus_eps = self
            .alpha
            .broadcast_add(&Tensor::new(1e-9f32, x.device())?)?;
        let inv_alpha = alpha_plus_eps.recip()?;
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
            convs1.push(load_wn_conv1d(
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding,
                    dilation,
                    ..Default::default()
                },
                vb.pp(format!("convs1.{}", i)),
            )?);
            activations1.push(Snake::new(channels, vb.pp(format!("activations1.{}", i)))?);

            convs2.push(load_wn_conv1d(
                channels,
                channels,
                kernel_size,
                Conv1dConfig {
                    padding: kernel_size / 2,
                    ..Default::default()
                },
                vb.pp(format!("convs2.{}", i)),
            )?);
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
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.activations1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.activations2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

/// ConvRNNF0Predictor
struct F0Predictor {
    condnet_convs: Vec<Conv1d>,
    classifier: Linear,
}

impl F0Predictor {
    fn new(in_channels: usize, cond_channels: usize, vb: VarBuilder) -> Result<Self> {
        let mut condnet_convs = Vec::new();
        let condnet_vb = vb.pp("condnet");

        // 5 blocks of (WN-Conv1d + ELU)
        for i in 0..5 {
            let in_c = if i == 0 { in_channels } else { cond_channels };
            condnet_convs.push(load_wn_conv1d(
                in_c,
                cond_channels,
                3,
                Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
                condnet_vb.pp(i * 2), // Index 0, 2, 4, 6, 8
            )?);
        }

        let classifier = candle_nn::linear(cond_channels, 1, vb.pp("classifier"))?;

        Ok(Self {
            condnet_convs,
            classifier,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for conv in &self.condnet_convs {
            x = conv.forward(&x)?;
            x = x.elu(1.0)?;
        }
        x = x.transpose(1, 2)?;
        let x = self.classifier.forward(&x)?;
        x.abs()?.squeeze(2)
    }
}

/// Sine generator for harmonic source
struct SineGen {
    harmonic_num: usize,
    sine_amp: f32,
    sampling_rate: u32,
    voiced_threshold: f32,
}

impl SineGen {
    fn new(
        sampling_rate: u32,
        harmonic_num: usize,
        sine_amp: f32,
        _noise_std: f32,
        voiced_threshold: f32,
    ) -> Self {
        Self {
            harmonic_num,
            sine_amp,
            sampling_rate,
            voiced_threshold,
        }
    }

    fn forward(&self, f0: &Tensor, upsmp_rate: usize) -> Result<(Tensor, Tensor)> {
        let device = f0.device();
        let (b, _, t_mel) = f0.dims3()?;
        let t_audio = t_mel * upsmp_rate;

        // F0 upsampling (nearest)
        let f0_upsampled = f0
            .unsqueeze(3)?
            .repeat((1, 1, 1, upsmp_rate))?
            .reshape((b, 1, t_audio))?;

        let mut f_mat_slices = Vec::new();
        for i in 0..=self.harmonic_num {
            let f = (&f0_upsampled * ((i + 1) as f64 / self.sampling_rate as f64))?;
            f_mat_slices.push(f);
        }
        let f_mat = Tensor::cat(&f_mat_slices, 1)?; // (B, H+1, T_audio)

        // theta = 2 * PI * (cumsum(f_mat) % 1)
        let f_mat_vec = f_mat.to_vec3::<f32>()?;
        let mut cumsum_data = vec![0.0f32; b * (self.harmonic_num + 1) * t_audio];
        for b_idx in 0..b {
            for h_idx in 0..=self.harmonic_num {
                let mut sum = 0.0;
                for t_idx in 0..t_audio {
                    sum += f_mat_vec[b_idx][h_idx][t_idx];
                    sum %= 1.0;
                    cumsum_data
                        [b_idx * (self.harmonic_num + 1) * t_audio + h_idx * t_audio + t_idx] = sum;
                }
            }
        }
        let cumsum = Tensor::from_vec(cumsum_data, f_mat.dims(), device)?;
        let phase = (cumsum * (2.0 * std::f32::consts::PI as f64))?;

        // Random initial phase for harmonics
        let rand_shape = (b, self.harmonic_num + 1, 1);
        let mut phase_vec_data = vec![0.0f32; b * (self.harmonic_num + 1)];
        for batch_idx in 0..b {
            // phase_vec[:, 0, :] = 0 (fundamental)
            phase_vec_data[batch_idx * (self.harmonic_num + 1)] = 0.0;
            for h in 1..=self.harmonic_num {
                phase_vec_data[batch_idx * (self.harmonic_num + 1) + h] =
                    (rand::random::<f32>() * 2.0 - 1.0) * std::f32::consts::PI;
            }
        }
        let phase_vec = Tensor::from_vec(phase_vec_data, rand_shape, device)?;
        let sine_waves = ((phase.broadcast_add(&phase_vec))?.sin()? * self.sine_amp as f64)?;

        let uv = f0_upsampled
            .gt(self.voiced_threshold as f64)?
            .to_dtype(DType::F32)?;

        // noise
        let noise_amp = {
            let voiced_part = (&uv * 0.003/* noise_std */)?;
            let unvoiced_part = ((Tensor::new(1.0f32, device)?.broadcast_as(uv.dims())? - &uv)?
                * (self.sine_amp / 3.0) as f64)?;
            (&voiced_part + &unvoiced_part)?
        };
        let noise =
            (Tensor::randn(0.0f32, 1.0, sine_waves.dims(), device)?.broadcast_mul(&noise_amp))?;

        let sine_waves = (sine_waves.broadcast_mul(&uv)?.broadcast_add(&noise))?;

        Ok((sine_waves, uv))
    }
}

/// Source module (SourceModuleHnNSF)
struct SourceModule {
    sine_gen: SineGen,
    l_linear: Linear,
}

impl SourceModule {
    fn new(
        sampling_rate: u32,
        harmonic_num: usize,
        sine_amp: f32,
        noise_std: f32,
        voiced_threshold: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen::new(
            sampling_rate,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
        );
        let l_linear = candle_nn::linear(harmonic_num + 1, 1, vb.pp("l_linear"))?;
        Ok(Self { sine_gen, l_linear })
    }

    fn forward(&self, f0: &Tensor, upsmp_rate: usize) -> Result<(Tensor, Tensor, Tensor)> {
        // f0: (B, T, 1) -> transpose to (B, 1, T)
        let f0_t = f0.transpose(1, 2)?;
        let (sine_wavs, uv) = self.sine_gen.forward(&f0_t, upsmp_rate)?;

        // sine_merge = tanh(linear(sine_wavs))
        let sine_wavs_t = sine_wavs.transpose(1, 2)?;
        let sine_merge = self
            .l_linear
            .forward(&sine_wavs_t)?
            .transpose(1, 2)?
            .tanh()?;

        // noise branch: same shape as uv
        let noise = (Tensor::randn(0.0f32, 1.0, uv.dims(), uv.device())?
            * (self.sine_gen.sine_amp / 3.0) as f64)?;

        Ok((sine_merge, noise, uv))
    }
}

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
        eprintln!(
            "[HiFTGenerator::new] START loading from {}",
            vb.prefix().to_string()
        );
        eprintln!(
            "[HiFTGenerator::new] in_ch={}, base_ch={}, n_fft={}",
            config.in_channels, config.base_channels, config.n_fft
        );

        let f0_predictor = F0Predictor::new(config.in_channels, 512, vb.pp("f0_predictor"))?;
        eprintln!("[HiFTGenerator::new] f0_predictor loaded");

        let _upsample_scale: usize =
            config.upsample_rates.iter().product::<usize>() * config.hop_len;
        let source_module = SourceModule::new(
            config.sampling_rate,
            config.nb_harmonics,
            0.1,   // sine_amp
            0.003, // noise_std
            10.0,  // voiced_threshold
            vb.pp("m_source"),
        )?;

        let conv_pre = load_wn_conv1d(
            config.in_channels,
            config.base_channels,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;

        let mut ups = Vec::new();
        for (i, (&rate, &kernel)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_ch = config.base_channels / (1 << i);
            let out_ch = config.base_channels / (1 << (i + 1));
            ups.push(load_wn_conv_transpose1d(
                in_ch,
                out_ch,
                kernel,
                ConvTranspose1dConfig {
                    stride: rate,
                    padding: (kernel - rate) / 2,
                    ..Default::default()
                },
                vb.pp(format!("ups.{}", i)),
            )?);
        }

        let mut source_downs = Vec::new();
        let mut source_resblocks = Vec::new();

        let factors = vec![15, 3, 1]; // Derived from [15, 3, 1] which is cumprod([1, 3, 5])[::-1]
        for (i, (&kernel, dilations)) in [7, 7, 11]
            .iter()
            .zip([vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]].iter())
            .enumerate()
        {
            let u = factors[i];
            let ch = config.base_channels / (1 << (i + 1));

            // source_downs are regular Conv1d in Python
            let cfg = if u == 1 {
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    ..Default::default()
                }
            } else {
                Conv1dConfig {
                    padding: u / 2,
                    stride: u,
                    ..Default::default()
                }
            };
            let k_size = if u == 1 { 1 } else { u * 2 };

            source_downs.push(candle_nn::conv1d(
                config.n_fft + 2,
                ch,
                k_size,
                cfg,
                vb.pp(format!("source_downs.{}", i)),
            )?);
            eprintln!(
                "[HiFTGenerator::new] source_downs.{} loaded (ch={}, k={}, u={})",
                i, ch, k_size, u
            );

            source_resblocks.push(ResBlock::new(
                ch,
                kernel,
                dilations,
                vb.pp(format!("source_resblocks.{}", i)),
            )?);
            eprintln!("[HiFTGenerator::new] source_resblocks.{} loaded", i);
        }

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

        let final_ch = config.base_channels / (1 << ups.len());
        let conv_post = load_wn_conv1d(
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

    pub fn inference(&self, mel: &Tensor) -> Result<Tensor> {
        let f0 = self.f0_predictor.forward(mel)?; // (B, T_mel, 1)

        let upsample_factor =
            self.config.upsample_rates.iter().product::<usize>() * self.config.hop_len;

        let (sine_merge, noise, _uv) = self.source_module.forward(&f0, upsample_factor)?;
        let source = (sine_merge + noise)?;

        let audio = self.decode(mel, &source)?;
        Ok(audio)
    }

    fn decode(&self, mel: &Tensor, source: &Tensor) -> Result<Tensor> {
        // source_squeezed: (B, 1, T) -> (B, T)
        let (s_real, s_imag) =
            simple_stft(&source.squeeze(1)?, self.config.n_fft, self.config.hop_len)?;
        let s_stft = Tensor::cat(&[&s_real, &s_imag], 1)?;

        let mut x = self.conv_pre.forward(mel)?;
        let num_kernels = self.config.resblock_kernel_sizes.len();

        for i in 0..self.ups.len() {
            x = candle_nn::ops::leaky_relu(&x, 0.1)?;
            x = self.ups[i].forward(&x)?;

            if i == self.ups.len() - 1 {
                // ReflectionPad1d((1, 0)) - adds 1 sample at the beginning
                let pad_val = x.narrow(2, 0, 1)?;
                x = Tensor::cat(&[pad_val, x], 2)?;
            }

            // Fusion
            let si = self.source_downs[i].forward(&s_stft)?;
            let si = self.source_resblocks[i].forward(&si)?;

            // Alignment: crop or pad si to match x
            let target_len = x.dim(2)?;
            let current_len = si.dim(2)?;
            let si = if current_len > target_len {
                si.narrow(2, 0, target_len)?
            } else if current_len < target_len {
                si.pad_with_zeros(2, 0, target_len - current_len)?
            } else {
                si
            };
            x = (x + si)?;

            let mut xs: Option<Tensor> = None;
            for j in 0..num_kernels {
                let rb = &self.resblocks[i * num_kernels + j];
                let out = rb.forward(&x)?;
                xs = Some(match xs {
                    Some(acc) => (acc + out)?,
                    None => out,
                });
            }
            x = (xs.unwrap() / num_kernels as f64)?;
        }

        x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        x = self.conv_post.forward(&x)?;

        let half = (self.config.n_fft / 2) + 1;
        let magnitude = x.narrow(1, 0, half)?.exp()?;
        let phase = x.narrow(1, half, half)?.sin()?;

        let audio = simple_istft(&magnitude, &phase, self.config.n_fft, self.config.hop_len)?;
        audio.clamp(-0.99f32, 0.99f32)
    }
}

/// Real STFT implementation using realfft crate
fn simple_stft(x: &Tensor, n_fft: usize, hop_len: usize) -> Result<(Tensor, Tensor)> {
    use realfft::RealFftPlanner;
    let (b, t) = x.dims2()?;
    let device = x.device();

    if t < n_fft {
        let n_bins = n_fft / 2 + 1;
        return Ok((
            Tensor::zeros((b, n_bins, 1), DType::F32, device)?,
            Tensor::zeros((b, n_bins, 1), DType::F32, device)?,
        ));
    }
    let n_frames = (t - n_fft) / hop_len + 1;
    let n_bins = n_fft / 2 + 1;

    // Check for overflow or massive allocation - safety limit
    if n_frames > 2_000_000 {
        return Err(candle_core::Error::Msg(format!(
            "Too many STFT frames: {}",
            n_frames
        )));
    }

    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();
    let x_data = x.to_vec2::<f32>()?;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut all_real = Vec::with_capacity(b * n_bins * n_frames);
    let mut all_imag = Vec::with_capacity(b * n_bins * n_frames);

    for batch_data in &x_data {
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_len;
            let mut windowed: Vec<f32> = batch_data[start..start + n_fft]
                .iter()
                .zip(window.iter())
                .map(|(s, w)| s * w)
                .collect();
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut windowed, &mut spectrum)
                .map_err(|e| candle_core::Error::Msg(format!("FFT error: {:?}", e)))?;
            for c in &spectrum {
                all_real.push(c.re);
                all_imag.push(c.im);
            }
        }
    }
    let real = Tensor::from_vec(all_real, (b, n_frames, n_bins), device)?
        .permute((0, 2, 1))?
        .contiguous()?;
    let imag = Tensor::from_vec(all_imag, (b, n_frames, n_bins), device)?
        .permute((0, 2, 1))?
        .contiguous()?;
    Ok((real, imag))
}

/// Real iSTFT implementation using realfft crate with overlap-add synthesis
fn simple_istft(
    magnitude: &Tensor,
    phase: &Tensor,
    n_fft: usize,
    hop_len: usize,
) -> Result<Tensor> {
    use realfft::RealFftPlanner;
    use rustfft::num_complex::Complex;
    let (b, n_bins, n_frames) = magnitude.dims3()?;
    let device = magnitude.device();
    if n_frames == 0 {
        return Tensor::zeros((b, 1), DType::F32, device);
    }
    let out_len = (n_frames - 1) * hop_len + n_fft;
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();
    let mag_data = magnitude.to_vec3::<f32>()?;
    let phase_data = phase.to_vec3::<f32>()?;
    let mut planner = RealFftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    let mut all_output = Vec::with_capacity(b * out_len);

    for batch_idx in 0..b {
        let mut output = vec![0.0f32; out_len];
        let mut window_sum = vec![0.0f32; out_len];

        for frame_idx in 0..n_frames {
            let mut spectrum = ifft.make_input_vec();
            for i in 0..n_bins {
                let mag = mag_data[batch_idx][i][frame_idx].min(1e2);
                let ph = phase_data[batch_idx][i][frame_idx];
                let re = mag * ph.cos();
                let im = if i == 0 || i == n_bins - 1 {
                    0.0
                } else {
                    mag * ph.sin()
                };
                spectrum[i] = Complex::new(re, im);
            }

            let mut time_data = ifft.make_output_vec();
            ifft.process(&mut spectrum, &mut time_data)
                .map_err(|e| candle_core::Error::Msg(format!("FFT error: {:?}", e)))?;

            let norm = 1.0 / n_fft as f32;
            let start = frame_idx * hop_len;
            for (i, &sample) in time_data.iter().enumerate() {
                if start + i < out_len {
                    let w = window[i];
                    output[start + i] += sample * norm * w;
                    window_sum[start + i] += w * w;
                }
            }
        }

        for i in 0..out_len {
            if window_sum[i] > 1e-8 {
                output[i] /= window_sum[i];
            }
            all_output.push(output[i]);
        }
    }
    Tensor::from_vec(all_output, (b, 1, out_len), device)
}
