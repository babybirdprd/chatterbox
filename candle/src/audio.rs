// candle/src/audio.rs
use candle_core::{DType, Device, Result, Tensor};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use realfft::RealFftPlanner;
use rubato::{FftFixedIn, Resampler};

use std::f32::consts::PI;
use std::path::Path;

pub const S3GEN_SR: u32 = 24000;
pub const S3_SR: u32 = 16000;

// --- IO & Resampling ---

pub fn load_wav<P: AsRef<Path>>(path: P) -> std::result::Result<(Vec<f32>, u32), String> {
    let reader = WavReader::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    // Convert to mono by averaging
    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

pub fn save_wav<P: AsRef<Path>>(
    path: P,
    samples: &[f32],
    sample_rate: u32,
) -> std::result::Result<(), String> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer =
        WavWriter::create(path, spec).map_err(|e| format!("Failed to create WAV: {}", e))?;
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((clamped * 32767.0) as i16)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;
    Ok(())
}

pub fn resample(
    samples: &[f32],
    from_sr: u32,
    to_sr: u32,
) -> std::result::Result<Vec<f32>, String> {
    if from_sr == to_sr {
        return Ok(samples.to_vec());
    }
    // Fixed chunk size for rubato
    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(from_sr as usize, to_sr as usize, chunk_size, 2, 1)
        .map_err(|e| format!("Failed to create resampler: {}", e))?;

    let mut output = Vec::new();
    let mut input_frames = samples.to_vec();

    // Pad to chunk size
    let remainder = input_frames.len() % chunk_size;
    if remainder != 0 {
        input_frames.extend(std::iter::repeat(0.0).take(chunk_size - remainder));
    }

    for chunk in input_frames.chunks(chunk_size) {
        let input = vec![chunk.to_vec()];
        let mut resampled = resampler
            .process(&input, None)
            .map_err(|e| format!("Resample error: {}", e))?;
        output.append(&mut resampled[0]);
    }
    Ok(output)
}

pub fn normalize_loudness(samples: &mut [f32], target_db: f32) {
    let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms > 1e-9 {
        let gain = 10.0f32
            .powf((target_db - 20.0 * rms.log10()) / 20.0)
            .min(10.0); // Cap gain to avoid exploding noise
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

// --- Mel Spectrogram Logic ---

#[derive(Debug, Clone, Copy)]
pub enum STFTMode {
    Magnitude, // |STFT|
    Power,     // |STFT|^2
}

#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
    pub stft_mode: STFTMode,
    pub center: bool, // Whether to use center padding (like torch.stft center=True)
    pub manual_pad: usize, // Manual reflect padding for S3Gen
    pub drop_last_bin: bool, // Drop last FFT bin (for S3Tokenizer)
}

impl MelConfig {
    /// Configuration for S3Gen / HiFiGAN / Matcha (24kHz)
    /// Expects Magnitude STFT + Natural Log
    pub fn for_s3gen() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 1920, // Explicit from matcha config
            hop_length: 480,
            win_length: 1920,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8000.0,
            stft_mode: STFTMode::Magnitude,
            center: false,   // S3Gen/Matcha uses center=False
            manual_pad: 720, // (1920 - 480) / 2
            drop_last_bin: false,
        }
    }

    /// Configuration for S3Tokenizer (16kHz)
    /// Expects Power STFT + Log10
    pub fn for_s3tokenizer() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 128,
            fmin: 0.0,
            fmax: 8000.0,
            stft_mode: STFTMode::Power,
            center: true, // S3Tokenizer uses center=True
            manual_pad: 0,
            drop_last_bin: true,
        }
    }

    /// Configuration for VoiceEncoder (16kHz)
    /// Expects Power STFT + Linear (No Log)
    pub fn for_voice_encoder() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            win_length: 400,
            n_mels: 40,
            fmin: 0.0,
            fmax: 8000.0,
            stft_mode: STFTMode::Power, // VoiceEncoder uses Power (mel_power=2.0)
            center: true,               // VoiceEncoder uses center=True
            manual_pad: 0,
            drop_last_bin: false,
        }
    }

    /// Configuration for CAMPPlus (16kHz)
    /// Expects Power or Mag? Python xvector uses Kaldi.fbank.
    /// Kaldi fbank usually uses Power.
    /// However, the Python code uses `torchaudio.compliance.kaldi.fbank`.
    /// Default `use_power=True`.
    pub fn for_campplus() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512, // Closest ^2 for 25ms window (400 samples)
            hop_length: 160,
            win_length: 400,
            n_mels: 80,
            fmin: 20.0,   // Standard Kaldi default
            fmax: 7600.0, // Standard Kaldi default (Nyquist - epsilon)
            stft_mode: STFTMode::Power,
            center: false, // Kaldi compliance typically uses no centering
            manual_pad: 0,
            drop_last_bin: false,
        }
    }
}

pub struct AudioProcessor;

impl AudioProcessor {
    pub fn compute_mel_spectrogram(
        samples: &[f32],
        device: &Device,
        config: &MelConfig,
    ) -> Result<Tensor> {
        let n_fft = config.n_fft;
        let hop_length = config.hop_length;
        let win_length = config.win_length;

        // 1. Windowing
        let window: Vec<f32> = (0..win_length)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / win_length as f32).cos()))
            .collect();

        // 2. Padding (Conditional based on config.center)
        let padded = if config.center {
            // Reflect padding - matches librosa/torch.stft(center=True)
            let pad = n_fft / 2;
            let mut buf = Vec::with_capacity(samples.len() + 2 * pad);

            // Reflect Left
            for i in (1..=pad).rev() {
                buf.push(samples.get(i).copied().unwrap_or(0.0));
            }
            buf.extend_from_slice(samples);
            // Reflect Right
            let n = samples.len();
            for i in 1..=pad {
                buf.push(
                    samples
                        .get(n.saturating_sub(1).saturating_sub(i))
                        .copied()
                        .unwrap_or(0.0),
                );
            }
            buf
        } else if config.manual_pad > 0 {
            let pad = config.manual_pad;
            let mut buf = Vec::with_capacity(samples.len() + 2 * pad);
            // Reflect Left
            for i in (1..=pad).rev() {
                buf.push(samples.get(i).copied().unwrap_or(0.0));
            }
            buf.extend_from_slice(samples);
            // Reflect Right
            let n = samples.len();
            for i in 1..=pad {
                buf.push(
                    samples
                        .get(n.saturating_sub(1).saturating_sub(i))
                        .copied()
                        .unwrap_or(0.0),
                );
            }
            buf
        } else {
            // No centering - matches torch.stft(center=False)
            samples.to_vec()
        };

        // 3. FFT
        let n_frames = (padded.len() - n_fft) / hop_length + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        // We only need n_fft/2 + 1 bins (unless we drop the last one)
        let n_bins = if config.drop_last_bin {
            n_fft / 2
        } else {
            n_fft / 2 + 1
        };
        let mut spectrogram = vec![vec![0.0f32; n_bins]; n_frames];

        for (frame_idx, frame_start) in (0..padded.len().saturating_sub(n_fft) + 1)
            .step_by(hop_length)
            .enumerate()
        {
            if frame_idx >= n_frames {
                break;
            }

            let mut frame_data = vec![0.0f32; n_fft];
            for i in 0..win_length {
                frame_data[i] = padded[frame_start + i] * window[i];
            }

            let mut spectrum = fft.make_output_vec();
            fft.process(&mut frame_data, &mut spectrum)
                .map_err(|e| candle_core::Error::Msg(format!("FFT error: {}", e)))?;

            for (i, bin) in spectrogram[frame_idx].iter_mut().enumerate() {
                let mag_sq = spectrum[i].norm_sqr(); // |X|^2
                match config.stft_mode {
                    STFTMode::Magnitude => *bin = (mag_sq + 1e-9).sqrt(),
                    STFTMode::Power => *bin = mag_sq,
                }
            }
        }

        // 4. Mel Filterbank (Slaney)
        let mel_filters = create_slaney_mel_filterbank(
            config.sample_rate,
            n_fft,
            config.n_mels,
            config.fmin,
            config.fmax,
            config.drop_last_bin,
        );

        let mut mel_spec = vec![vec![0.0f32; config.n_mels]; n_frames];

        for (t, frame) in spectrogram.iter().enumerate() {
            for (m, filter) in mel_filters.iter().enumerate() {
                let mut sum = 0.0f32;
                for (bin_idx, &weight) in filter.iter().enumerate() {
                    // Filter is sparse, but here represented densely or via indices.
                    // This implementation assumes filter matches n_bins size or we handle indices.
                    // For simplicity in this snippet, let's assume dense, but optimize in production loop.
                    if weight > 0.0 {
                        sum += weight * frame[bin_idx];
                    }
                }
                mel_spec[t][m] = sum;
            }
        }

        // Return (B=1, n_mels, n_frames) - Channel First
        let flat: Vec<f32> = mel_spec.into_iter().flatten().collect();
        Tensor::from_vec(flat, (1, n_frames, config.n_mels), device)?
            .permute((0, 2, 1))?
            .contiguous()?
            .to_dtype(DType::F32)
    }

    // Helper to log-process based on model requirements
    pub fn log_process(mel: &Tensor, config: &MelConfig) -> Result<Tensor> {
        // S3Gen: Natural Log, clamp 1e-5
        // S3Tokenizer: Log10, clamp 1e-10
        // VoiceEncoder: No log

        // Note: The input `mel` here is (B, C, T)

        if config.n_mels == 80 && config.sample_rate == 24000 {
            // S3Gen
            return mel.clamp(1e-5, f32::MAX)?.log();
        }

        if config.n_mels == 128 {
            // S3Tokenizer
            let log10_val = (10.0f32).ln();
            return mel.clamp(1e-10, f32::MAX)?.log()? / (log10_val as f64);
        }

        // VoiceEncoder (40 mels) - Linear (No log)
        if config.n_mels == 40 {
            return Ok(mel.clone());
        }

        Ok(mel.clone())
    }
}

/// Slaney Mel Scale Filterbank Implementation (Matches Librosa default)
fn create_slaney_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
    drop_last_bin: bool,
) -> Vec<Vec<f32>> {
    let f_min = fmin;
    let f_max = fmax;

    // Slaney frequencies
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - 0.0) / (200.0 / 3.0); // 15.0
    let step = (6.4f32).ln() / 27.0; // logstep

    // Hz to Mel
    let hz_to_mel = |f: f32| -> f32 {
        if f < min_log_hz {
            (f - 0.0) / (200.0 / 3.0)
        } else {
            min_log_mel + (f / min_log_hz).ln() / step
        }
    };

    // Mel to Hz
    let mel_to_hz = |m: f32| -> f32 {
        if m < min_log_mel {
            0.0 + (200.0 / 3.0) * m
        } else {
            min_log_hz * (step * (m - min_log_mel)).exp()
        }
    };

    let min_mel = hz_to_mel(f_min);
    let max_mel = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity(n_mels + 2);
    for i in 0..(n_mels + 2) {
        mel_points.push(mel_to_hz(
            min_mel + (i as f32 * (max_mel - min_mel) / (n_mels as f32 + 1.0)),
        ));
    }

    let n_bins = if drop_last_bin {
        n_fft / 2
    } else {
        n_fft / 2 + 1
    };
    let mut filters = vec![vec![0.0f32; n_bins]; n_mels];
    let fft_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    for i in 0..n_mels {
        let f_left = mel_points[i];
        let f_center = mel_points[i + 1];
        let f_right = mel_points[i + 2];

        // Slaney Area Normalization
        // 2.0 / (f_right - f_left)
        let enorm = 2.0 / (f_right - f_left);

        for j in 0..n_bins {
            let f = fft_freqs[j];
            let mut weight = 0.0f32;

            if f >= f_left && f <= f_center {
                weight = (f - f_left) / (f_center - f_left);
            } else if f > f_center && f <= f_right {
                weight = (f_right - f) / (f_right - f_center);
            }

            filters[i][j] = weight * enorm;
        }
    }
    filters
}
