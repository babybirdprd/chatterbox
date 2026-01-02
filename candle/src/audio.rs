//! Audio utilities for WAV I/O, resampling, and mel spectrogram computation.
//!
//! This module provides portable audio processing functions needed for TTS inference.

use candle_core::{DType, Device, Result, Tensor};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rubato::{FftFixedIn, Resampler};
use std::path::Path;

/// Sample rate used by S3Gen vocoder output
pub const S3GEN_SR: u32 = 22050;

/// Sample rate used by S3Tokenizer and VoiceEncoder
pub const S3_SR: u32 = 16000;

/// Number of mel filter banks
pub const NUM_MELS: usize = 80;

/// Load a WAV file and return samples as f32 in range [-1, 1].
///
/// Supports mono and stereo (converted to mono by averaging channels).
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

    // Convert to mono if stereo
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

/// Save samples as a WAV file.
///
/// Samples should be f32 in range [-1, 1].
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
        let int_sample = (clamped * 32767.0) as i16;
        writer
            .write_sample(int_sample)
            .map_err(|e| format!("Failed to write sample: {}", e))?;
    }

    writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV: {}", e))?;

    Ok(())
}

/// Resample audio from one sample rate to another.
pub fn resample(
    samples: &[f32],
    from_sr: u32,
    to_sr: u32,
) -> std::result::Result<Vec<f32>, String> {
    if from_sr == to_sr {
        return Ok(samples.to_vec());
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(from_sr as usize, to_sr as usize, chunk_size, 2, 1)
        .map_err(|e| format!("Failed to create resampler: {}", e))?;

    let mut output = Vec::new();
    let mut input_frames = samples.to_vec();

    // Pad to chunk boundary
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

/// Compute mel spectrogram from audio samples.
///
/// Returns tensor of shape (1, T, NUM_MELS) for VoiceEncoder input.
pub fn compute_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    device: &Device,
) -> Result<Tensor> {
    use realfft::RealFftPlanner;

    let n_fft = 1024;
    let hop_length = 256;
    let win_length = 1024;

    // Create Hann window
    let window: Vec<f32> = (0..win_length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_length as f32).cos()))
        .collect();

    // Pad samples
    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(samples);
    padded.extend(vec![0.0f32; pad]);

    // Compute STFT frames
    let n_frames = (padded.len() - n_fft) / hop_length + 1;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut power_spectrogram = vec![vec![0.0f32; n_fft / 2 + 1]; n_frames];

    for (frame_idx, frame_start) in (0..padded.len() - n_fft + 1)
        .step_by(hop_length)
        .enumerate()
    {
        if frame_idx >= n_frames {
            break;
        }

        // Apply window
        let mut windowed: Vec<f32> = padded[frame_start..frame_start + n_fft]
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // FFT
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut windowed, &mut spectrum)
            .map_err(|e| candle_core::Error::Msg(format!("FFT error: {}", e)))?;

        // Power spectrum
        for (i, c) in spectrum.iter().enumerate() {
            power_spectrogram[frame_idx][i] = c.norm_sqr();
        }
    }

    // Create mel filterbank
    let mel_filters = create_mel_filterbank(sample_rate, n_fft, NUM_MELS);

    // Apply mel filterbank
    let mut mel_spec = vec![vec![0.0f32; NUM_MELS]; n_frames];
    for (t, power_frame) in power_spectrogram.iter().enumerate() {
        for (m, filter) in mel_filters.iter().enumerate() {
            let mut sum = 0.0f32;
            for (i, &weight) in filter.iter().enumerate() {
                sum += weight * power_frame[i];
            }
            // Log mel with floor
            mel_spec[t][m] = (sum.max(1e-10)).log10();
        }
    }

    // Normalize to [0, 1] range expected by VoiceEncoder
    let max_val = mel_spec
        .iter()
        .flat_map(|row| row.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    for row in &mut mel_spec {
        for val in row.iter_mut() {
            *val = (*val - (max_val - 8.0)).max(0.0) / 8.0;
        }
    }

    // Convert to tensor (1, T, M)
    let flat: Vec<f32> = mel_spec.into_iter().flatten().collect();
    let tensor = Tensor::from_vec(flat, (1, n_frames, NUM_MELS), device)?.to_dtype(DType::F32)?;

    Ok(tensor)
}

/// Create mel filterbank matrix.
fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let fmax = sample_rate as f32 / 2.0;
    let fmin = 0.0f32;

    // Mel scale conversion
    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Mel points evenly spaced
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((n_fft + 1) as f32 * hz / sample_rate as f32).floor() as usize)
        .collect();

    // Create filterbank
    let n_bins = n_fft / 2 + 1;
    let mut filterbank = vec![vec![0.0f32; n_bins]; n_mels];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in left..center {
            if center > left {
                filterbank[m][k] = (k - left) as f32 / (center - left) as f32;
            }
        }
        for k in center..right {
            if right > center {
                filterbank[m][k] = (right - k) as f32 / (right - center) as f32;
            }
        }
    }

    filterbank
}

/// Normalize audio loudness (simplified version).
pub fn normalize_loudness(samples: &mut [f32], target_db: f32) {
    // Simple RMS-based normalization
    let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    if rms > 1e-10 {
        let current_db = 20.0 * rms.log10();
        let gain = 10.0f32.powf((target_db - current_db) / 20.0);
        let gain = gain.min(10.0); // Limit gain to avoid clipping
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let resampled = resample(&samples, 16000, 16000).unwrap();
        assert_eq!(samples.len(), resampled.len());
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = create_mel_filterbank(22050, 1024, 80);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), 513);
    }
}
