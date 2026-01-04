import torch
import numpy as np
import librosa
from safetensors.torch import save_file

def mel_spectrogram(y, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480, win_size=1920,
                    fmin=0, fmax=8000, center=False):
    if isinstance(y, np.ndarray):
        y = torch.tensor(y).float()
    if len(y.shape) == 1:
        y = y[None, ]

    # Reflect padding as per Python: (n_fft - hop_size) / 2 = 720
    pad = (n_fft - hop_size) // 2
    y_padded = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)

    spec = torch.stft(
        y_padded,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=torch.hann_window(win_size).to(y.device),
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + (1e-9))

    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel_basis).float().to(y.device)
    
    spec = torch.matmul(mel_basis, spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))

    return spec

def main():
    # Load reference.wav
    audio_path = 'reference.wav'
    wav, sr = librosa.load(audio_path, sr=24000)
    
    # Extract mel
    mel = mel_spectrogram(wav)
    
    # Save to safetensors
    tensors = {
        "ref_audio": torch.from_numpy(wav),
        "ref_mel": mel.squeeze(0), # [80, T]
    }
    save_file(tensors, "debug_tensors.safetensors")
    print(f"Saved debug_tensors.safetensors: audio={wav.shape}, mel={mel.shape}")

if __name__ == "__main__":
    main()
