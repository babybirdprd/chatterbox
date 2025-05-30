from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from .s3_local_utils import padding # Use local padding
from .s3_model_v2_local import S3TokenizerV2Base as S3TokenizerV2, ModelConfig # Use local S3TokenizerV2Base and ModelConfig


# def local_padding(list_of_tensors: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]: # Replaced by s3_local_utils.padding
# """
# Pads a list of 2D tensors (n_mels, n_frames) along the n_frames dimension.
# Returns the padded batch tensor and a tensor of original lengths (n_frames).
# """
# # Assuming each tensor in list_of_tensors is of shape (n_mels, n_frames)
# # We want to pad along the n_frames dimension (dim=1)
# if not list_of_tensors:
# return torch.empty(0), torch.empty(0)
#
# n_mels = list_of_tensors[0].size(0)
# original_lengths = [t.size(1) for t in list_of_tensors]
# max_len = max(original_lengths)
#    
# n_batch = len(list_of_tensors)
# # Create a padded tensor: (batch_size, n_mels, max_len_frames)
# padded_tensor = list_of_tensors[0].new_full((n_batch, n_mels, max_len), pad_value)
#    
# for i, tensor in enumerate(list_of_tensors):
# current_len = tensor.size(1)
# padded_tensor[i, :, :current_len] = tensor
#        
# return padded_tensor, torch.tensor(original_lengths, dtype=torch.long)


# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str="speech_tokenizer_v2_25hz", # This name is used by S3TokenizerV2Base
        config: ModelConfig = ModelConfig()
    ):
        super().__init__(name=name, config=config) # Pass name and config to base

        self.n_fft = 400
        # Ensure _mel_filters is compatible with config.n_mels from ModelConfig
        _mel_filters_data = librosa.filters.mel( 
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters_data),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        """
        Given a list of wavs with the same `sample_rate`, pad them so that the length is multiple of 40ms (S3 runs at 25 token/sec).
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav,
                (0, intended_wav_len - wav.shape[-1]),
                mode="constant",
                value=0
            )
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor, # Should be List[torch.Tensor] or similar based on usage
        accelerator=None, # Removed type hint 'Accelerator' as it's not defined
        max_len: int=None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).
        FIXME: this class inherits `nn.Module` but doesn't accept `torch.Tensor` and handles a list of wavs one by one, which is unexpected.
               The base class S3TokenizerV2Base expects batched mels.

        Args
        ----
        - `wavs`: List of 16 kHz speech audio tensors, or a batched tensor.
                  The loop below suggests it processes a list of wavs.
        - `max_len` max length to truncate the output sequence to (25 token/sec).
        NOTE: please pad the waveform if longer sequence is needed.
        """
        # If wavs is a single tensor, make it a list for the loop
        if isinstance(wavs, torch.Tensor) and wavs.dim() == 2: # Simple case: (B, L)
             wavs_list = [wav.squeeze(0) for wav in torch.split(wavs, 1, dim=0)]
        elif isinstance(wavs, torch.Tensor) and wavs.dim() == 1: # Single unbatched wav
             wavs_list = [wavs]
        elif isinstance(wavs, list):
             wavs_list = wavs
        else:
            # This path needs to be clarified based on expected input.
            # For now, assume wavs_list can be derived or is directly provided.
            # This might be an issue if T3 passes a batched tensor directly.
            # The original comment "handles a list of wavs one by one" is key.
            raise ValueError("Input 'wavs' format not handled by this S3Tokenizer wrapper.")

        processed_wavs = self._prepare_audio(wavs_list)
        mels_list = [] # Renamed mels to mels_list to avoid confusion before padding
        for wav_item in processed_wavs: # wav_item is (1, L) or (L)
            wav_item = wav_item.to(self.device if hasattr(self, 'device') else next(self.parameters()).device)
            # log_mel_spectrogram expects (L) or (B,L) where B=1
            mel = self.log_mel_spectrogram(wav_item)  # [1, F, T_mel] or [F, T_mel]
            if mel.dim() == 2: # if (F, T_mel)
                mel = mel.unsqueeze(0) # Make it (1, F, T_mel)
            
            if max_len is not None:
                # num_mel_frames = 4 * num_tokens (S3_TOKEN_HOP / S3_HOP = 640 / 160 = 4)
                mel = mel[..., :max_len * 4] 
            mels_list.append(mel.squeeze(0)) # list of (F, T_mel)

        # `padding` from s3_local_utils expects a list of (F, T_mel) tensors
        # and returns (B, F, T_max_mel), (B,)
        batched_mels, mel_lens = padding(mels_list) 
        
        # S3TokenizerV2Base.quantize expects mel: (B, F, T_mel), mel_len: (B,)
        # Ensure device consistency
        target_device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        batched_mels = batched_mels.to(target_device)
        mel_lens = mel_lens.to(target_device)

        # The accelerator logic might need to be re-evaluated if it's used.
        # For now, assume direct call to self.quantize (which is from S3TokenizerV2Base)
        # speech_tokens, speech_token_lens = self.quantize(batched_mels, mel_lens)
        
        # Call the base class quantize method directly
        # The S3TokenizerV2Base (this class's parent) has the quantize method
        speech_tokens, speech_token_lens = super().quantize(batched_mels, mel_lens)

        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor, # Expects (L) or (1,L)
        padding_val: int = 0, # Renamed from padding to avoid conflict with imported padding function
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding_val: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (n_mels, n_frames) or (1, n_mels, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if not torch.is_tensor(audio): # Should already be tensor from _prepare_audio
            audio = torch.from_numpy(audio)

        current_device = self.device if hasattr(self, 'device') else next(self.parameters()).device
        audio = audio.to(current_device)

        if audio.dim() == 1: # Ensure it's (1,L) for stft
            audio = audio.unsqueeze(0)

        if padding_val > 0:
            audio = F.pad(audio, (0, padding_val))
        
        window_on_device = self.window.to(current_device)
        
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=window_on_device,
            return_complex=True
        ) # stft is (B, Freq, Time, 2) for real or (B, Freq, Time) for complex
        
        # For complex stft output from torch >= 1.7.0
        # magnitudes will be (B, Freq, Time)
        magnitudes = stft[..., :-1].abs()**2 # Exclude last freq bin for Nyquist if n_fft is even (400)

        mel_filters_on_device = self._mel_filters.to(current_device)
        # mel_filters (n_mels, n_fft/2 + 1) e.g. (128, 201)
        # magnitudes (B, Freq=n_fft/2, Time) e.g. (B, 200, Time)
        # So matmul should be mel_filters @ magnitudes
        mel_spec = mel_filters_on_device @ magnitudes # (B, n_mels, Time)

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0) # Ensure this max is over correct dims
        log_spec = (log_spec + 4.0) / 4.0
        
        # Return (n_mels, n_frames) if input was effectively unbatched after _prepare_audio
        # Squeeze if B=1, consistent with how mels_list.append(mel.squeeze(0)) was used
        return log_spec.squeeze(0) if audio.shape[0] == 1 else log_spec


# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    """
    s3tokenizer.S3TokenizerV2 with the following changes:
    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str="speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig()
    ):
        super().__init__(name)

        self.n_fft = 400
        _mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels
        )
        self.register_buffer(
            "_mel_filters",
            torch.FloatTensor(_mel_filters),
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
        )

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        """
        Given a list of wavs with the same `sample_rate`, pad them so that the length is multiple of 40ms (S3 runs at 25 token/sec).
        """
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav,
                (0, intended_wav_len - wav.shape[-1]),
                mode="constant",
                value=0
            )
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor,
        accelerator: 'Accelerator'=None,
        max_len: int=None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).
        FIXME: this class inherits `nn.Module` but doesn't accept `torch.Tensor` and handles a list of wavs one by one, which is unexpected.

        Args
        ----
        - `wavs`: 16 kHz speech audio
        - `max_len` max length to truncate the output sequence to (25 token/sec).
        NOTE: please pad the waveform if longer sequence is needed.
        """
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = local_padding(mels) # Use the local_padding function
        if accelerator is None:
            tokenizer = self
        else:
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(
            audio, self.n_fft, S3_HOP,
            window=self.window.to(self.device),
            return_complex=True
        )
        magnitudes = stft[..., :-1].abs()**2

        mel_spec = self._mel_filters.to(self.device) @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec
