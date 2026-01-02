# Chatterbox Rust Port - Full Report

## Current Status

The port currently implements the core **Acoustic Models** and **Helper Models** for Chatterbox Turbo and Standard:

1.  **T3 (Text-to-Token):** GPT-2 based model for predicting speech tokens from text.
    *   Conditioning logic for Speaker Embeddings, Emotion, and Prompts is implemented.
    *   Inference loop (`generate`) is implemented.

2.  **S3Gen (Token-to-Mel):** Conditional Flow Matching (CFM) decoder using a U-Net architecture.
    *   Full U-Net structure (Down/Mid/Up blocks with ResNets and Transformers) is implemented in `s3gen.rs`.
    *   Upsample Conformer Encoder is implemented.
    *   Flow Matching inference with Euler ODE solver is implemented.

3.  **Speaker Encoder (CAMPPlus):**
    *   **Status:** Implemented in `src/campplus.rs`.
    *   Extracts speaker embeddings from reference audio.
    *   Integrated into `S3Gen` and `ChatterboxTTS`.

4.  **S3Tokenizer (Speech Tokenizer):**
    *   **Status:** Implemented in `src/s3tokenizer.rs`.
    *   Ported from SenseVoice-Large/S3Tokenizer logic.
    *   Uses Finite Scalar Quantization (FSQ) codebook.
    *   Integrated for prompt token generation.

## Missing Components for Full Audio Generation

To generate listenable `.wav` files ("End-to-End Inference"), the following critical component is missing:

### 1. Vocoder (HiFi-GAN / HiFTGenerator)
*   **Role:** Converts the Mel Spectrograms (output of S3Gen) into time-domain audio waveforms.
*   **Status:** **Not implemented.**
    *   `src/hifigan.rs` contains the model structure definition (`HiFTGenerator`, `ResBlock`, etc.).
    *   **Critical Gap:** The STFT and iSTFT functions (`simple_stft`, `simple_istft`) are placeholders that return zeros.
    *   The `ChatterboxTurboTTS` pipeline currently skips the vocoder entirely and attempts to save flattened Mel Spectrogram values as audio samples.
*   **Requirement:** Implement proper Short-Time Fourier Transform (STFT) and Inverse STFT (iSTFT) using `realfft` or similar, and wire up the `HiFTGenerator` in the inference pipeline.

## Summary

You **cannot** generate usable audio files yet.
*   The system currently produces a WAV file containing raw Mel Spectrogram data (float values), which sounds like static/garbage noise.
*   The acoustic models (Text -> Tokens -> Mel) are working and producing tensors.
*   The final conversion step (Mel -> Audio) is missing.
