# Chatterbox Rust Port - Full Report

## Current Status

The port currently implements the core **Acoustic Models** for Chatterbox Turbo and Standard:
1.  **T3 (Text-to-Token):** GPT-2 based model for predicting speech tokens from text.
    *   Conditioning logic for Speaker Embeddings, Emotion, and Prompts is implemented.
    *   Inference loop (`generate`) is implemented.
2.  **S3Gen (Token-to-Mel):** Conditional Flow Matching (CFM) decoder using a U-Net architecture.
    *   Full U-Net structure (Down/Mid/Up blocks with ResNets and Transformers) is implemented in `s3gen.rs`.
    *   Upsample Conformer Encoder is implemented.
    *   Flow Matching inference with Euler ODE solver is implemented.

## Missing Components for Full Audio Generation

To generate listenable `.wav` files ("End-to-End Inference"), the following components are still required:

### 1. Vocoder (HiFi-GAN / HiFTGenerator)
*   **Role:** Converts the Mel Spectrograms (output of S3Gen) into time-domain audio waveforms.
*   **Status:** Not ported. The current pipeline outputs a Tensor of shape `(B, 80, T)` (Mel Spectrograms), but cannot convert this to audio.
*   **Requirement:** Port `HiFTGenerator` (source-filter HiFi-GAN) from `src/chatterbox/models/s3gen/hifigan.py`.

### 2. Speaker Encoder (CAMPPlus)
*   **Role:** Extracts a fixed-size speaker embedding (e.g., 512-dim or 192-dim x-vector) from a reference audio file. This embedding is a critical conditioning input for S3Gen to clone the voice.
*   **Status:** Not ported. The current implementation accepts a placeholder/dummy tensor (zeros) for the `spks` argument in S3Gen to allow the model to run without crashing, but the output mels will lack the correct speaker timbre or be degraded.
*   **Requirement:** Port `CAMPPlus` model (or similar x-vector extractor) and its pre-processing.

### 3. Text Tokenizer Integration
*   **Role:** Converts input text string into token IDs expected by the T3 model.
*   **Status:** `main.rs` currently uses dummy/random tokens for demonstration. The project includes `tokenizers` dependency, but the specific `tokenizer.json` logic needs to be wired up in `main.rs` to process user input text.
*   **Requirement:** Load `tokenizer.json` and use it to encode CLI text argument.

### 4. S3Tokenizer (Speech Tokenizer)
*   **Role:** Converts reference audio into speech tokens. This is used for "Prompt Conditioning" in the Standard Chatterbox model (to continue speech from a prompt).
*   **Status:** Not implemented. `ChatterboxTTS` (Standard) currently skips prompt token generation.
*   **Requirement:** Port `S3Tokenizer` (likely involving a quantizer/codebook lookup) if prompt conditioning is needed.

## Summary

You **cannot** generate usable audio files yet. You can:
1.  Load the weights (if downloaded).
2.  Run the Text-to-Mel inference pipeline on CPU or GPU.
3.  Inspect the output Mel Spectrogram tensor shapes.

To enable audio generation, the priority is porting the **Vocoder**.

May need to clone this as well, if needed for reference.
https://github.com/xingchensong/S3Tokenizer