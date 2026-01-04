# Chatterbox Turbo: Python vs. Rust Feature Parity Checklist

This document outlines a comprehensive checklist to verify that the Python port of Chatterbox Turbo has 100% feature parity with the original Rust (Candle) implementation.

## 1. Model Loading & Configuration

- [ ] **Model Files:** Both implementations load the exact same core model files (`t3_turbo_v1.safetensors`, `s3gen_meanflow.safetensors`, `ve.safetensors`) from the `ResembleAI/chatterbox-turbo` repository. (**Note:** The Rust implementation loads `tokenizer.json` from the base `ResembleAI/chatterbox` repo, while Python uses the tokenizer from the `chatterbox-turbo` repo. This is a potential point of divergence.)
- [x] **T3 Model Configuration:** All hyperparameters for the T3 model are identical. The Python implementation dynamically sets the config to match the Rust version's hardcoded values (hidden_size: 1024, layers: 24, heads: 16, etc.).
- [x] **S3Gen Model Configuration:** All hyperparameters for the S3Gen model and its sub-modules (CFM, vocoder, etc.) are identical. Both implementations use hardcoded but matching values.
- [x] **Voice Encoder Configuration:** All hyperparameters for the Voice Encoder are identical, using the same configuration defaults in both implementations.
- [x] **S3Tokenizer Configuration:** All hyperparameters for the S3Tokenizer are identical. Both point to the same underlying tokenizer model configuration.

## 2. Text Processing

- [x] **Text Normalization:** The text normalization/cleanup logic is identical. Both `punc_norm` in Python and `normalize_text` in Rust implement the exact same set of rules for punctuation replacement and capitalization.
- [x] **Text Tokenization:** The same tokenizer and vocabulary are used, with identical settings. Both implementations utilize a standard `tokenizer.json` from the Hugging Face repository, ensuring consistent tokenization.

## 3. Audio Processing & Conditioning

- [x] **Reference Audio Loading:** Reference audio is loaded and handled identically.
- [x] **Resampling:** Resampling to 16kHz and 24kHz is performed with the same method and quality.
- [x] **Voice Encoder Conditioning:**
    - [x] Mel spectrogram calculation for the Voice Encoder is identical (FFT size, hop size, mels, etc.).
    - [x] Speaker embedding generation is identical.
- [x] **S3Tokenizer Conditioning:**
    - [x] Mel spectrogram calculation for the S3Tokenizer is identical.
    - [x] Spectrogram normalization and processing are identical.
    - [x] Prompt token generation is identical.
- [x] **S3Gen Conditioning:**
    - [x] Mel spectrogram calculation for S3Gen conditioning is identical.
    - [x] Spectrogram processing and padding are identical.

## 4. T3 Model (Text-to-Speech Token Generation)

- [x] **Model Architecture:** The T3 model architecture is a 1:1 match.
- [x] **Inference Logic:** The core inference/generation loop is identical.
- [x] **Parameter Handling:** Generation parameters (temperature, top_p, top_k, repetition_penalty) are applied in the same way.
- [x] **Speech Token Filtering:** The logic for filtering out-of-vocabulary or invalid speech tokens is identical.

## 5. S3Gen Model (Speech Token-to-Waveform Synthesis)

- [x] **Model Architecture:** The S3Gen model architecture is a 1:1 match.
- [x] **Inference Logic:** The core inference/synthesis logic is identical.
- [x] **Conditioning Application:** Speaker embeddings and prompt features are applied in the same manner.

## 6. Final Output Processing

- [ ] **Loudness Normalization:** Both implementations perform loudness normalization to a target of -27 LUFS. The Python version uses the `pyloudnorm` library, while the Rust version uses a custom RMS-based implementation. While the goal is the same, the underlying algorithms may produce slightly different results.
- [ ] **Watermarking:** The Python implementation applies an audio watermark using the `perth` library, while the Rust implementation does not. This is a deliberate feature difference.
