
## 5. Orchestration (TTS Pipeline)

| Feature | Python Implementation (`tts_turbo.py`) | Rust Implementation (`chatterbox.rs`) | Status |
| :--- | :--- | :--- | :--- |
| **Pipeline Structure** | VoiceEncoder -> T3 -> S3Gen -> Watermarker | VoiceEncoder -> T3 -> S3Gen (No Watermarker) | ❌ **Discrepancy** (Missing Watermarker) |
| **Normalization (Text)** | `punc_norm` (Capitalize, Punc replace, Ensure end punc) | `normalize_text` (Capitalize, Punc replace, Ensure end punc) | ✅ Verified (Logic matches) |
| **Normalization (Audio)** | `norm_loudness` (pyloudnorm -27 LUFS) | `audio::normalize_loudness` (-27.0) | ✅ Verified |
| **Reference Processing** | Load -> Resample (16k & 24k) -> Embed | Load -> Resample (16k & 24k) -> Embed | ✅ Verified |
| **Voice Encoder Input** | 16k Wav -> 40 Mels -> Embed -> Mean | 16k Wav -> 40 Mels -> Embed (No Mean?) | ❌ **Discrepancy** (Python averages embeddings, Rust takes single forward?) |
| **S3Tokenizer Input** | 16k Wav -> 128 Mels -> Log -> Norm -> Tokenize | 16k Wav -> 128 Mels -> Log -> Norm -> Tokenize | ✅ Verified |
| **S3Tokenizer Norm** | `max - 8.0`, `(x + 4) / 4` (Implied from Rust code reading) | `max - 8.0`, `(x + 4) / 4` | ✅ Verified |
| **CAMPPlus Input** | 16k Wav -> 80 Mels -> Log -> MeanNorm -> Forward | 16k Wav -> 80 Mels -> Log -> MeanNorm -> Forward | ✅ Verified |
| **S3Gen Condition** | 24k Wav -> 80 Mels -> Log -> Pad/Crop | 24k Wav -> 80 Mels -> Log -> Pad/Crop | ✅ Verified |
| **Generation Loop** | T3 Generate -> Filter (<6561) -> Cat Silence -> S3Gen | T3 Generate -> Filter (<6561) -> **No Silence** -> S3Gen | ❌ **Discrepancy** (Missing Silence Padding) |
| **T3 Params** | `hp.llama_config_name = "GPT2_medium"` | `T3Config` (Hardcoded defaults match GPT2 Medium) | ✅ Verified |
| **Conditionals** | `Conditionals` dataclass | Passed as separate args | ✅ Acceptable |
| **S3Gen Prompt** | `ref_dict` (Mel + Tokens + Embed) | `cond` (Mel) + `spk_emb` | ✅ Verified |
| **Weights Loading** | `from_pretrained` (HF Hub) or `from_local` | `from_pretrained` (HF Hub) or `from_local` | ✅ Verified |
| **Output** | `watermarked_wav` | `samples` (Vec<f32>) | ❌ **Discrepancy** (Missing Watermark) |

## Summary of Critical Discrepancies

1.  **Model Architecture**:
    *   **Voice Encoder**: Stride mismatch in `FCM` and `BasicResBlock`. Rust uses symmetric stride (likely 2x2), Python uses `(2, 1)` (Freq only). This fundamentally changes the time dimension resolution of the embeddings.
    *   **T3**: Rust implementation lacks KV Caching, leading to O(N^2) inference complexity instead of O(N).
    *   **T3**: Position Embeddings are initialized differently (Rust reuses WPE vs Python's separate LearnedPositionEmbeddings).
    *   **HiFi-GAN**: Final output layer in Rust returns raw chunks (Real/Imag?) instead of `exp(mag)` and `sin(phase)` before iSTFT.

2.  **Missing Features**:
    *   **Watermarking**: Rust pipeline lacks the `perth` watermarking step.
    *   **Silence Padding**: Rust pipeline does not append silence tokens before S3Gen synthesis, which might cut off the end of audio.
    *   **CFG**: T3 inference in Rust does not support Classifier-Free Guidance.
    *   **Trim Fade**: S3Gen in Rust lacks the post-synthesis fade-in/out to reduce artifacts.
    *   **Voice Encoder Averaging**: Python averages speaker embeddings from the prompt, Rust appears to use the result directly (needs verification if `VoiceEncoder` handles batches/averaging internally).

3.  **Normalization**:
    *   **DenseLayer**: Rust skips `BatchNorm1d` (affine=False) in the final projection of CAMPPlus.

4.  **Vocoder**:
    *   **Output**: HiFi-GAN output processing (Mag/Phase) seems incorrect in Rust.

## Conclusion
The Rust port has significant architectural and functional discrepancies compared to the Python implementation. The most critical are the Voice Encoder stride issues (likely breaking speaker embedding consistency), the HiFi-GAN output logic (likely breaking audio synthesis), and the lack of KV caching (performance).
