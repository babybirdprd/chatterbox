# Chatterbox TTS ONNX Export Guide

## Introduction

This document outlines the efforts undertaken to export components of the Chatterbox TTS model to the ONNX format, with the goal of enabling inference using `onnxruntime`. It details the intended ONNX pipeline, the Python scripts developed to facilitate these exports, the current status of the project, known limitations, and guidance for users wishing to continue this work.

## Intended ONNX Pipeline Overview

The envisioned end-to-end TTS pipeline using ONNX models would consist of several stages:

1.  **Speaker Encoding:** A speaker encoder model (e.g., `CAMPPlus`) would process a reference audio to produce a speaker embedding.
    *   **ONNX Model:** `campplus_speaker_encoder.onnx`
2.  **Text-to-SpeechTokens (T3 Stage):** The T3 model would take input text and the speaker embedding to generate a sequence of S3 speech tokens.
    *   **ONNX Model:** `t3_backbone.onnx` (for the core Llama-based transformer).
    *   **Python Handler:** A Python-based `ONNXAlignmentStreamAnalyzer` would be needed to manage the generation process dynamically, interacting with attention maps from the T3 ONNX model to guide token sampling (e.g., handling false starts, repetitions, and forcing EOS).
3.  **SpeechTokens-to-Waveform (S3Gen Stage):**
    *   **S3 Token Encoder (Missing Component):** S3 speech tokens from T3 need to be converted into conditioning inputs (`mu`, `cond`) suitable for the CFM estimator. This component is currently not explicitly designed or exported.
    *   **CFM Estimator (Mel Spectrogram Generation):** A Conditional Flow Matching (CFM) estimator (`ConditionalDecoder`) would generate a mel spectrogram from the S3 token-derived conditioning, speaker embedding, and initial noise.
        *   **ONNX Model:** `cfm_estimator.onnx`
        *   **Python Handler:** The CFM ODE solver loop (e.g., `solve_euler_onnx`) would be managed in Python, iteratively calling the ONNX CFM estimator.
    *   **HiFiGAN Vocoder (Waveform Synthesis):** A HiFiGAN vocoder (`HiFTGenerator`) would convert the mel spectrogram into the final audio waveform.
        *   **ONNX Model:** `hifigan_vocoder.onnx`

## Export Scripts Provided

The following Python scripts were developed to prepare individual model components for ONNX export. These scripts set up the model, generate dummy inputs, and define export parameters. The actual `torch.onnx.export(...)` calls within these scripts are **commented out** and need to be uncommented by the user to perform the export in a suitable environment.

*   **T3 Backbone Export Logic:**
    *   The logic for exporting the T3 model's Llama backbone was developed and executed within `run_in_bash_session` calls during previous tasks. This involved a `T3ExportWrapper` to manage inputs and trace the `prepare_input_embeds` and Llama transformer forward pass. This logic would need to be encapsulated in a dedicated script (e.g., `export_t3_backbone.py`) by the user, based on the successful attempts.
*   **`export_campplus.py`:**
    *   Prepares the `CAMPPlus` speaker encoder model for ONNX export.
    *   Targets the `CAMPPlus.inference()` method via a wrapper for easier input handling.
*   **`export_cfm_estimator.py`:**
    *   Prepares the `ConditionalDecoder` (the CFM estimator) for ONNX export.
    *   Targets the `ConditionalDecoder.forward()` method directly.
*   **`export_hifigan.py`:**
    *   Prepares the `HiFTGenerator` vocoder for ONNX export.
    *   Targets the `HiFTGenerator.inference()` method.

## Integration Script

*   **`onnx_tts_pipeline.py`:**
    *   This script provides a high-level, conceptual outline of how the different ONNX models and Python-managed dynamic components (like `ONNXAlignmentStreamAnalyzer` and `solve_euler_onnx`) would integrate to form an end-to-end TTS pipeline.
    *   It uses placeholder classes and functions to simulate the behavior of ONNX sessions and Python handlers, as actual ONNX files were not consistently generated.

## Current Status and Limitations

*   **Environment Instability:** The primary challenge during this work was persistent environment instability. Most attempts to run `torch.onnx.export` (even for smaller models or with `batch_size=1`) resulted in "Internal error occurred when running command" or "Killed" messages, suspected to be due to Out-of-Memory (OOM) conditions or other resource limitations in the provided sandbox.
*   **Generated ONNX Files:**
    *   The only ONNX model successfully generated during the development process is **`t3_backbone.onnx`**.
    *   This model was exported with `batch_size=1`.
    *   **Known Limitation for `t3_backbone.onnx`:** During its export, a "Tensor Iteration Warning" was observed. This indicates that the embedding concatenation step (`torch.stack` with `zip` in `T3.prepare_input_embeds`) was likely unrolled for `batch_size=1`, meaning the ONNX graph for this specific operation is not dynamic and may fail or produce incorrect results for batch sizes greater than 1. An attempt to refactor this to `torch.cat` also resulted in "Internal errors" during export.
    *   No other ONNX files for `CAMPPlus`, `CFMEstimator`, or `HiFiGAN` were reliably generated due to the aforementioned environment errors.
*   **Missing "S3 Token Encoder":**
    *   The `onnx_tts_pipeline.py` script highlighted a critical missing component: an "S3 Token Encoder." This module is needed to convert the S3 speech tokens (output by the T3 stage) into the appropriate conditioning tensors (`mu` and `cond`) required by the CFM estimator stage. This component would need to be designed, implemented, and exported to ONNX for the full pipeline to be functional.

## Guidance for Users

1.  **Stable Environment:** To successfully generate the ONNX models, users must run the provided export scripts (`export_*.py` files, and a user-created script for T3 based on provided logic) in a stable Python environment with sufficient memory and all dependencies correctly installed. Refer to `pyproject.toml` and the dependencies installed during the development process (e.g., `torch==2.6.0`, `transformers==4.46.3`, `onnx==1.18.0`, `librosa`, `diffusers==0.29.0`, `conformer`, `omegaconf`). Ensure `PYTHONPATH` is correctly set to include the `src` directory of the project.
2.  **Running Export Scripts:** Uncomment the `torch.onnx.export(...)` call within each script and execute it.
3.  **`t3_backbone.onnx` Batch Dynamicity:** For robust batch size support in the T3 model, the `T3.prepare_input_embeds` method's embedding concatenation should be refactored to use batch-wise operations (e.g., `torch.cat` as attempted) if the environment allows for successful export after such changes. The current `t3_backbone.onnx` is likely limited to `batch_size=1`.
4.  **Develop "S3 Token Encoder":** A crucial step for the S3Gen stage is to design, implement, and export an ONNX model for the "S3 Token Encoder" component that processes S3 speech tokens into inputs for the CFM estimator.
5.  **Implement Python Handlers:** The Python logic for `ONNXAlignmentStreamAnalyzer` and `solve_euler_onnx` sketched in `onnx_tts_pipeline.py` needs to be fully implemented and tested with the actual ONNX models.

## Disclaimer

The ONNX export scripts and pipeline outline provided are for development and experimental purposes. Significant further work, including ensuring a stable export environment, resolving ONNX graph warnings (like the tensor iteration), implementing the missing "S3 Token Encoder", and thoroughly testing each component and the full pipeline, is required to achieve a fully functional and robust ONNX-based Chatterbox TTS system.The `README_ONNX.md` file has been successfully created with the detailed summary of the ONNX export efforts, including the intended pipeline, scripts developed, current status, limitations (especially environment instability and the single generated `t3_backbone.onnx` with its batch size caveat), the identified missing "S3 Token Encoder" component, and guidance for users.

This documentation accurately reflects the outcomes of the preceding tasks and provides a clear path forward for anyone continuing this work.
