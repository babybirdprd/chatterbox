use candle_core::{IndexOp, Result, Tensor};
use candle_nn::rnn::LSTMState;
use candle_nn::{LSTMConfig, Module, VarBuilder, LSTM, RNN};

pub struct VoiceEncoderConfig {
    pub ve_hidden_size: usize,
    pub speaker_embed_size: usize,
    pub num_mels: usize,
    pub num_layers: usize,
    pub ve_final_relu: bool,
    pub ve_partial_frames: usize,
}

impl Default for VoiceEncoderConfig {
    fn default() -> Self {
        Self {
            ve_hidden_size: 256,
            speaker_embed_size: 256,
            num_mels: 40,
            num_layers: 3,
            ve_final_relu: true,
            ve_partial_frames: 160,
        }
    }
}

pub struct VoiceEncoder {
    lstm: Vec<LSTM>,
    proj: candle_nn::Linear,
    config: VoiceEncoderConfig,
}

impl VoiceEncoder {
    pub fn new(config: VoiceEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut lstms = Vec::new();
        let lstm_vb = vb.pp("lstm");

        for i in 0..config.num_layers {
            let lstm_config = LSTMConfig {
                layer_idx: i,
                direction: candle_nn::rnn::Direction::Forward,
                ..Default::default()
            };

            let input_size = if i == 0 {
                config.num_mels
            } else {
                config.ve_hidden_size
            };
            let lstm = LSTM::new(
                input_size,
                config.ve_hidden_size,
                lstm_config,
                lstm_vb.clone(),
            )?;
            lstms.push(lstm);
        }

        let proj = candle_nn::linear(
            config.ve_hidden_size,
            config.speaker_embed_size,
            vb.pp("proj"),
        )?;

        Ok(Self {
            lstm: lstms,
            proj,
            config,
        })
    }

    /// Forward pass for a batch of partial utterances.
    /// mels: (B, T, 40) where T is ideally ve_partial_frames
    pub fn forward(&self, mels: &Tensor) -> Result<Tensor> {
        let (b, t, _m) = mels.dims3()?;
        let mut hidden_states = mels.clone();

        for (layer_idx, layer) in self.lstm.iter().enumerate() {
            let h = Tensor::zeros((b, self.config.ve_hidden_size), mels.dtype(), mels.device())?;
            let c = Tensor::zeros((b, self.config.ve_hidden_size), mels.dtype(), mels.device())?;
            let mut state = LSTMState { h, c };

            let mut outputs = Vec::new();
            for i in 0..t {
                let input_step = hidden_states.i((.., i, ..))?.contiguous()?;
                state = layer.step(&input_step, &state)?;
                outputs.push(state.h.clone());
            }
            hidden_states = Tensor::stack(&outputs, 1)?;
        }

        let last_hidden = hidden_states.i((.., t - 1, ..))?;
        let mut raw_embeds = self.proj.forward(&last_hidden)?;

        if self.config.ve_final_relu {
            raw_embeds = raw_embeds.relu()?;
        }

        let norm = raw_embeds.sqr()?.sum_keepdim(1)?.sqrt()?;
        raw_embeds.broadcast_div(&norm)
    }

    /// Computes the embedding of a batch of full utterances by splitting them into partials,
    /// running forward on partials, and averaging the results.
    /// mels: (B, T, 40) - full utterances.
    pub fn inference(&self, mels: &Tensor, overlap: f64) -> Result<Tensor> {
        let (b, t, _m) = mels.dims3()?;
        let partial_frames = self.config.ve_partial_frames;

        // Calculate frame step
        let frame_step = (partial_frames as f64 * (1.0 - overlap)).round() as usize;
        let frame_step = frame_step.max(1).min(partial_frames);

        let mut embeddings = Vec::new();

        // Process each utterance in the batch
        for i in 0..b {
            let mel = mels.i(i)?; // (T, 40)

            // Split into partials
            let mut partials = Vec::new();

            if t <= partial_frames {
                // If shorter than partial window, pad?
                // Python: if target_len > len(mel) -> pad.
                // We just pad if needed.
                if t < partial_frames {
                     let pad_len = partial_frames - t;
                     let pad = Tensor::zeros((pad_len, 40), mel.dtype(), mel.device())?;
                     let padded = Tensor::cat(&[&mel, &pad], 0)?;
                     partials.push(padded);
                } else {
                     partials.push(mel.clone());
                }
            } else {
                // Sliding window
                let mut start = 0;
                while start + partial_frames <= t {
                    let end = start + partial_frames;
                    partials.push(mel.i(start..end)?);
                    start += frame_step;
                }
                // Check if we need one more partial for coverage?
                // Python `get_num_wins` logic is a bit complex.
                // We'll stick to simple sliding window for now, or ensure we cover the end.
                // If the last window didn't reach the end, we might add one aligned to the end?
                // Python `stride_as_partials` does padding if needed.
                // Let's keep it simple: if last partial doesn't cover end, add one that ends at T?
                // Or just proceed.
                if start < t {
                     // Add a partial that ends at t (if possible)
                     if t >= partial_frames {
                          partials.push(mel.i((t - partial_frames)..t)?);
                     }
                }
            }

            if partials.is_empty() {
                 // Should not happen if t > 0
                 return Err(candle_core::Error::Msg("No partials generated".into()));
            }

            let partials_tensor = Tensor::stack(&partials, 0)?; // (P, partial_frames, 40)

            // Forward pass on partials
            // We can batch this if P is large.
            // For now, just pass all partials as a batch.
            let partial_embeds = self.forward(&partials_tensor)?; // (P, embed_size)

            // Average
            let avg_embed = partial_embeds.mean(0)?; // (embed_size)

            // Normalize
            let norm = avg_embed.sqr()?.sum_all()?.sqrt()?;
            let embed = (avg_embed / norm)?;

            embeddings.push(embed);
        }

        Tensor::stack(&embeddings, 0)
    }
}
