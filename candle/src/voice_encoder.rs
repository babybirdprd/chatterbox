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
    /// Input: (B, T, M) where T is typically ve_partial_frames (160).
    /// Output: (B, E) L2-normalized embeddings.
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

        // Take last hidden state
        let last_hidden = hidden_states.i((.., t - 1, ..))?;
        let mut raw_embeds = self.proj.forward(&last_hidden)?;

        if self.config.ve_final_relu {
            raw_embeds = raw_embeds.relu()?;
        }

        // L2 Normalize
        let norm = raw_embeds.sqr()?.sum_keepdim(1)?.sqrt()?;
        raw_embeds.broadcast_div(&norm)
    }

    /// Inference for a full utterance (or batch of utterances).
    /// Splits input into partials, computes embeddings, and averages them.
    /// Logic mirrors Python `VoiceEncoder.inference` / `embeds_from_wavs`.
    pub fn inference(&self, mels: &Tensor) -> Result<Tensor> {
        // mels: (1, T, 40) - Supporting batch size 1 for now for simplicity in slicing
        let (b, t, _m) = mels.dims3()?;
        if b != 1 {
             candle_core::bail!("VoiceEncoder inference currently supports batch size 1 only");
        }

        let partial_frames = self.config.ve_partial_frames; // 160
        let rate = 1.3;
        let sample_rate = 16000.0;
        let min_coverage = 0.8;

        // frame_step calculation matching Python
        let frame_step = ((sample_rate / rate) / partial_frames as f64).round() as usize;

        // Calculate number of windows
        let val = if t >= partial_frames {
            t - partial_frames + frame_step
        } else {
            frame_step // Ensure we handle short clips
        };
        let mut n_wins = val / frame_step;
        let remainder = val % frame_step;

        // Check coverage logic from Python
        let coverage = (remainder as f32 + (partial_frames - frame_step) as f32) / partial_frames as f32;
        if n_wins == 0 || coverage >= min_coverage {
            n_wins += 1;
        }
        if n_wins == 0 {
            n_wins = 1;
        }

        let target_len = partial_frames + frame_step * (n_wins - 1);

        // Pad input if necessary
        let mels_padded = if t < target_len {
            let pad_len = target_len - t;
            let pad = Tensor::zeros((b, pad_len, _m), mels.dtype(), mels.device())?;
            Tensor::cat(&[mels, &pad], 1)?
        } else {
            mels.narrow(1, 0, target_len)?
        };

        // Extract partials
        let mut partials = Vec::with_capacity(n_wins);
        for i in 0..n_wins {
            let start = i * frame_step;
            // narrow: dim, start, len
            let part = mels_padded.narrow(1, start, partial_frames)?; // (1, 160, 40)
            partials.push(part);
        }
        // Concat along batch dimension -> (N, 160, 40)
        let partials_tensor = Tensor::cat(&partials, 0)?;

        // Compute embeddings for each partial -> (N, 256)
        let partial_embeds = self.forward(&partials_tensor)?;

        // Average embeddings -> (1, 256)
        let mean_embed = partial_embeds.mean(0)?.unsqueeze(0)?;

        // L2 Normalize final embedding
        let norm = mean_embed.sqr()?.sum_keepdim(1)?.sqrt()?;
        mean_embed.broadcast_div(&norm)
    }
}
