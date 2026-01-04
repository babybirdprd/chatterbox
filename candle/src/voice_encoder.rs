use candle_core::{IndexOp, Result, Tensor};
use candle_nn::rnn::LSTMState;
use candle_nn::{LSTMConfig, Module, VarBuilder, LSTM, RNN};

pub struct VoiceEncoderConfig {
    pub ve_hidden_size: usize,
    pub speaker_embed_size: usize,
    pub num_mels: usize,
    pub num_layers: usize,
    pub ve_final_relu: bool,
    pub ve_partial_frames: usize, // Number of frames per partial utterance
}

impl Default for VoiceEncoderConfig {
    fn default() -> Self {
        // FIXED: num_mels must be 40 to match ve.safetensors
        Self {
            ve_hidden_size: 256,
            speaker_embed_size: 256,
            num_mels: 40,
            num_layers: 3,
            ve_final_relu: true,
            ve_partial_frames: 160, // Python default
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

    /// Forward pass for a single partial utterance batch
    /// mels: (B, T, 40) where T should be ve_partial_frames
    pub fn forward(&self, mels: &Tensor) -> Result<Tensor> {
        let (b, t, _m) = mels.dims3()?;
        let mut hidden_states = mels.clone();

        for (_layer_idx, layer) in self.lstm.iter().enumerate() {
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

    /// Inference with partial utterance averaging (matches Python implementation)
    /// mels: (B, T, M) unscaled mels - full utterance
    /// overlap: fraction of overlap between partials (default 0.5)
    /// min_coverage: minimum coverage for the last partial (default 0.8)
    pub fn inference(&self, mels: &Tensor, overlap: f32, min_coverage: f32) -> Result<Tensor> {
        let (b, t, m) = mels.dims3()?;
        let partial_frames = self.config.ve_partial_frames;

        // Calculate frame step (how many frames between partial starts)
        let frame_step = ((partial_frames as f32) * (1.0 - overlap)).round() as usize;
        let frame_step = frame_step.max(1).min(partial_frames);

        // Calculate number of partials needed
        let (n_partials, target_len) =
            get_num_partials(t, frame_step, min_coverage, partial_frames);

        eprintln!(
            "[VoiceEncoder::inference] t={}, partial_frames={}, frame_step={}, n_partials={}, target_len={}",
            t, partial_frames, frame_step, n_partials, target_len
        );

        // Pad mels if needed to reach target length
        let mels = if target_len > t {
            let pad_len = target_len - t;
            let pad = Tensor::zeros((b, pad_len, m), mels.dtype(), mels.device())?;
            Tensor::cat(&[mels, &pad], 1)?
        } else {
            mels.narrow(1, 0, target_len)?
        };

        // Extract partials and forward each
        let mut all_embeds = Vec::new();
        for batch_idx in 0..b {
            let batch_mels = mels.i(batch_idx)?; // (T, M)
            let mut partial_embeds = Vec::new();

            for i in 0..n_partials {
                let start = i * frame_step;
                let partial = batch_mels.narrow(0, start, partial_frames)?; // (P, M)
                let partial = partial.unsqueeze(0)?; // (1, P, M)

                let embed = self.forward(&partial)?; // (1, E)
                partial_embeds.push(embed.squeeze(0)?); // (E,)
            }

            // Average the partial embeddings
            let stacked = Tensor::stack(&partial_embeds, 0)?; // (N, E)
            let mean_embed = stacked.mean(0)?; // (E,)

            all_embeds.push(mean_embed);
        }

        // Stack batch embeddings
        let raw_embeds = Tensor::stack(&all_embeds, 0)?; // (B, E)

        // L2 normalize the averaged embeddings
        let norm = raw_embeds.sqr()?.sum_keepdim(1)?.sqrt()?;
        raw_embeds.broadcast_div(&norm)
    }
}

/// Calculate number of partials needed to cover the mel spectrogram
/// Returns (n_partials, target_length)
fn get_num_partials(
    n_frames: usize,
    step: usize,
    min_coverage: f32,
    win_size: usize,
) -> (usize, usize) {
    if n_frames == 0 {
        return (1, win_size);
    }

    let n_wins_base = if n_frames >= win_size {
        (n_frames - win_size + step) / step
    } else {
        0
    };
    let remainder = if n_frames >= win_size {
        (n_frames - win_size + step) % step
    } else {
        n_frames
    };

    // Check if we need an extra window for coverage
    let coverage = if n_wins_base == 0 {
        1.0 // Will need at least 1 window
    } else {
        (remainder + (win_size - step)) as f32 / win_size as f32
    };

    let n_wins = if n_wins_base == 0 || coverage >= min_coverage {
        n_wins_base + 1
    } else {
        n_wins_base
    };

    let target_len = win_size + step * (n_wins - 1);
    (n_wins, target_len)
}
