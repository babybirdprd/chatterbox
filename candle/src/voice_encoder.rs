use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{LSTMConfig, LSTM, Module, VarBuilder, RNN};
use candle_nn::rnn::LSTMState;

pub struct VoiceEncoderConfig {
    pub ve_hidden_size: usize,
    pub speaker_embed_size: usize,
    pub num_mels: usize,
    pub num_layers: usize,
    pub ve_final_relu: bool,
}

impl Default for VoiceEncoderConfig {
    fn default() -> Self {
        Self {
            ve_hidden_size: 768,
            speaker_embed_size: 256,
            num_mels: 80,
            num_layers: 3,
            ve_final_relu: true,
        }
    }
}

pub struct VoiceEncoder {
    lstm: Vec<LSTM>,
    proj: candle_nn::Linear,
    config: VoiceEncoderConfig,
    similarity_weight: Tensor,
    similarity_bias: Tensor,
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

            let input_size = if i == 0 { config.num_mels } else { config.ve_hidden_size };
            let lstm = LSTM::new(input_size, config.ve_hidden_size, lstm_config, lstm_vb.clone())?;
            lstms.push(lstm);
        }

        let proj = candle_nn::linear(config.ve_hidden_size, config.speaker_embed_size, vb.pp("proj"))?;

        let similarity_weight = vb.get(1, "similarity_weight")?;
        let similarity_bias = vb.get(1, "similarity_bias")?;

        Ok(Self {
            lstm: lstms,
            proj,
            config,
            similarity_weight,
            similarity_bias,
        })
    }

    pub fn forward(&self, mels: &Tensor) -> Result<Tensor> {
        // mels: (B, T, M)
        let (b, t, _m) = mels.dims3()?;
        let mut hidden_states = mels.clone();

        for layer in &self.lstm {
             // Initialize state as (Batch, Hidden)
             let h = Tensor::zeros((b, self.config.ve_hidden_size), mels.dtype(), mels.device())?;
             let c = Tensor::zeros((b, self.config.ve_hidden_size), mels.dtype(), mels.device())?;
             let state = LSTMState { h, c };

             let mut outputs = Vec::new();
             let mut current_state = state;

             for i in 0..t {
                 let input_step = hidden_states.i((.., i, ..))?;
                 let next_state = layer.step(&input_step, &current_state)?;
                 outputs.push(next_state.h.clone());
                 current_state = next_state;
             }

             let stacked = Tensor::stack(&outputs, 0)?; // (T, B, H)
             hidden_states = stacked.transpose(0, 1)?; // (B, T, H)
        }

        // Take the last time step hidden state
        // hidden_states: (B, T, H)
        let last_hidden = hidden_states.i((.., t-1, ..))?;

        let mut raw_embeds = self.proj.forward(&last_hidden)?;

        if self.config.ve_final_relu {
            raw_embeds = raw_embeds.relu()?;
        }

        // L2 normalize
        // raw_embeds: (B, E)
        let norm = raw_embeds.sqr()?.sum_keepdim(1)?.sqrt()?;
        let embeds = raw_embeds.broadcast_div(&norm)?;

        Ok(embeds)
    }
}
