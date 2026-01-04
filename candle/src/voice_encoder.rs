use candle_core::{IndexOp, Result, Tensor};
use candle_nn::rnn::LSTMState;
use candle_nn::{LSTMConfig, Module, VarBuilder, LSTM, RNN};

pub struct VoiceEncoderConfig {
    pub ve_hidden_size: usize,
    pub speaker_embed_size: usize,
    pub num_mels: usize,
    pub num_layers: usize,
    pub ve_final_relu: bool,
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

    pub fn forward(&self, mels: &Tensor) -> Result<Tensor> {
        // mels: (B, T, 40)
        let (b, t, _m) = mels.dims3()?;
        eprintln!("[VoiceEncoder] input mels: b={}, t={}, m={}", b, t, _m);
        let mut hidden_states = mels.clone();

        for (layer_idx, layer) in self.lstm.iter().enumerate() {
            eprintln!(
                "[VoiceEncoder] LSTM layer {}: creating zeros (b={}, h={})",
                layer_idx, b, self.config.ve_hidden_size
            );
            let h = Tensor::zeros((b, self.config.ve_hidden_size), mels.dtype(), mels.device())?;
            let c = Tensor::zeros((b, self.config.ve_hidden_size), mels.dtype(), mels.device())?;
            let mut state = LSTMState { h, c };

            let mut outputs = Vec::new();
            for i in 0..t {
                let input_step = hidden_states.i((.., i, ..))?.contiguous()?;
                state = layer.step(&input_step, &state)?;
                outputs.push(state.h.clone());
            }
            eprintln!(
                "[VoiceEncoder] LSTM layer {}: stacking {} outputs",
                layer_idx,
                outputs.len()
            );
            hidden_states = Tensor::stack(&outputs, 1)?;
            eprintln!(
                "[VoiceEncoder] LSTM layer {}: hidden_states: {:?}",
                layer_idx,
                hidden_states.dims()
            );
        }

        let last_hidden = hidden_states.i((.., t - 1, ..))?;
        let mut raw_embeds = self.proj.forward(&last_hidden)?;

        if self.config.ve_final_relu {
            raw_embeds = raw_embeds.relu()?;
        }

        let norm = raw_embeds.sqr()?.sum_keepdim(1)?.sqrt()?;
        raw_embeds.broadcast_div(&norm)
    }
}
