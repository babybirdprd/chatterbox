use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use crate::gpt2::{GPT2Model, Config as GPT2Config};

pub struct T3Config {
    pub text_tokens_dict_size: usize,
    pub speech_tokens_dict_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub speaker_embed_size: usize,
    pub start_speech_token: u32,
    pub stop_speech_token: u32,
}

impl Default for T3Config {
    fn default() -> Self {
        Self {
            text_tokens_dict_size: 50276,
            speech_tokens_dict_size: 6563,
            hidden_size: 1024, // GPT2 Medium
            num_layers: 24,
            num_heads: 16,
            vocab_size: 50276, // Placeholder
            speaker_embed_size: 256,
            start_speech_token: 6561,
            stop_speech_token: 6562,
        }
    }
}

pub struct T3CondEnc {
    spkr_enc: Linear,
}

impl T3CondEnc {
    pub fn new(config: &T3Config, vb: VarBuilder) -> Result<Self> {
        let spkr_enc = candle_nn::linear(config.speaker_embed_size, config.hidden_size, vb.pp("spkr_enc"))?;
        Ok(Self { spkr_enc })
    }

    pub fn forward(&self, spk_emb: &Tensor) -> Result<Tensor> {
        // spk_emb: (B, E)
        let cond_spkr = self.spkr_enc.forward(spk_emb)?;
        // (B, 1, H)
        cond_spkr.unsqueeze(1)
    }
}

pub struct T3 {
    gpt2: GPT2Model,
    text_emb: Embedding,
    speech_emb: Embedding,
    _text_head: Linear,
    speech_head: Linear,
    cond_enc: T3CondEnc,
    config: T3Config,
}

impl T3 {
    pub fn new(config: T3Config, vb: VarBuilder) -> Result<Self> {
        let gpt2_config = GPT2Config {
            vocab_size: config.vocab_size,
            n_embd: config.hidden_size,
            n_layer: config.num_layers,
            n_head: config.num_heads,
            ..Default::default()
        };

        let gpt2 = GPT2Model::new(gpt2_config, vb.pp("gpt2"))?;
        let text_emb = candle_nn::embedding(config.text_tokens_dict_size, config.hidden_size, vb.pp("text_emb"))?;
        let speech_emb = candle_nn::embedding(config.speech_tokens_dict_size, config.hidden_size, vb.pp("speech_emb"))?;
        let _text_head = candle_nn::linear(config.hidden_size, config.text_tokens_dict_size, vb.pp("text_head"))?;
        let speech_head = candle_nn::linear(config.hidden_size, config.speech_tokens_dict_size, vb.pp("speech_head"))?;
        let cond_enc = T3CondEnc::new(&config, vb.pp("cond_enc"))?;

        Ok(Self {
            gpt2,
            text_emb,
            speech_emb,
            _text_head,
            speech_head,
            cond_enc,
            config,
        })
    }

    pub fn prepare_input_embeds(&self, text_tokens: &Tensor, speech_tokens: &Tensor, spk_emb: &Tensor) -> Result<Tensor> {
        // text_tokens: (B, Lt)
        // speech_tokens: (B, Ls)
        // spk_emb: (B, E)

        let cond_emb = self.cond_enc.forward(spk_emb)?; // (B, 1, H)
        let text_emb = self.text_emb.forward(text_tokens)?; // (B, Lt, H)
        let speech_emb = self.speech_emb.forward(speech_tokens)?; // (B, Ls, H)

        // GPT2Model adds wpe (position embeddings) to inputs_embeds.
        // The original implementation relies on absolute positional embeddings provided by the GPT2 model.

        // Concatenate along time dimension (dim 1)
        // cond, text, speech
        Tensor::cat(&[&cond_emb, &text_emb, &speech_emb], 1)
    }

    pub fn generate(&self, text_tokens: &Tensor, spk_emb: &Tensor, max_gen_len: usize) -> Result<Tensor> {
        let (b, _lt) = text_tokens.dims2()?;
        let device = text_tokens.device();

        // Start with start_speech_token
        let start_token = Tensor::new(&[[self.config.start_speech_token]], device)?; // (1, 1)
        let mut speech_tokens = start_token.repeat((b, 1))?; // (B, 1)

        for _ in 0..max_gen_len {
            let embeds = self.prepare_input_embeds(text_tokens, &speech_tokens, spk_emb)?;

            // Forward pass
            let hidden_states = self.gpt2.forward_embeds(&embeds)?; // (B, L, H)

            // Get last token logits
            let last_hidden = hidden_states.i((.., hidden_states.dim(1)? - 1, ..))?; // (B, H)
            let logits = self.speech_head.forward(&last_hidden)?; // (B, Vocab)

            // Greedy decoding (argmax) for simplicity, or sample?
            // Python uses multinomial sampling.
            // Let's implement simple greedy first to ensure it runs.
            let next_token = logits.argmax(1)?.unsqueeze(1)?; // (B, 1)

            // Append
            speech_tokens = Tensor::cat(&[&speech_tokens, &next_token], 1)?;

            // Check EOS (assuming batch size 1 for simplicity of check)
            let token_scalar: u32 = next_token.i((0,0))?.to_scalar()?;
            if token_scalar == self.config.stop_speech_token {
                break;
            }
        }

        // Remove start token? Python code says:
        // "Remove EOS token if present ... return all tokens"
        // It includes start token in loop but python example starts loop with start token.
        // The output usually expects just the content.
        // Let's return the whole sequence for now.

        Ok(speech_tokens)
    }
}
