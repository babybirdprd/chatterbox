use crate::gpt2::{Config as GPT2Config, GPT2Model};
use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

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
    pub speech_cond_prompt_len: Option<usize>,
    pub use_perceiver_resampler: bool,
    pub emotion_adv: bool,
    pub n_positions: usize,
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
            speech_cond_prompt_len: Some(375),
            use_perceiver_resampler: false,
            emotion_adv: false,
            n_positions: 8196,
        }
    }
}

pub struct T3CondEnc {
    spkr_enc: Linear,
    emotion_adv_fc: Option<Linear>,
    // perceiver: Option<Perceiver>, // Placeholder for now if needed
    config: T3Config,
}

impl T3CondEnc {
    pub fn new(config: T3Config, vb: VarBuilder) -> Result<Self> {
        let spkr_enc = candle_nn::linear(
            config.speaker_embed_size,
            config.hidden_size,
            vb.pp("spkr_enc"),
        )?;

        let emotion_adv_fc = if config.emotion_adv {
            Some(candle_nn::linear_no_bias(
                1,
                config.hidden_size,
                vb.pp("emotion_adv_fc"),
            )?)
        } else {
            None
        };

        Ok(Self {
            spkr_enc,
            emotion_adv_fc,
            config,
        })
    }

    pub fn forward(
        &self,
        spk_emb: &Tensor,
        cond_prompt_speech_emb: Option<&Tensor>,
        emotion_adv: Option<&Tensor>,
    ) -> Result<Tensor> {
        // spk_emb: (B, E)
        let cond_spkr = self.spkr_enc.forward(spk_emb)?; // (B, H)
        let cond_spkr = cond_spkr.unsqueeze(1)?; // (B, 1, H)

        // Empty tensor for cat if needed (B, 0, H)
        let mut embeds = vec![cond_spkr];

        // CLAP embed (placeholder)

        // Cond prompt speech emb
        if let Some(prompt_emb) = cond_prompt_speech_emb {
            embeds.push(prompt_emb.clone());
        }

        if self.config.emotion_adv {
            if let Some(emo) = emotion_adv {
                // emo: (B, 1, 1) or (B, 1)
                let emo_proj = self.emotion_adv_fc.as_ref().unwrap().forward(emo)?; // (B, 1, H) or similar
                                                                                    // Ensure dims
                let emo_proj = if emo_proj.rank() == 2 {
                    emo_proj.unsqueeze(1)?
                } else {
                    emo_proj
                };
                embeds.push(emo_proj);
            }
        }

        Tensor::cat(&embeds, 1)
    }
}

pub struct T3 {
    gpt2: GPT2Model,
    text_emb: Embedding,
    speech_emb: Embedding,
    pos_emb: Embedding,
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
            n_positions: config.n_positions,
            ..Default::default()
        };

        let gpt2 = GPT2Model::new(gpt2_config, vb.pp("tfmr"))?;
        let text_emb = candle_nn::embedding(
            config.text_tokens_dict_size,
            config.hidden_size,
            vb.pp("text_emb"),
        )?;
        let speech_emb = candle_nn::embedding(
            config.speech_tokens_dict_size,
            config.hidden_size,
            vb.pp("speech_emb"),
        )?;
        // Use shared position embeddings from transformer
        let pos_emb = candle_nn::embedding(
            config.n_positions,
            config.hidden_size,
            vb.pp("tfmr").pp("wpe"),
        )?;
        let _text_head = candle_nn::linear_no_bias(
            config.hidden_size,
            config.text_tokens_dict_size,
            vb.pp("text_head"),
        )?;
        let speech_head = candle_nn::linear(
            config.hidden_size,
            config.speech_tokens_dict_size,
            vb.pp("speech_head"),
        )?;

        // cond_enc might need config clone
        let cond_enc = T3CondEnc::new(
            T3Config {
                text_tokens_dict_size: config.text_tokens_dict_size,
                speech_tokens_dict_size: config.speech_tokens_dict_size,
                hidden_size: config.hidden_size,
                num_layers: config.num_layers,
                num_heads: config.num_heads,
                vocab_size: config.vocab_size,
                speaker_embed_size: config.speaker_embed_size,
                start_speech_token: config.start_speech_token,
                stop_speech_token: config.stop_speech_token,
                speech_cond_prompt_len: config.speech_cond_prompt_len,
                use_perceiver_resampler: config.use_perceiver_resampler,
                emotion_adv: config.emotion_adv,
                n_positions: config.n_positions,
            },
            vb.pp("cond_enc"),
        )?;

        Ok(Self {
            gpt2,
            text_emb,
            speech_emb,
            pos_emb,
            _text_head,
            speech_head,
            cond_enc,
            config,
        })
    }

    pub fn prepare_input_embeds(
        &self,
        text_tokens: &Tensor,
        speech_tokens: &Tensor,
        spk_emb: &Tensor,
        cond_prompt_speech_tokens: Option<&Tensor>,
        emotion_adv: Option<&Tensor>,
    ) -> Result<Tensor> {
        // text_tokens: (B, Lt)
        // speech_tokens: (B, Ls)
        // spk_emb: (B, E)

        // Handle prompt tokens embedding if present
        let cond_prompt_speech_emb = if let Some(tokens) = cond_prompt_speech_tokens {
            Some(self.speech_emb.forward(tokens)?)
        } else {
            None
        };

        let cond_emb =
            self.cond_enc
                .forward(spk_emb, cond_prompt_speech_emb.as_ref(), emotion_adv)?; // (B, Lc, H)

        let text_emb = self.text_emb.forward(text_tokens)?; // (B, Lt, H)

        let speech_emb = self.speech_emb.forward(speech_tokens)?; // (B, Ls, H)

        // Concatenate all embeddings first: [Cond, Text, Speech]
        let embeds = Tensor::cat(&[&cond_emb, &text_emb, &speech_emb], 1)?; // (B, TotalLen, H)

        // Create absolute positional embeddings for the entire sequence
        let (_b, total_len, _h) = embeds.dims3()?;
        let pos_ids = Tensor::arange(0u32, total_len as u32, embeds.device())?.unsqueeze(0)?;
        let pos_emb = self.pos_emb.forward(&pos_ids)?;

        // Add position embeddings
        // Note: GPT2 adds WPE (position embeddings) to inputs_embeds
        (embeds + pos_emb)
    }

    pub fn generate(
        &self,
        text_tokens: &Tensor,
        spk_emb: &Tensor,
        cond_prompt_speech_tokens: Option<&Tensor>,
        emotion_adv: Option<&Tensor>,
        max_gen_len: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        seed: u64,
    ) -> Result<Tensor> {
        let (b, _lt) = text_tokens.dims2()?;
        let device = text_tokens.device();

        let mut logits_processor = crate::sampling::LogitsProcessor::new(
            seed,
            Some(temperature as f64),
            Some(top_p as f64),
            Some(top_k),
        );

        // Start with start_speech_token
        let start_token = Tensor::new(&[[self.config.start_speech_token]], device)?; // (1, 1)
        let mut speech_tokens_v = vec![self.config.start_speech_token];
        let mut speech_tokens_tensor = start_token.repeat((b, 1))?; // (B, 1)

        for i in 0..max_gen_len {
            if i % 10 == 0 {
                eprintln!("[T3] Generating token {}/{}...", i, max_gen_len);
            }
            let embeds = self.prepare_input_embeds(
                text_tokens,
                &speech_tokens_tensor,
                spk_emb,
                cond_prompt_speech_tokens,
                emotion_adv,
            )?;

            // Forward pass
            // IMPORTANT: We use forward_embeds_no_pos because we manually added position embeddings above
            // to ensure they are continuous across cond+text+speech
            let hidden_states = self.gpt2.forward_embeds_no_pos(&embeds)?; // (B, L, H)

            // Get last token logits
            let last_hidden = hidden_states.i((.., hidden_states.dim(1)? - 1, ..))?; // (B, H)
            let logits = self.speech_head.forward(&last_hidden)?; // (B, Vocab)

            // Sample for each batch (assuming B=1 for now as per Python sync inference)
            let logits_0 = logits.i(0)?;
            let next_token =
                logits_processor.sample(&logits_0, &speech_tokens_v, Some(repetition_penalty))?;

            speech_tokens_v.push(next_token);
            let next_token_tensor = Tensor::new(&[[next_token]], device)?;

            // Append
            speech_tokens_tensor = Tensor::cat(&[&speech_tokens_tensor, &next_token_tensor], 1)?;

            // Check EOS
            if next_token == self.config.stop_speech_token {
                break;
            }
        }

        Ok(speech_tokens_tensor)
    }
}
