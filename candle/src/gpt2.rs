use candle_core::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, Embedding, LayerNorm, Linear, VarBuilder};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub n_positions: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub activation_function: Activation,
    pub layer_norm_epsilon: f64,
    pub n_inner: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            activation_function: Activation::Gelu,
            layer_norm_epsilon: 1e-5,
            n_inner: None,
        }
    }
}

fn conv1d(nf: usize, nx: usize, vb: VarBuilder) -> Result<Linear> {
    // let weight = vb.get((nf, nx), "weight")?; // (out, in)
    // Transpose
    let weight = vb.get((nx, nf), "weight")?;
    let weight = weight.transpose(0, 1)?;
    let bias = vb.get(nf, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    act: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n_inner = cfg.n_inner.unwrap_or(4 * cfg.n_embd);
        let c_fc = conv1d(n_inner, cfg.n_embd, vb.pp("c_fc"))?;
        let c_proj = conv1d(cfg.n_embd, n_inner, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_proj,
            act: cfg.activation_function,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = self.act.forward(&x)?;
        self.c_proj.forward(&x)
    }
}

struct Attention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    _n_embd: usize,
    _scale: bool,
}

impl Attention {
    fn new(cfg: &Config, scale: bool, vb: VarBuilder) -> Result<Self> {
        let n_embd = cfg.n_embd;
        let n_head = cfg.n_head;
        let c_attn = conv1d(3 * n_embd, n_embd, vb.pp("c_attn"))?;
        let c_proj = conv1d(n_embd, n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_attn,
            c_proj,
            n_head,
            _n_embd: n_embd,
            _scale: scale,
        })
    }

    fn forward(&self, x: &Tensor, bias: Option<&Tensor>, layer_past: Option<&(Tensor, Tensor)>) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.n_head, c / self.n_head))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?; // (3, B, n_head, T, head_dim)
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let (k, v) = match layer_past {
            Some((past_k, past_v)) => {
                let k = Tensor::cat(&[past_k, &k], 2)?;
                let v = Tensor::cat(&[past_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };

        let current_cache = Some((k.clone(), v.clone()));

        let k_t = k.transpose(2, 3)?;
        // q: (B, H, T_q, D)
        // k_t: (B, H, D, T_k)
        // att: (B, H, T_q, T_k)
        let mut att = (q.matmul(&k_t)? / (c as f64 / self.n_head as f64).sqrt())?;

        if let Some(bias) = bias {
            // bias: (1, 1, T_q, T_k) or (1, 1, 1, T_k) depending on masking
            // When using cache, T_q=1, T_k=L_past+1.
            // Bias should match T_k.
            // Caller must ensure bias is broadcastable.
            att = att.broadcast_add(bias)?;
        }

        att = candle_nn::ops::softmax(&att, 3)?;
        let y = att.matmul(&v)?;
        let y = y.permute((0, 2, 1, 3))?.reshape((b, t, c))?;

        let out = self.c_proj.forward(&y)?;
        Ok((out, current_cache))
    }
}

struct Block {
    ln_1: LayerNorm,
    attn: Attention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let attn = Attention::new(cfg, true, vb.pp("attn"))?;
        let ln_2 = candle_nn::layer_norm(cfg.n_embd, cfg.layer_norm_epsilon, vb.pp("ln_2"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        bias: Option<&Tensor>,
        layer_past: Option<&(Tensor, Tensor)>
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let (x, past) = self.attn.forward(&x, bias, layer_past)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        Ok((x, past))
    }
}

pub struct GPT2Model {
    wte: Embedding,
    wpe: Embedding,
    h: Vec<Block>,
    ln_f: LayerNorm,
    _config: Config,
}

impl GPT2Model {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(config.n_positions, config.n_embd, vb.pp("wpe"))?;

        let mut h = Vec::new();
        let blocks_vb = vb.pp("h");
        for i in 0..config.n_layer {
            h.push(Block::new(&config, blocks_vb.pp(i))?);
        }

        let ln_f = candle_nn::layer_norm(config.n_embd, config.layer_norm_epsilon, vb.pp("ln_f"))?;

        Ok(Self {
            wte,
            wpe,
            h,
            ln_f,
            _config: config,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, past_key_values: Option<&Vec<(Tensor, Tensor)>>) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (_b, t) = input_ids.dims2()?;
        let past_len = if let Some(past) = past_key_values {
            if !past.is_empty() {
                past[0].0.dim(2)?
            } else {
                0
            }
        } else {
            0
        };

        let input_embeds = self.wte.forward(input_ids)?;
        let position_ids = Tensor::arange(past_len as u32, (past_len + t) as u32, input_ids.device())?.unsqueeze(0)?;
        let position_embeds = self.wpe.forward(&position_ids)?;

        let mut hidden_states = (input_embeds + position_embeds)?;

        // Masking
        // If we are using cache (past_len > 0), t is typically 1 (decoding).
        // We attend to past_len + t tokens.
        // The mask should be (1, 1, 1, past_len + t) or similar?
        // Wait, standard causal mask: each token i can attend to 0..i.
        // If we have past, we are at index `past_len + i`. We attend to 0..past_len+i.
        // So we need a mask that allows attention to all previous positions.
        // Since we process `t` new tokens, for token `j` in `0..t` (absolute `past_len + j`),
        // it attends to `0..past_len + j`.
        // If `t=1`, it attends to `0..past_len+1`. All ones.
        // So mask is just all ones?
        // Wait, standard GPT2 mask is strictly causal.
        // If `t > 1` (e.g. prefix processing), we need causal mask for the new `t` tokens,
        // but they can all attend to `past`.

        let mask = if t == 1 && past_len > 0 {
             // Decoding step: attend to everything. No mask needed (or all zeros which means allow).
             // But we might need to match shape for broadcast.
             // bias shape: (1, 1, 1, total_len) or just None?
             None
        } else {
            // Context encoding (or initial step)
            let total_len = past_len + t;
            let mask_indexes = Tensor::arange(past_len as u32, total_len as u32, input_ids.device())?; // (t)
            let mask_indexes_row = mask_indexes.unsqueeze(1)?.broadcast_as((t, total_len))?; // (t, total_len)

            let key_indexes = Tensor::arange(0u32, total_len as u32, input_ids.device())?; // (total_len)
            let mask_indexes_col = key_indexes.unsqueeze(0)?.broadcast_as((t, total_len))?; // (t, total_len)

            let mask = mask_indexes_row.ge(&mask_indexes_col)?; // (t, total_len)

            let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, t, total_len)
            let mask = mask.to_dtype(hidden_states.dtype())?;
            let wide_mask_val = if hidden_states.dtype() == DType::F16 {
                1e4
            } else {
                1e9
            };
            let mask = ((mask - 1.0)? * wide_mask_val)?;
            Some(mask)
        };

        let mut next_past = Vec::new();
        for (i, block) in self.h.iter().enumerate() {
            let layer_past = past_key_values.map(|p| &p[i]);
            let (new_h, past) = block.forward(&hidden_states, mask.as_ref(), layer_past)?;
            hidden_states = new_h;
            if let Some(p) = past {
                next_past.push(p);
            }
        }

        let out = self.ln_f.forward(&hidden_states)?;
        Ok((out, next_past))
    }

    // Helper to allow custom embeddings input.
    // DOES NOT ADD POS EMBEDDINGS automatically.
    // DOES NOT UPDATE PAST_KEY_VALUES automatically unless passed.
    pub fn forward_embeds_no_pos(
        &self,
        inputs_embeds: &Tensor,
        past_key_values: Option<&Vec<(Tensor, Tensor)>>
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (_b, t, _c) = inputs_embeds.dims3()?;

        let past_len = if let Some(past) = past_key_values {
            if !past.is_empty() {
                past[0].0.dim(2)?
            } else {
                0
            }
        } else {
            0
        };

        let mut hidden_states = inputs_embeds.clone();

        let mask = if t == 1 && past_len > 0 {
             None
        } else {
            let total_len = past_len + t;
            let mask_indexes = Tensor::arange(past_len as u32, total_len as u32, inputs_embeds.device())?;
            let mask_indexes_row = mask_indexes.unsqueeze(1)?.broadcast_as((t, total_len))?;

            let key_indexes = Tensor::arange(0u32, total_len as u32, inputs_embeds.device())?;
            let mask_indexes_col = key_indexes.unsqueeze(0)?.broadcast_as((t, total_len))?;

            let mask = mask_indexes_row.ge(&mask_indexes_col)?;
            let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
            let mask = mask.to_dtype(inputs_embeds.dtype())?;
            let wide_mask_val = if inputs_embeds.dtype() == DType::F16 {
                1e4
            } else {
                1e9
            };
            let mask = ((mask - 1.0)? * wide_mask_val)?;
            Some(mask)
        };

        let mut next_past = Vec::new();
        for (i, block) in self.h.iter().enumerate() {
            let layer_past = past_key_values.map(|p| &p[i]);
            let (new_h, past) = block.forward(&hidden_states, mask.as_ref(), layer_past)?;
            hidden_states = new_h;
            if let Some(p) = past {
                next_past.push(p);
            }
        }

        let out = self.ln_f.forward(&hidden_states)?;
        Ok((out, next_past))
    }
}
