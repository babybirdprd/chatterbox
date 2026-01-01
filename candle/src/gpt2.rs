use candle_core::{DType, IndexOp, Result, Tensor, Module};
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
    n_embd: usize,
    scale: bool,
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
            n_embd,
            scale,
        })
    }

    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.n_head, c / self.n_head))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let k_t = k.transpose(2, 3)?;
        let mut att = (q.matmul(&k_t)? / (c as f64 / self.n_head as f64).sqrt())?;

        if let Some(bias) = bias {
            att = att.broadcast_add(bias)?;
        }

        att = candle_nn::ops::softmax(&att, 3)?;
        let y = att.matmul(&v)?;
        let y = y.permute((0, 2, 1, 3))?.reshape((b, t, c))?;

        self.c_proj.forward(&y)
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

    fn forward(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x, bias)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }
}

pub struct GPT2Model {
    wte: Embedding,
    wpe: Embedding,
    h: Vec<Block>,
    ln_f: LayerNorm,
    config: Config,
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
            config,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, t) = input_ids.dims2()?;
        let input_embeds = self.wte.forward(input_ids)?;
        let position_ids = Tensor::arange(0, t as u32, input_ids.device())?.unsqueeze(0)?;
        let position_embeds = self.wpe.forward(&position_ids)?;

        let mut hidden_states = (input_embeds + position_embeds)?;

        let mask_indexes = Tensor::arange(0, t as u32, input_ids.device())?;
        let mask_indexes_row = mask_indexes.unsqueeze(1)?;
        let mask_indexes_col = mask_indexes.unsqueeze(0)?;
        let mask = mask_indexes_row.ge(&mask_indexes_col)?;

        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
        let mask = mask.to_dtype(DType::F32)?;
        let mask = ((mask - 1.0)? * 1e9)?;

        for block in &self.h {
            hidden_states = block.forward(&hidden_states, Some(&mask))?;
        }

        self.ln_f.forward(&hidden_states)
    }

    // Helper to allow custom embeddings input
    pub fn forward_embeds(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b, t, c) = inputs_embeds.dims3()?;
        let position_ids = Tensor::arange(0, t as u32, inputs_embeds.device())?.unsqueeze(0)?;
        let position_embeds = self.wpe.forward(&position_ids)?;

        let mut hidden_states = (inputs_embeds + position_embeds)?;

        let mask_indexes = Tensor::arange(0, t as u32, inputs_embeds.device())?;
        let mask_indexes_row = mask_indexes.unsqueeze(1)?.broadcast_as((t, t))?;
        let mask_indexes_col = mask_indexes.unsqueeze(0)?.broadcast_as((t, t))?;
        let mask = mask_indexes_row.ge(&mask_indexes_col)?;

        let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
        let mask = mask.to_dtype(DType::F32)?;
        let mask = ((mask - 1.0)? * 1e9)?;

        for block in &self.h {
            hidden_states = block.forward(&hidden_states, Some(&mask))?;
        }

        self.ln_f.forward(&hidden_states)
    }
}
