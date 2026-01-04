//! S3Tokenizer V2 implementation based on SenseVoice-Large.
//! Ported derived from: https://github.com/xingchensong/S3Tokenizer

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_codebook_size: usize,
    pub use_sdpa: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_mels: 128,
            n_audio_ctx: 1500,
            n_audio_state: 1280,
            n_audio_head: 20,
            n_audio_layer: 6,
            n_codebook_size: 6561, // 3^8
            use_sdpa: false,
        }
    }
}

/// Finite Scalar Quantization Codebook
pub struct FSQCodebook {
    project_down: Linear,
    level: usize,
}

impl FSQCodebook {
    pub fn new(dim: usize, level: usize, vb: VarBuilder) -> Result<Self> {
        let project_down = candle_nn::linear(dim, 8, vb.pp("project_down"))?;
        Ok(Self {
            project_down,
            level,
        })
    }

    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let x = x.reshape((b * t, d))?;

        // h = project_down(x).tanh()
        let h = self.project_down.forward(&x)?.tanh()?;

        // h = h * 0.9990000128746033
        let h = (h * 0.9990000128746033)?;

        // h = h.round() + 1
        let h = (h.round()? + 1.0)?;

        // levels = level ^ arange(8)
        let device = x.device();
        let powers: Vec<f32> = (0..8).map(|i| (self.level as f32).powi(i as i32)).collect();
        let powers = Tensor::from_vec(powers, (8,), device)?.to_dtype(h.dtype())?;

        // mu = sum(h * powers)
        let mu = h.broadcast_mul(&powers.unsqueeze(0)?)?.sum(1)?;

        mu.reshape((b, t))?.to_dtype(DType::U32)
    }
}

/// Rotary Positional Embedding Utilities
pub fn precompute_freqs_cis(dim: usize, end: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let theta: f32 = 10000.0;
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0 / (theta.powf(i as f32 / dim as f32)))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;

    let t = Tensor::arange(0.0, end as f32, device)?;
    let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // (end, dim/2)

    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // Concatenate for simpler application to half-vectors
    let cos = Tensor::cat(&[&cos, &cos], 1)?; // (end, dim)
    let sin = Tensor::cat(&[&sin, &sin], 1)?; // (end, dim)

    Ok((cos, sin))
}

pub fn apply_rotary_emb(
    xq: &Tensor,
    xk: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // xq: (B, T, H, D), cos: (T, D)
    let (_b, _t, _h, d_full) = xq.dims4()?;
    let d = d_full / 2;

    let xq_l = xq.narrow(D::Minus1, 0, d)?;
    let xq_r = xq.narrow(D::Minus1, d, d)?;

    let xk_l = xk.narrow(D::Minus1, 0, d)?;
    let xk_r = xk.narrow(D::Minus1, d, d)?;

    // xq_r_rot = cat(-xq_r, xq_l)
    let xq_rot = Tensor::cat(&[&xq_r.neg()?, &xq_l], D::Minus1)?;
    let xk_rot = Tensor::cat(&[&xk_r.neg()?, &xk_l], D::Minus1)?;

    // Broadcast cos/sin: (1, T, 1, D)
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    let xq_unbound = (xq.broadcast_mul(&cos)? + xq_rot.broadcast_mul(&sin)?)?;
    let xk_unbound = (xk.broadcast_mul(&cos)? + xk_rot.broadcast_mul(&sin)?)?;

    Ok((xq_unbound, xk_unbound))
}

/// Feedforward Sequential Memory Network Block
pub struct FSMNMultiHeadAttention {
    n_head: usize,
    _n_state: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    fsmn_block: Conv1d,
    kernel_size: usize,
}

impl FSMNMultiHeadAttention {
    pub fn new(n_state: usize, n_head: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let query = candle_nn::linear_no_bias(n_state, n_state, vb.pp("query"))?;
        let key = candle_nn::linear_no_bias(n_state, n_state, vb.pp("key"))?;
        let value = candle_nn::linear_no_bias(n_state, n_state, vb.pp("value"))?;
        let out = candle_nn::linear(n_state, n_state, vb.pp("out"))?;

        let fsmn_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: n_state,
            ..Default::default()
        };
        // FSMN block uses no bias in Python
        let fsmn_weight = vb
            .pp("fsmn_block")
            .get((n_state, 1, kernel_size), "weight")?;
        let fsmn_block = Conv1d::new(fsmn_weight, None, fsmn_config);

        Ok(Self {
            n_head,
            _n_state: n_state,
            query,
            key,
            value,
            out,
            fsmn_block,
            kernel_size,
        })
    }

    fn forward_fsmn(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // x: (B, T, D)
        let mut inputs = x.clone();
        if let Some(m) = mask {
            inputs = inputs.broadcast_mul(m)?;
        }

        // Pad for same-length FSMN
        let left_pad = (self.kernel_size - 1) / 2;
        let right_pad = self.kernel_size - 1 - left_pad;

        let x_padded = inputs
            .transpose(1, 2)?
            .pad_with_zeros(D::Minus1, left_pad, right_pad)?;
        let mut fsmn_out = self.fsmn_block.forward(&x_padded)?;
        fsmn_out = fsmn_out.transpose(1, 2)?;

        let res = (fsmn_out + inputs)?;
        if let Some(m) = mask {
            res.broadcast_mul(m)
        } else {
            Ok(res)
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        mask_pad: Option<&Tensor>,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let head_dim = d / self.n_head;

        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        let mut q = q.reshape((b, t, self.n_head, head_dim))?;
        let mut k = k.reshape((b, t, self.n_head, head_dim))?;
        let v_reshaped = v.reshape((b, t, self.n_head, head_dim))?;

        if let (Some(c), Some(s)) = (cos, sin) {
            let (q_rot, k_rot) = apply_rotary_emb(&q, &k, c, s)?;
            q = q_rot;
            k = k_rot;
        }

        let fsm_memory = self.forward_fsmn(&v, mask_pad)?;

        // Attention calculation
        let q = q.transpose(1, 2)?.contiguous()?; // (B, H, T, D/H)
        let k = k.transpose(1, 2)?.transpose(2, 3)?.contiguous()?; // (B, H, D/H, T)

        let scale = (head_dim as f32).powf(-0.5);
        let mut qk = (q.matmul(&k)? * (scale as f64))?;

        if let Some(m) = mask {
            qk = qk.broadcast_add(m)?;
        }

        let soft_qk = candle_nn::ops::softmax(&qk, D::Minus1)?;
        let v_heads = v_reshaped.transpose(1, 2)?.contiguous()?; // (B, H, T, D/H)
        let attn_out = soft_qk.matmul(&v_heads)?; // (B, H, T, D/H)

        let wv = attn_out.transpose(1, 2)?.reshape((b, t, d))?;
        let out = self.out.forward(&wv)?;

        out + fsm_memory
    }
}

pub struct ResidualAttentionBlock {
    attn: FSMNMultiHeadAttention,
    attn_ln: LayerNorm,
    mlp: Vec<Box<dyn Module>>,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    pub fn new(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let attn = FSMNMultiHeadAttention::new(n_state, n_head, 31, vb.pp("attn"))?;
        let attn_ln = candle_nn::layer_norm(n_state, 1e-5, vb.pp("attn_ln"))?;

        let n_mlp = n_state * 4;
        let mlp_0 = candle_nn::linear(n_state, n_mlp, vb.pp("mlp.0"))?;
        let mlp_2 = candle_nn::linear(n_mlp, n_state, vb.pp("mlp.2"))?;

        let mlp_ln = candle_nn::layer_norm(n_state, 1e-5, vb.pp("mlp_ln"))?;

        Ok(Self {
            attn,
            attn_ln,
            mlp: vec![Box::new(mlp_0), Box::new(mlp_2)],
            mlp_ln,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        mask_pad: Option<&Tensor>,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x_ln = self.attn_ln.forward(x)?;
        let attn_out = self.attn.forward(&x_ln, mask, mask_pad, cos, sin)?;
        let x = (x + attn_out)?;

        let x_ln_mlp = self.mlp_ln.forward(&x)?;
        let mut res = x_ln_mlp;
        // Manual MLP forward to handle GELU easily if not using Sequential
        res = self.mlp[0].forward(&res)?;
        res = res.gelu()?;
        res = self.mlp[1].forward(&res)?;

        x + res
    }
}

pub struct AudioEncoderV2 {
    conv1: Conv1d,
    conv2: Conv1d,
    blocks: Vec<ResidualAttentionBlock>,
    cos: Tensor,
    sin: Tensor,
    _stride: usize,
}

impl AudioEncoderV2 {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let n_state = config.n_audio_state;
        let c1_config = Conv1dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv1d(config.n_mels, n_state, 3, c1_config, vb.pp("conv1"))?;

        let c2_config = Conv1dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv2 = candle_nn::conv1d(n_state, n_state, 3, c2_config, vb.pp("conv2"))?;

        let mut blocks = Vec::new();
        let vb_blocks = vb.pp("blocks");
        for i in 0..config.n_audio_layer {
            blocks.push(ResidualAttentionBlock::new(
                n_state,
                config.n_audio_head,
                vb_blocks.pp(&i.to_string()),
            )?);
        }

        // Precompute RoPE: 64 is head_dim, 2048 matches Python's 1024*2
        let (cos, sin) = precompute_freqs_cis(n_state / config.n_audio_head, 2048, vb.device())?;

        Ok(Self {
            conv1,
            conv2,
            blocks,
            cos,
            sin,
            _stride: 2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, Mel, T)
        let mut x = self.conv1.forward(x)?.gelu()?;
        x = self.conv2.forward(&x)?.gelu()?;

        x = x.transpose(1, 2)?; // (B, T/4, D)
        let (_, t, _) = x.dims3()?;

        let cos = self.cos.narrow(0, 0, t)?;
        let sin = self.sin.narrow(0, 0, t)?;

        for block in &self.blocks {
            x = block.forward(&x, None, None, Some(&cos), Some(&sin))?;
        }

        Ok(x)
    }
}

pub struct S3TokenizerV2 {
    encoder: AudioEncoderV2,
    quantizer: FSQCodebook,
}

impl S3TokenizerV2 {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("s3_model");
        let encoder = AudioEncoderV2::new(config, vb.pp("encoder"))?;
        let quantizer = FSQCodebook::new(config.n_audio_state, 3, vb.pp("quantizer._codebook"))?;

        Ok(Self { encoder, quantizer })
    }

    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        let hidden = self.encoder.forward(mel)?;
        self.quantizer.encode(&hidden)
    }
}
