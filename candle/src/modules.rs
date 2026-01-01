use candle_core::{Result, Tensor, Module};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

// Swish activation
pub struct Swish;
impl Module for Swish {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let sig = candle_nn::ops::sigmoid(xs)?;
        xs.broadcast_mul(&sig)
    }
}

// Convolution Module for Conformer
// Pointwise -> Gated Conv -> Pointwise
pub struct ConvolutionModule {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: LayerNorm,
    pointwise_conv2: Conv1d,
    activation: Swish,
}

impl ConvolutionModule {
    pub fn new(channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        // Pointwise 1
        let pointwise_conv1 = candle_nn::conv1d(channels, 2 * channels, 1, Default::default(), vb.pp("pointwise_conv1"))?;

        // Depthwise
        let cfg = Conv1dConfig {
            padding: (kernel_size - 1) / 2,
            groups: channels,
            ..Default::default()
        };
        let depthwise_conv = candle_nn::conv1d(channels, channels, kernel_size, cfg, vb.pp("depthwise_conv"))?;

        let norm = candle_nn::layer_norm(channels, 1e-5, vb.pp("norm"))?;
        let pointwise_conv2 = candle_nn::conv1d(channels, channels, 1, Default::default(), vb.pp("pointwise_conv2"))?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            norm,
            pointwise_conv2,
            activation: Swish,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T)
        let x = self.pointwise_conv1.forward(x)?;
        // GLU
        let chunks = x.chunk(2, 1)?;
        let x1 = &chunks[0];
        let x2 = &chunks[1];
        let sig = candle_nn::ops::sigmoid(x2)?;
        let x = x1.broadcast_mul(&sig)?;

        let x = self.depthwise_conv.forward(&x)?;

        // Norm expects (B, T, C)? LayerNorm in Candle usually works on last dim.
        // Conv1d output is (B, C, T).
        let x = x.transpose(1, 2)?; // (B, T, C)
        let x = self.norm.forward(&x)?;
        let x = x.transpose(1, 2)?; // (B, C, T)

        let x = self.activation.forward(&x)?;
        self.pointwise_conv2.forward(&x)
    }
}

// FeedForward Module
pub struct FeedForwardModule {
    linear1: Linear,
    dropout: f64, // Todo: implement dropout
    linear2: Linear,
    activation: Swish,
    norm: LayerNorm,
}

impl FeedForwardModule {
    pub fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;
        let linear1 = candle_nn::linear(dim, hidden_dim, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(hidden_dim, dim, vb.pp("linear2"))?;

        Ok(Self {
            linear1,
            dropout: 0.1,
            linear2,
            activation: Swish,
            norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, C)
        let residual = x;
        let x = self.norm.forward(x)?;
        let x = self.linear1.forward(&x)?;
        let x = self.activation.forward(&x)?;
        // dropout
        let x = self.linear2.forward(&x)?;
        // dropout
        x.broadcast_add(residual)
    }
}

// MultiHeadedAttention
pub struct MultiHeadedAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadedAttention {
    pub fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let linear_q = candle_nn::linear(dim, dim, vb.pp("linear_q"))?;
        let linear_k = candle_nn::linear(dim, dim, vb.pp("linear_k"))?;
        let linear_v = candle_nn::linear(dim, dim, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear(dim, dim, vb.pp("linear_out"))?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // x: (B, T, C)
        let (b, t, c) = x.dims3()?;
        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;

        let q = q.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?; // (B, H, T, D)
        let k = k.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let k_t = k.transpose(2, 3)?;
        let att = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?; // (B, H, T, T)

        let att = if let Some(mask) = mask {
            att.broadcast_add(mask)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax(&att, 3)?;
        let out = att.matmul(&v)?; // (B, H, T, D)

        let out = out.transpose(1, 2)?.reshape((b, t, c))?; // (B, T, C)
        self.linear_out.forward(&out)
    }
}

// ConformerEncoderLayer
pub struct ConformerEncoderLayer {
    ff1: FeedForwardModule,
    self_attn: MultiHeadedAttention,
    conv_module: ConvolutionModule,
    ff2: FeedForwardModule,
    norm: LayerNorm,
    // dropout: f64,
}

impl ConformerEncoderLayer {
    pub fn new(dim: usize, hidden_dim: usize, num_heads: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let ff1 = FeedForwardModule::new(dim, hidden_dim, vb.pp("ff1"))?;
        let self_attn = MultiHeadedAttention::new(dim, num_heads, vb.pp("self_attn"))?;
        let conv_module = ConvolutionModule::new(dim, kernel_size, vb.pp("conv_module"))?;
        let ff2 = FeedForwardModule::new(dim, hidden_dim, vb.pp("ff2"))?;
        let norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            ff1,
            self_attn,
            conv_module,
            ff2,
            norm,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, C)

        // FF1
        let residual = x;
        let x = self.ff1.forward(x)?;
        let x = (x * 0.5)?; // Macaron style half step?
        let x = x.broadcast_add(residual)?;

        // Self Attention
        let residual = &x;
        let x = self.norm.forward(&x)?; // Norm before attention in Conformer? Or after? Usually before or in sandwich
        let x = self.self_attn.forward(&x, None)?;
        let x = x.broadcast_add(residual)?;

        // Conv
        // x: (B, T, C) -> (B, C, T) for conv
        let residual = &x;
        let x_t = x.transpose(1, 2)?;
        let x_conv = self.conv_module.forward(&x_t)?;
        let x = x_conv.transpose(1, 2)?;
        let x = x.broadcast_add(residual)?;

        // FF2
        let residual = &x;
        let x = self.ff2.forward(&x)?;
        let x = (x * 0.5)?;
        let x = x.broadcast_add(residual)?;

        let x = self.norm.forward(&x)?;
        Ok(x)
    }
}
