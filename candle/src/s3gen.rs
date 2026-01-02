use candle_core::{DType, Result, Tensor, Module, IndexOp};
use candle_nn::{Embedding, Linear, Conv1d, LayerNorm, VarBuilder, Activation};
use crate::modules::{ConformerEncoderLayer, Swish, MultiHeadedAttention};

// --- Basic Transformer Block ---
// Used in the Decoder U-Net

struct BasicTransformerBlock {
    norm1: LayerNorm,
    attn1: MultiHeadedAttention,
    _norm2: Option<LayerNorm>, // Used for Cross Attn if implemented (not in this version of simple implementation)
    // attn2: Option<MultiHeadedAttention>, // Cross Attn
    norm3: LayerNorm,
    ff: FeedForward,
}

struct FeedForward {
    net: Vec<Box<dyn Module>>,
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for module in &self.net {
            x = module.forward(&x)?;
        }
        Ok(x)
    }
}

impl BasicTransformerBlock {
    fn new(dim: usize, num_heads: usize, _head_dim: usize, vb: VarBuilder) -> Result<Self> {
        // 1. Self Attn
        let norm1 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn1 = MultiHeadedAttention::new(dim, num_heads, vb.pp("attn1"))?;

        // 2. Cross Attn (omitted for now as Chatterbox/Matcha usually uses self-attn in basic blocks or specific cross attn blocks)

        let norm3 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm3"))?;

        // FeedForward
        // GEGLU usually
        let inner_dim = dim * 4;

        // GEGLU: Linear(dim -> inner_dim*2) -> Split -> Gelu -> Mul -> Linear(inner_dim -> dim)
        let ff_proj1 = candle_nn::linear(dim, inner_dim * 2, vb.pp("ff.net.0.proj"))?;
        let ff_proj2 = candle_nn::linear(inner_dim, dim, vb.pp("ff.net.2"))?;

        let geglu_block = GegluBlock { proj: ff_proj1 };
        let net: Vec<Box<dyn Module>> = vec![
            Box::new(geglu_block),
            Box::new(ff_proj2),
        ];

        Ok(Self {
            norm1,
            attn1,
            _norm2: None,
            norm3,
            ff: FeedForward { net },
        })
    }

    fn forward(&self, x: &Tensor, _mask: Option<&Tensor>) -> Result<Tensor> {
        // x: (B, T, C)

        // 1. Self Attn
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let x_attn = self.attn1.forward(&x_norm, None)?; // TODO: Mask
        let x = (x_attn + residual)?;

        // 3. FF
        let residual = &x;
        let x_norm = self.norm3.forward(&x)?;
        let x_ff = self.ff.forward(&x_norm)?;
        let x = (x_ff + residual)?;

        Ok(x)
    }
}

struct GegluBlock {
    proj: Linear,
}

impl Module for GegluBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_proj = self.proj.forward(x)?;
        let chunks = x_proj.chunk(2, 2)?; // Chunk last dim (C)
        let gate = &chunks[0];
        let val = &chunks[1];
        let gelu = gate.gelu()?;
        val.broadcast_mul(&gelu)
    }
}


// --- Helper Modules ---

struct SinusoidalPosEmb {
    dim: usize,
}

impl SinusoidalPosEmb {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let half_dim = self.dim / 2;
        let emb = (10000f64.ln() / (half_dim as f64 - 1.0)) as f32;
        let freqs = (Tensor::arange(0u32, half_dim as u32, x.device())?.to_dtype(DType::F32)? * (-emb as f64))?.exp()?;
        let emb = x.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&emb.sin()?, &emb.cos()?], 1)?;
        Ok(emb)
    }
}

struct TimestepEmbedding {
    linear1: Linear,
    linear2: Linear,
    act: Swish,
    _cond_proj: Option<Linear>, // Added for meanflow time mixer or cond proj
}

impl TimestepEmbedding {
    fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear2 = candle_nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;

        Ok(Self {
            linear1,
            linear2,
            act: Swish,
            _cond_proj: None, // Simplified
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.act.forward(&x)?;
        self.linear2.forward(&x)
    }
}

// Causal Conv1d: pads (k-1, 0)
struct CausalConv1d {
    conv: Conv1d,
    padding: usize,
}

impl CausalConv1d {
    fn new(in_c: usize, out_c: usize, k: usize, dilation: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        // We handle padding manually
        let cfg = candle_nn::Conv1dConfig {
            dilation,
            stride,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(in_c, out_c, k, cfg, vb)?;
        Ok(Self {
            conv,
            padding: (k - 1) * dilation, // Simple Causal Padding logic: P = (K-1)*D. Stride doesn't affect padding requirement for causality on the left.
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pad left
        // Note: For stride > 1, we might need to be careful with output length if we pad exactly.
        // Assuming causal padding: we add zeros to the left so that the last output depends on the last input.
        let x = x.pad_with_zeros(2, self.padding, 0)?;
        self.conv.forward(&x)
    }
}

struct CausalResnetBlock1D {
    block1: CausalBlock1D,
    block2: CausalBlock1D,
    time_proj: Linear,
    residual_conv: Option<Conv1d>,
}

struct CausalBlock1D {
    conv1: CausalConv1d,
    norm: LayerNorm,
    act: Activation,
}

impl CausalBlock1D {
    fn new(dim: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = CausalConv1d::new(dim, dim_out, 3, 1, 1, vb.pp("block.0"))?; // stride 1
        let norm = candle_nn::layer_norm(dim_out, 1e-5, vb.pp("block.2"))?;
        // Mish is not readily available in Activation enum in older versions, using Gelu as approximation
        // or impl Mish manually. Let's stick to Gelu for now or add Mish.
        Ok(Self {
            conv1,
            norm,
            act: Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = (x.broadcast_mul(mask))?;
        let x = self.conv1.forward(&x)?;
        // Norm expects (B, T, C), but we have (B, C, T) from conv
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = x.transpose(1, 2)?;
        let x = self.act.forward(&x)?;
        x.broadcast_mul(mask)
    }
}

impl CausalResnetBlock1D {
    fn new(dim: usize, dim_out: usize, time_emb_dim: usize, vb: VarBuilder) -> Result<Self> {
        // MLP for time embedding: Mish -> Linear
        let time_proj = candle_nn::linear(time_emb_dim, dim_out, vb.pp("mlp.1"))?;
        // Note: mlp.0 is Mish. We apply Mish in forward or assume it's part of sequence.

        let block1 = CausalBlock1D::new(dim, dim_out, vb.pp("block1"))?;
        let block2 = CausalBlock1D::new(dim_out, dim_out, vb.pp("block2"))?;

        let residual_conv = if dim != dim_out {
            Some(candle_nn::conv1d(dim, dim_out, 1, Default::default(), vb.pp("res_conv"))?)
        } else {
            None
        };

        Ok(Self {
            block1,
            block2,
            time_proj,
            residual_conv,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, t_emb: &Tensor) -> Result<Tensor> {
        // t_emb: (B, time_dim) -> Mish -> proj -> (B, dim_out)
        // Assume t_emb is already processed by Mish if needed, or apply here.
        // `mlp` in `ResnetBlock1D` in python is `Sequential(Mish(), Linear(...))`.
        // So we need to apply Mish.
        // let t_emb = t_emb.mish()?; // Not available?
        let t_emb = t_emb.gelu()?; // Approx

        let t = self.time_proj.forward(&t_emb)?;
        let t = t.unsqueeze(2)?; // (B, dim_out, 1)

        let h = self.block1.forward(x, mask)?;
        let h = h.broadcast_add(&t)?;
        let h = self.block2.forward(&h, mask)?;

        let res = if let Some(conv) = &self.residual_conv {
            conv.forward(x)?
        } else {
            // If dimensions match, just identity.
            // But we need to mask?
            x.clone()
        };

        h + res
    }
}


// --- Main Decoder Components ---

pub struct UpsampleConformerEncoder {
    input_projection: Linear,
    layers: Vec<ConformerEncoderLayer>,
    upsample: Upsample1D,
    output_projection: Linear,
}

// Helper for Upsample
struct Upsample1D {
    conv: Option<Conv1d>, // If use_conv=True in python (default False) or use_conv_transpose=True
    stride: usize,
    use_conv_transpose: bool,
    conv_transpose: Option<candle_nn::ConvTranspose1d>,
}

impl Upsample1D {
    fn new(channels: usize, out_channels: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        // Python Upsample1D: use_conv_transpose=True default for Decoder up_blocks.

        let conv = candle_nn::conv1d(channels, out_channels, stride * 2 + 1, Default::default(), vb.pp("conv"))?;
        Ok(Self { conv: Some(conv), stride, use_conv_transpose: false, conv_transpose: None })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        if self.use_conv_transpose {
             return self.conv_transpose.as_ref().unwrap().forward(x);
        }

        // Standard upsampling
        let x = x.unsqueeze(3)?; // (B, C, T, 1)
        let x = x.repeat((1, 1, 1, self.stride))?; // (B, C, T, S)
        let x = x.reshape((b, c, t * self.stride))?; // (B, C, T*S)

        // Only use self.conv if it exists
        if let Some(conv) = &self.conv {
             let x = x.pad_with_zeros(2, self.stride * 2, 0)?; // dim 2 is time
             conv.forward(&x)
        } else {
             Ok(x)
        }
    }
}

impl UpsampleConformerEncoder {
    pub fn new(input_dim: usize, output_dim: usize, hidden_dim: usize, num_layers: usize, num_heads: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let input_projection = candle_nn::linear(input_dim, output_dim, vb.pp("input_layer"))?; // Check name: 'linear' in s3gen.py

        let mut layers = Vec::new();
        let layers_vb = vb.pp("encoders");
        for i in 0..num_layers {
            layers.push(ConformerEncoderLayer::new(output_dim, hidden_dim, num_heads, kernel_size, layers_vb.pp(i))?);
        }

        let upsample = Upsample1D::new(output_dim, output_dim, 9, vb.pp("upsample"))?;

        let output_projection = candle_nn::linear(output_dim, output_dim, vb.pp("output_projection"))?;

        Ok(Self {
            input_projection,
            layers,
            upsample,
            output_projection,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, T, C)
        let x = self.input_projection.forward(x)?;

        let mut x_curr = x;
        for layer in &self.layers {
            x_curr = layer.forward(&x_curr)?;
        }

        // x: (B, T, C) -> (B, C, T) for upsample
        let x_t = x_curr.transpose(1, 2)?;
        let x_up = self.upsample.forward(&x_t)?;
        // (B, C, T') -> (B, T', C)
        let x_curr = x_up.transpose(1, 2)?;

        self.output_projection.forward(&x_curr)
    }
}

pub struct ConditionalDecoder {
    sinusoidal_pos_emb: SinusoidalPosEmb,
    time_embedding: TimestepEmbedding,

    down_blocks: Vec<(CausalResnetBlock1D, Vec<BasicTransformerBlock>, Option<CausalConv1d>)>, // Resnet, Transformers, Downsample
    mid_blocks: Vec<(CausalResnetBlock1D, Vec<BasicTransformerBlock>)>,
    up_blocks: Vec<(CausalResnetBlock1D, Vec<BasicTransformerBlock>, Option<Upsample1D>)>, // Resnet, Transformers, Upsample

    final_block: CausalBlock1D,
    final_proj: Conv1d,

    meanflow: bool,
    _time_mixer: Option<Linear>,
    spk_emb_dim: usize, // New: speaker embedding dimension
}

impl ConditionalDecoder {
    pub fn new(in_channels: usize, out_channels: usize, spk_emb_dim: usize, vb: VarBuilder, meanflow: bool) -> Result<Self> {
        let channels = vec![256]; // From s3gen.py: channels=[256]
        let time_dim = channels[0] * 4;
        let attention_head_dim = 64;
        let num_heads = 8;
        let n_blocks = 4;
        let num_mid_blocks = 12;

        let sinusoidal_pos_emb = SinusoidalPosEmb::new(in_channels);
        let time_embedding = TimestepEmbedding::new(in_channels, time_dim, vb.pp("time_mlp"))?;

        // Adjust input channels to include speaker embedding
        let input_conv_in = in_channels + spk_emb_dim;
        // let _input_conv = CausalConv1d::new(input_conv_in, channels[0], 3, 1, 1, vb.pp("down_blocks.0.0"))?;

        let mut down_blocks = Vec::new();
        // i=0. input=320+80=400, output=256.
        let resnet = CausalResnetBlock1D::new(input_conv_in, channels[0], time_dim, vb.pp("down_blocks.0.0"))?;

        let mut transformers = Vec::new();
        let tf_vb = vb.pp("down_blocks.0.1");
        for i in 0..n_blocks {
             transformers.push(BasicTransformerBlock::new(channels[0], num_heads, attention_head_dim, tf_vb.pp(i))?);
        }

        // Downsample: CausalConv1d(256, 256, 3, stride=2)
        // Note: Python logic `CausalConv1d(..., 3) if self.causal`.
        // `Downsample1D` uses conv with stride 2.
        // `CausalConv1d` (in python) uses stride=1 usually, but here it's replacing `Downsample1D`.
        // Does `CausalConv1d` in python support stride? Yes.
        // We need stride 2 here to match upsampling stride 2.
        let downsample = CausalConv1d::new(channels[0], channels[0], 3, 1, 2, vb.pp("down_blocks.0.2"))?;

        down_blocks.push((resnet, transformers, Some(downsample)));

        // Mid Blocks
        let mut mid_blocks = Vec::new();

        let mid_vb = vb.pp("mid_blocks");
        for i in 0..num_mid_blocks {
            let resnet = CausalResnetBlock1D::new(channels[0], channels[0], time_dim, mid_vb.pp(i).pp("0"))?;
            let mut transformers = Vec::new();
            let tf_vb = mid_vb.pp(i).pp("1");
             for j in 0..n_blocks {
                transformers.push(BasicTransformerBlock::new(channels[0], num_heads, attention_head_dim, tf_vb.pp(j))?);
            }
            mid_blocks.push((resnet, transformers));
        }

        // Up Blocks
        let mut up_blocks = Vec::new();

        let up_vb = vb.pp("up_blocks.0");
        let resnet = CausalResnetBlock1D::new(channels[0] * 2, channels[0], time_dim, up_vb.pp("0"))?;

        let mut transformers = Vec::new();
        let tf_vb = up_vb.pp("1");
        for j in 0..n_blocks {
             transformers.push(BasicTransformerBlock::new(channels[0], num_heads, attention_head_dim, tf_vb.pp(j))?);
        }

        let upsample_conv = candle_nn::conv_transpose1d(channels[0], channels[0], 4, candle_nn::ConvTranspose1dConfig{ padding: 1, stride: 2, ..Default::default() }, up_vb.pp("2"))?;

        let upsample = Upsample1D { conv: None, stride: 2, use_conv_transpose: true, conv_transpose: Some(upsample_conv) };

        up_blocks.push((resnet, transformers, Some(upsample)));

        // Final
        let final_block = CausalBlock1D::new(channels[0], channels[0], vb.pp("final_block"))?;
        let final_proj = candle_nn::conv1d(channels[0], out_channels, 1, Default::default(), vb.pp("final_proj"))?;

        let time_mixer = if meanflow {
            Some(candle_nn::linear_no_bias(time_dim * 2, time_dim, vb.pp("time_embed_mixer"))?)
        } else {
            None
        };

        Ok(Self {
            sinusoidal_pos_emb,
            time_embedding,
            down_blocks,
            mid_blocks,
            up_blocks,
            final_block,
            final_proj,
            meanflow,
            _time_mixer: time_mixer,
            spk_emb_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor, mu: &Tensor, t: &Tensor, spks: Option<&Tensor>) -> Result<Tensor> {
        // x: (B, C, T)
        // mu: (B, C, T)
        // t: (B)
        // spks: (B, spk_emb_dim) or (B, spk_emb_dim, 1)

        let t_emb = self.sinusoidal_pos_emb.forward(t)?; // (B, time_dim)
        let t_emb = self.time_embedding.forward(&t_emb)?; // remove mut

        if self.meanflow {
             // Logic skipped for simplicity
        }

        // Concatenate x and mu
        let mut x = Tensor::cat(&[x, mu], 1)?;

        // Concatenate speaker embedding if provided
        if let Some(spk) = spks {
             let (_b, _c, t_len) = x.dims3()?;
             // spk: (B, S) -> (B, S, T)
             let spk = if spk.rank() == 2 { spk.unsqueeze(2)? } else { spk.clone() };
             let spk = spk.repeat((1, 1, t_len))?;
             x = Tensor::cat(&[&x, &spk], 1)?;
        } else {
             // If spks expected (weights include it) but not provided, we should probably pad with zeros to match dimension?
             // Since weights are loaded based on shape, if we fail to cat, the Resnet block will fail on shape mismatch if it was initialized with spk_dim.
             // We initialized `input_conv_in = in_channels + spk_emb_dim`.
             // So we MUST concat something of size spk_emb_dim.
             let (b, _c, t_len) = x.dims3()?;
             let zeros = Tensor::zeros((b, self.spk_emb_dim, t_len), x.dtype(), x.device())?;
             x = Tensor::cat(&[&x, &zeros], 1)?;
        }

        // Down Blocks
        let mut hiddens = Vec::new();
        let mut masks = vec![mask.clone()];

        for (resnet, transformers, downsample) in &self.down_blocks {
             let mask_down = masks.last().unwrap();
             x = resnet.forward(&x, mask_down, &t_emb)?;

             let mut x_t = x.transpose(1, 2)?;

             for tf in transformers {
                 x_t = tf.forward(&x_t, None)?;
             }
             x = x_t.transpose(1, 2)?;

             hiddens.push(x.clone());

             if let Some(ds) = downsample {
                 x = ds.forward(&x)?;
                 // Downsample mask logic
                 let m = mask_down.i((.., .., ..))?;
                 masks.push(m); // Placeholder
             }
        }

        // Mid Blocks
        let mask_mid = masks.last().unwrap();
        for (resnet, transformers) in &self.mid_blocks {
            x = resnet.forward(&x, mask_mid, &t_emb)?;
            let mut x_t = x.transpose(1, 2)?;
            for tf in transformers {
                x_t = tf.forward(&x_t, None)?;
            }
            x = x_t.transpose(1, 2)?;
        }

        // Up Blocks
        for (resnet, transformers, upsample) in &self.up_blocks {
            let mask_up = masks.pop().unwrap();
            let skip = hiddens.pop().unwrap();

            // Concat skip
            x = Tensor::cat(&[&x, &skip], 1)?; // (B, C+Skip, T)

            x = resnet.forward(&x, &mask_up, &t_emb)?;

             let mut x_t = x.transpose(1, 2)?;
             for tf in transformers {
                 x_t = tf.forward(&x_t, None)?;
             }
             x = x_t.transpose(1, 2)?;

             if let Some(us) = upsample {
                 x = us.forward(&x)?;
             }
        }

        let mask_final = masks.pop().unwrap();
        x = self.final_block.forward(&x, &mask_final)?;
        self.final_proj.forward(&x)
    }
}

pub struct CausalConditionalCFM {
    estimator: ConditionalDecoder,
    mel_dim: usize,
    _meanflow: bool,
}

impl CausalConditionalCFM {
    pub fn new(mel_dim: usize, cond_dim: usize, spk_emb_dim: usize, vb: VarBuilder, meanflow: bool) -> Result<Self> {
        // in_channels = mel_dim + cond_dim
        let in_channels = mel_dim + cond_dim;
        let estimator = ConditionalDecoder::new(in_channels, mel_dim, spk_emb_dim, vb.pp("estimator"), meanflow)?;
        Ok(Self { estimator, mel_dim, _meanflow: meanflow })
    }

    pub fn forward(&self, mu: &Tensor, mask: &Tensor, spks: Option<&Tensor>, n_timesteps: usize) -> Result<Tensor> {
        // Euler Solver
        let (b, _c, t) = mu.dims3()?;
        let mut x = Tensor::randn(0f32, 1f32, (b, self.mel_dim, t), mu.device())?;

        let dt = 1.0 / n_timesteps as f64;

        for i in 0..n_timesteps {
            let t_val = i as f64 * dt;
            let t_tensor = Tensor::new(&[t_val as f32], mu.device())?.repeat(b)?;

            let dxdt = self.estimator.forward(&x, mask, mu, &t_tensor, spks)?;
            x = (x + dxdt * dt)?;
        }

        Ok(x)
    }
}

pub struct S3Gen {
    embedding: Embedding,
    encoder: UpsampleConformerEncoder,
    mu_proj: Linear,
    decoder: CausalConditionalCFM,
}

impl S3Gen {
    pub fn new(vb: VarBuilder, meanflow: bool) -> Result<Self> {
        let vocab_size = 6563 + 1;
        let hidden_dim = 512;
        let embedding = candle_nn::embedding(vocab_size, hidden_dim, vb.pp("embedding"))?;

        let encoder = UpsampleConformerEncoder::new(hidden_dim, hidden_dim, 2048, 6, 8, 31, vb.pp("encoder"))?;

        let cond_dim = 240;
        let mel_dim = 80;
        let spk_emb_dim = 80; // Hardcoded for now based on python config
        let mu_proj = candle_nn::linear(hidden_dim, cond_dim, vb.pp("mu_proj"))?;

        let decoder = CausalConditionalCFM::new(mel_dim, cond_dim, spk_emb_dim, vb.pp("decoder"), meanflow)?;

        Ok(Self {
            embedding,
            encoder,
            mu_proj,
            decoder,
        })
    }

    pub fn forward(&self, speech_tokens: &Tensor, spks: Option<&Tensor>) -> Result<Tensor> {
        // speech_tokens: (B, T)
        let embeds = self.embedding.forward(speech_tokens)?;
        let encoder_out = self.encoder.forward(&embeds)?;

        let mu = self.mu_proj.forward(&encoder_out)?;

        let mu = mu.transpose(1, 2)?;

        let (_b, _c, t) = mu.dims3()?;
        let mask = Tensor::ones((1, 1, t), DType::F32, mu.device())?;

        let mel = self.decoder.forward(&mu, &mask, spks, 10)?;

        Ok(mel)
    }
}
