use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Activation, Conv1d, Embedding, LayerNorm, Linear, VarBuilder};

// --- Basic Transformer Block ---

struct DecoderAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl DecoderAttention {
    fn new(dim: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = num_heads * head_dim;
        let to_q = candle_nn::linear_no_bias(dim, inner_dim, vb.pp("to_q"))?;
        let to_k = candle_nn::linear_no_bias(dim, inner_dim, vb.pp("to_k"))?;
        let to_v = candle_nn::linear_no_bias(dim, inner_dim, vb.pp("to_v"))?;
        let to_out = candle_nn::linear(inner_dim, dim, vb.pp("to_out.0"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, _mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;

        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let k_t = k.transpose(2, 3)?;
        let att = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;

        let att = candle_nn::ops::softmax(&att, 3)?;
        let out = att.matmul(&v)?;

        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.num_heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

struct BasicTransformerBlock {
    norm1: LayerNorm,
    attn1: DecoderAttention,
    _norm2: Option<LayerNorm>,
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
    fn new(dim: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let attn1 = DecoderAttention::new(dim, num_heads, head_dim, vb.pp("attn1"))?;
        let norm3 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm3"))?;
        let inner_dim = dim * 4;
        let ff_proj1 = candle_nn::linear(dim, inner_dim, vb.pp("ff.net.0.proj"))?;
        let ff_proj2 = candle_nn::linear(inner_dim, dim, vb.pp("ff.net.2"))?;
        let gelu_block = GeluBlock { proj: ff_proj1 };
        let net: Vec<Box<dyn Module>> = vec![Box::new(gelu_block), Box::new(ff_proj2)];

        Ok(Self {
            norm1,
            attn1,
            _norm2: None,
            norm3,
            ff: FeedForward { net },
        })
    }

    fn forward(&self, x: &Tensor, _mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let x_norm = self.norm1.forward(x)?;
        let x_attn = self.attn1.forward(&x_norm, None)?;
        let x = (x_attn + residual)?;
        let residual = &x;
        let x_norm = self.norm3.forward(&x)?;
        let x_ff = self.ff.forward(&x_norm)?;
        let x = (x_ff + residual)?;
        Ok(x)
    }
}

struct GeluBlock {
    proj: Linear,
}

impl Module for GeluBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)?.gelu()
    }
}

struct Swish;
impl Module for Swish {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&candle_nn::ops::sigmoid(xs)?)
    }
}

struct EspnetRelPositionalEncoding {
    xscale: f64,
}

impl EspnetRelPositionalEncoding {
    fn new(d_model: usize, _max_len: usize) -> Result<Self> {
        Ok(Self {
            xscale: (d_model as f64).sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, t, c) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let position = Tensor::arange(0u32, t as u32, device)?
            .to_dtype(dtype)?
            .unsqueeze(1)?;

        let mut pe = Tensor::zeros((t, c), dtype, device)?;
        let log_10000 = 10000.0f64.ln();

        for i in 0..(c / 2) {
            let div_term = (-((i * 2) as f64 * log_10000 / c as f64)).exp();
            let pos_dt = (&position * div_term)?;
            pe = pe.slice_assign(&[0..t, (i * 2)..(i * 2 + 1)], &pos_dt.sin()?)?;
            pe = pe.slice_assign(&[0..t, (i * 2 + 1)..(i * 2 + 2)], &pos_dt.cos()?)?;
        }

        let x_scaled = (x * self.xscale)?;
        Ok((x_scaled, pe.unsqueeze(0)?))
    }
}

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
        let freqs = (Tensor::arange(0u32, half_dim as u32, x.device())?.to_dtype(DType::F32)?
            * (-emb as f64))?
            .exp()?;
        let emb = x.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&emb.sin()?, &emb.cos()?], 1)?;
        Ok(emb)
    }
}

struct TimestepEmbedding {
    linear1: Linear,
    linear2: Linear,
    act: Swish,
}

impl TimestepEmbedding {
    fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let linear2 = candle_nn::linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?;
        Ok(Self {
            linear1,
            linear2,
            act: Swish,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.act.forward(&x)?;
        self.linear2.forward(&x)
    }
}

struct CausalConv1d {
    conv: Conv1d,
    padding: usize,
}

impl CausalConv1d {
    fn new(
        in_c: usize,
        out_c: usize,
        k: usize,
        dilation: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = candle_nn::Conv1dConfig {
            dilation,
            stride,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(in_c, out_c, k, cfg, vb)?;
        Ok(Self {
            conv,
            padding: (k - 1) * dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.pad_with_zeros(2, self.padding, 0)?;
        self.conv.forward(&x)
    }
}

struct CausalBlock1D {
    conv1: CausalConv1d,
    norm: LayerNorm,
    act: Activation,
}

impl CausalBlock1D {
    fn new(dim: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = CausalConv1d::new(dim, dim_out, 3, 1, 1, vb.pp("block.0"))?;
        let norm = candle_nn::layer_norm(dim_out, 1e-5, vb.pp("block.2"))?;
        Ok(Self {
            conv1,
            norm,
            act: Activation::Gelu,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let x = x.broadcast_mul(mask)?;
        let x = self.conv1.forward(&x)?;
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = x.transpose(1, 2)?;
        let x = self.act.forward(&x)?;
        x.broadcast_mul(mask)
    }
}

struct CausalResnetBlock1D {
    block1: CausalBlock1D,
    block2: CausalBlock1D,
    time_proj: Linear,
    residual_conv: Option<Conv1d>,
}

impl CausalResnetBlock1D {
    fn new(dim: usize, dim_out: usize, time_emb_dim: usize, vb: VarBuilder) -> Result<Self> {
        let time_proj = candle_nn::linear(time_emb_dim, dim_out, vb.pp("mlp.1"))?;
        let block1 = CausalBlock1D::new(dim, dim_out, vb.pp("block1"))?;
        let block2 = CausalBlock1D::new(dim_out, dim_out, vb.pp("block2"))?;
        let residual_conv = if dim != dim_out {
            Some(candle_nn::conv1d(
                dim,
                dim_out,
                1,
                Default::default(),
                vb.pp("res_conv"),
            )?)
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
        let t_emb = t_emb.gelu()?;
        let t = self.time_proj.forward(&t_emb)?.unsqueeze(2)?;
        let h = self.block1.forward(x, mask)?;
        let h = h.broadcast_add(&t)?;
        let h = self.block2.forward(&h, mask)?;
        let res = if let Some(conv) = &self.residual_conv {
            conv.forward(x)?
        } else {
            x.clone()
        };
        h + res
    }
}

struct PositionwiseFeedForward {
    w_1: Linear,
    w_2: Linear,
    swish: Swish,
}

impl PositionwiseFeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w_1 = candle_nn::linear(dim, hidden_dim, vb.pp("w_1"))?;
        let w_2 = candle_nn::linear(hidden_dim, dim, vb.pp("w_2"))?;
        Ok(Self {
            w_1,
            w_2,
            swish: Swish,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.w_1.forward(x)?;
        let x = self.swish.forward(&x)?;
        self.w_2.forward(&x)
    }
}

struct RelPositionMultiHeadedAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl RelPositionMultiHeadedAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let linear_q = candle_nn::linear(dim, dim, vb.pp("linear_q"))?;
        let linear_k = candle_nn::linear(dim, dim, vb.pp("linear_k"))?;
        let linear_v = candle_nn::linear(dim, dim, vb.pp("linear_v"))?;
        let linear_out = candle_nn::linear(dim, dim, vb.pp("linear_out"))?;
        let linear_pos = candle_nn::linear_no_bias(dim, dim, vb.pp("linear_pos"))?;
        let pos_bias_u = vb.get((num_heads, head_dim), "pos_bias_u")?;
        let pos_bias_v = vb.get((num_heads, head_dim), "pos_bias_v")?;
        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor, _mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let q = self.linear_q.forward(x)?;
        let k = self.linear_k.forward(x)?;
        let v = self.linear_v.forward(x)?;
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let p = self.linear_pos.forward(pos_emb)?;
        let p = p
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q_u = q.broadcast_add(&self.pos_bias_u.unsqueeze(0)?.unsqueeze(2)?)?;
        let q_v = q.broadcast_add(&self.pos_bias_v.unsqueeze(0)?.unsqueeze(2)?)?;
        let matrix_ac = q_u.matmul(&k.transpose(2, 3)?)?;
        let matrix_bd = q_v.matmul(&p.transpose(2, 3)?)?;
        let scores = (matrix_ac.broadcast_add(&matrix_bd)? / (self.head_dim as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let x = attn.matmul(&v)?;
        let x = x.transpose(1, 2)?.reshape((b, t, c))?;
        self.linear_out.forward(&x)
    }
}

struct ConformerLayer {
    self_attn: RelPositionMultiHeadedAttention,
    feed_forward: PositionwiseFeedForward,
    norm_mha: LayerNorm,
    norm_ff: LayerNorm,
}

impl ConformerLayer {
    fn new(dim: usize, hidden_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let self_attn = RelPositionMultiHeadedAttention::new(dim, num_heads, vb.pp("self_attn"))?;
        let feed_forward = PositionwiseFeedForward::new(dim, hidden_dim, vb.pp("feed_forward"))?;
        let norm_mha = candle_nn::layer_norm(dim, 1e-12, vb.pp("norm_mha"))?;
        let norm_ff = candle_nn::layer_norm(dim, 1e-12, vb.pp("norm_ff"))?;
        Ok(Self {
            self_attn,
            feed_forward,
            norm_mha,
            norm_ff,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x_norm = self.norm_mha.forward(x)?;
        let x_attn = self.self_attn.forward(&x_norm, pos_emb, None)?;
        let x = (x_attn + residual)?;
        let residual = &x;
        let x_norm = self.norm_ff.forward(&x)?;
        let x_ff = self.feed_forward.forward(&x_norm)?;
        x_ff + residual
    }
}

struct LinearNoSubsampling {
    out: Vec<Box<dyn Module>>,
}

impl LinearNoSubsampling {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(input_dim, output_dim, vb.pp("0"))?;
        let ln = candle_nn::layer_norm(output_dim, 1e-5, vb.pp("1"))?;
        Ok(Self {
            out: vec![Box::new(linear), Box::new(ln)],
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for module in &self.out {
            x = module.forward(&x)?;
        }
        Ok(x)
    }
}

struct PreLookaheadLayer {
    conv1: Conv1d,
    conv2: Conv1d,
    pre_lookahead_len: usize,
}

impl PreLookaheadLayer {
    fn new(channels: usize, pre_lookahead_len: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv1d(
            channels,
            channels,
            pre_lookahead_len + 1,
            Default::default(),
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(channels, channels, 3, Default::default(), vb.pp("conv2"))?;
        Ok(Self {
            conv1,
            conv2,
            pre_lookahead_len,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_in = x.clone();
        let mut h = x.transpose(1, 2)?;
        h = h.pad_with_zeros(2, 0, self.pre_lookahead_len)?;
        h = self.conv1.forward(&h)?.gelu()?;
        h = h.pad_with_zeros(2, 2, 0)?;
        h = self.conv2.forward(&h)?;
        let h = h.transpose(1, 2)?;
        h + x_in
    }
}

struct Upsample1D {
    conv: Option<Conv1d>,
    stride: usize,
}

impl Upsample1D {
    fn new(channels: usize, out_channels: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let conv = candle_nn::conv1d(
            channels,
            out_channels,
            stride * 2 + 1,
            Default::default(),
            vb.pp("conv"),
        )?;
        Ok(Self {
            conv: Some(conv),
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let x = x
            .unsqueeze(3)?
            .repeat((1, 1, 1, self.stride))?
            .reshape((b, c, t * self.stride))?;
        if let Some(conv) = &self.conv {
            let x = x.pad_with_zeros(2, self.stride * 2, 0)?;
            conv.forward(&x)
        } else {
            Ok(x)
        }
    }
}

pub struct UpsampleConformerEncoder {
    embed: LinearNoSubsampling,
    pre_lookahead: PreLookaheadLayer,
    encoders: Vec<ConformerLayer>,
    up_layer: Upsample1D,
    up_embed: LinearNoSubsampling,
    up_encoders: Vec<ConformerLayer>,
    pe: EspnetRelPositionalEncoding,
    after_norm: LayerNorm,
}

impl UpsampleConformerEncoder {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed = LinearNoSubsampling::new(input_dim, output_dim, vb.pp("embed").pp("out"))?;
        let pre_lookahead = PreLookaheadLayer::new(output_dim, 3, vb.pp("pre_lookahead_layer"))?;
        let mut encoders = Vec::new();
        for i in 0..num_layers {
            encoders.push(ConformerLayer::new(
                output_dim,
                hidden_dim,
                num_heads,
                vb.pp("encoders").pp(i),
            )?);
        }
        let up_layer = Upsample1D::new(output_dim, output_dim, 2, vb.pp("up_layer"))?;
        let up_embed =
            LinearNoSubsampling::new(output_dim, output_dim, vb.pp("up_embed").pp("out"))?;
        let mut up_encoders = Vec::new();
        for i in 0..4 {
            up_encoders.push(ConformerLayer::new(
                output_dim,
                hidden_dim,
                num_heads,
                vb.pp("up_encoders").pp(i),
            )?);
        }
        let pe = EspnetRelPositionalEncoding::new(output_dim, 5000)?;
        let after_norm = candle_nn::layer_norm(output_dim, 1e-5, vb.pp("after_norm"))?;
        Ok(Self {
            embed,
            pre_lookahead,
            encoders,
            up_layer,
            up_embed,
            up_encoders,
            after_norm,
            pe,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embed.forward(x)?;
        let x = self.pre_lookahead.forward(&x)?;
        let (mut x, pos_emb) = self.pe.forward(&x)?;
        for layer in &self.encoders {
            x = layer.forward(&x, &pos_emb)?;
        }
        let x_up = self
            .up_layer
            .forward(&x.transpose(1, 2)?)?
            .transpose(1, 2)?;
        let x = self.up_embed.forward(&x_up)?;
        let (mut x, pos_emb_up) = self.pe.forward(&x)?;
        for layer in &self.up_encoders {
            x = layer.forward(&x, &pos_emb_up)?;
        }
        self.after_norm.forward(&x)
    }
}

pub struct ConditionalDecoder {
    sinusoidal_pos_emb: SinusoidalPosEmb,
    time_embedding: TimestepEmbedding,
    down_blocks: Vec<(
        CausalResnetBlock1D,
        Vec<BasicTransformerBlock>,
        Option<CausalConv1d>,
    )>,
    mid_blocks: Vec<(CausalResnetBlock1D, Vec<BasicTransformerBlock>)>,
    up_blocks: Vec<(
        CausalResnetBlock1D,
        Vec<BasicTransformerBlock>,
        Option<Upsample1D>,
    )>,
    final_block: CausalBlock1D,
    final_proj: Conv1d,
    pub meanflow: bool,
    _time_mixer: Option<Linear>,
    spk_emb_dim: usize,
}

impl ConditionalDecoder {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        spk_emb_dim: usize,
        vb: VarBuilder,
        meanflow: bool,
    ) -> Result<Self> {
        let channels = vec![256];
        let time_dim = channels[0] * 4;
        let attention_head_dim = 64;
        let num_heads = 8;
        let n_blocks = 4;

        let sinusoidal_pos_emb = SinusoidalPosEmb::new(in_channels);
        let time_embedding = TimestepEmbedding::new(in_channels, time_dim, vb.pp("time_mlp"))?;

        let mut down_blocks = Vec::new();
        let resnet =
            CausalResnetBlock1D::new(in_channels, channels[0], time_dim, vb.pp("down_blocks.0.0"))?;
        let mut transformers = Vec::new();
        for i in 0..n_blocks {
            transformers.push(BasicTransformerBlock::new(
                channels[0],
                num_heads,
                attention_head_dim,
                vb.pp("down_blocks.0.1").pp(i),
            )?);
        }
        let downsample =
            CausalConv1d::new(channels[0], channels[0], 3, 1, 1, vb.pp("down_blocks.0.2"))?;
        down_blocks.push((resnet, transformers, Some(downsample)));

        let mut mid_blocks = Vec::new();
        for i in 0..12 {
            let resnet = CausalResnetBlock1D::new(
                channels[0],
                channels[0],
                time_dim,
                vb.pp("mid_blocks").pp(i).pp("0"),
            )?;
            let mut transformers = Vec::new();
            for j in 0..n_blocks {
                transformers.push(BasicTransformerBlock::new(
                    channels[0],
                    num_heads,
                    attention_head_dim,
                    vb.pp("mid_blocks").pp(i).pp("1").pp(j),
                )?);
            }
            mid_blocks.push((resnet, transformers));
        }

        let mut up_blocks = Vec::new();
        let resnet = CausalResnetBlock1D::new(
            channels[0] * 2,
            channels[0],
            time_dim,
            vb.pp("up_blocks.0").pp("0"),
        )?;
        let mut transformers = Vec::new();
        for j in 0..n_blocks {
            transformers.push(BasicTransformerBlock::new(
                channels[0],
                num_heads,
                attention_head_dim,
                vb.pp("up_blocks.0").pp("1").pp(j),
            )?);
        }
        let upsample_conv = CausalConv1d::new(
            channels[0],
            channels[0],
            3,
            1,
            1,
            vb.pp("up_blocks.0").pp("2"),
        )?;
        let upsample = Upsample1D {
            conv: Some(upsample_conv.conv),
            stride: 1,
        };
        up_blocks.push((resnet, transformers, Some(upsample)));

        let final_block = CausalBlock1D::new(channels[0], channels[0], vb.pp("final_block"))?;
        let final_proj = candle_nn::conv1d(
            channels[0],
            out_channels,
            1,
            Default::default(),
            vb.pp("final_proj"),
        )?;
        let time_mixer = if meanflow {
            Some(candle_nn::linear_no_bias(
                time_dim * 2,
                time_dim,
                vb.pp("time_embed_mixer"),
            )?)
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

    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
        r: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut t_emb = self
            .time_embedding
            .forward(&self.sinusoidal_pos_emb.forward(t)?)?;

        if self.meanflow {
            if let Some(r) = r {
                let r_emb = self
                    .time_embedding
                    .forward(&self.sinusoidal_pos_emb.forward(r)?)?;
                let concat = Tensor::cat(&[&t_emb, &r_emb], 1)?;
                if let Some(mixer) = &self._time_mixer {
                    t_emb = mixer.forward(&concat)?;
                }
            }
        }
        let (b, _, t_len) = x.dims3()?;

        let spk = match spks {
            Some(s) => s.unsqueeze(2)?.repeat((1, 1, t_len))?,
            None => Tensor::zeros((b, self.spk_emb_dim, t_len), x.dtype(), x.device())?,
        };
        let c_tensor = match cond {
            Some(c) => {
                if c.dim(2)? == 1 {
                    c.repeat((1, 1, t_len))?
                } else {
                    c.clone()
                }
            }
            None => Tensor::zeros((b, 80, t_len), x.dtype(), x.device())?,
        };
        let mut current_x = Tensor::cat(&[x, mu, &spk, &c_tensor], 1)?;

        let mut hiddens = Vec::new();
        let mut masks = vec![mask.clone()];

        for (resnet, transformers, downsample) in &self.down_blocks {
            let m = masks.last().unwrap();
            current_x = resnet.forward(&current_x, m, &t_emb)?;
            let mut xt = current_x.transpose(1, 2)?;
            for tf in transformers {
                xt = tf.forward(&xt, None)?;
            }
            current_x = xt.transpose(1, 2)?;
            hiddens.push(current_x.clone());
            if let Some(ds) = downsample {
                current_x = ds.forward(&current_x)?;
                let t_len = current_x.dim(2)?;
                masks.push(m.narrow(2, 0, t_len)?);
            }
        }

        let m_mid = masks.last().unwrap();
        for (resnet, transformers) in &self.mid_blocks {
            current_x = resnet.forward(&current_x, m_mid, &t_emb)?;
            let mut xt = current_x.transpose(1, 2)?;
            for tf in transformers {
                xt = tf.forward(&xt, None)?;
            }
            current_x = xt.transpose(1, 2)?;
        }

        for (resnet, transformers, upsample) in &self.up_blocks {
            masks.pop();
            let m_up = masks.last().unwrap();
            let skip = hiddens.pop().unwrap();

            // Match python slicing: x[:, :, :skip.shape[-1]]
            let skip_len = skip.dim(2)?;
            if current_x.dim(2)? > skip_len {
                current_x = current_x.narrow(2, 0, skip_len)?;
            }
            let m_up = m_up.narrow(2, 0, skip_len)?; // Also narrow the mask

            current_x = Tensor::cat(&[&current_x, &skip], 1)?;
            current_x = resnet.forward(&current_x, &m_up, &t_emb)?;
            let mut xt = current_x.transpose(1, 2)?;
            for tf in transformers {
                xt = tf.forward(&xt, None)?;
            }
            current_x = xt.transpose(1, 2)?;

            if let Some(us) = upsample {
                current_x = us.forward(&current_x)?;
            }
        }

        let m_final = masks.pop().unwrap();
        current_x = self.final_block.forward(&current_x, &m_final)?;
        self.final_proj.forward(&current_x)
    }
}

pub struct CausalConditionalCFM {
    estimator: ConditionalDecoder,
    mel_dim: usize,
}

impl CausalConditionalCFM {
    pub fn new(
        mel_dim: usize,
        _cond_dim: usize,
        spk_emb_dim: usize,
        vb: VarBuilder,
        meanflow: bool,
    ) -> Result<Self> {
        let estimator =
            ConditionalDecoder::new(320, mel_dim, spk_emb_dim, vb.pp("estimator"), meanflow)?;
        Ok(Self { estimator, mel_dim })
    }

    pub fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        spks: Option<&Tensor>,
        cond: Option<&Tensor>,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        let (b, _c, t) = mu.dims3()?;
        let mut x = Tensor::randn(0f32, 1f32, (b, self.mel_dim, t), mu.device())?;
        let dt = 1.0 / n_timesteps as f64;
        for i in 0..n_timesteps {
            let t_val = i as f64 * dt;
            let r_val = t_val + dt;
            let t_tensor = Tensor::new(&[t_val as f32], mu.device())?.repeat(b)?;
            let r_tensor = Tensor::new(&[r_val as f32], mu.device())?.repeat(b)?;
            let dxdt =
                self.estimator
                    .forward(&x, mask, mu, &t_tensor, spks, cond, Some(&r_tensor))?;
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
    pub campplus: crate::campplus::CAMPPlus,
    hifigan: Option<crate::hifigan::HiFTGenerator>,
}

impl S3Gen {
    pub fn new(vb: VarBuilder, meanflow: bool) -> Result<Self> {
        let vocab_size = 6561;
        let hidden_dim = 512;
        let vb_flow = vb.pp("flow");
        let embedding =
            candle_nn::embedding(vocab_size, hidden_dim, vb_flow.pp("input_embedding"))?;
        let encoder = UpsampleConformerEncoder::new(
            hidden_dim,
            hidden_dim,
            2048,
            6,
            8,
            vb_flow.pp("encoder"),
        )?;
        let mu_proj = candle_nn::linear(hidden_dim, 80, vb_flow.pp("encoder_proj"))?;
        let decoder = CausalConditionalCFM::new(80, 80, 80, vb_flow.pp("decoder"), meanflow)?;
        let campplus = crate::campplus::CAMPPlus::new(80, 192, vb.pp("speaker_encoder"))?;

        // Try to load HiFTGenerator vocoder (mel2wav)
        // Config matches Python: sampling_rate=24000, upsample_rates=[8,5,3]
        let hifigan = {
            let config = crate::hifigan::HiFTConfig {
                in_channels: 80,
                base_channels: 512,
                nb_harmonics: 8,
                sampling_rate: 24000,
                upsample_rates: vec![8, 5, 3],
                upsample_kernel_sizes: vec![16, 11, 7],
                resblock_kernel_sizes: vec![3, 7, 11],
                resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
                n_fft: 16,
                hop_len: 4,
            };
            match crate::hifigan::HiFTGenerator::new(config, vb.pp("mel2wav")) {
                Ok(h) => {
                    eprintln!("[S3Gen] HiFTGenerator (mel2wav) loaded successfully");
                    Some(h)
                }
                Err(e) => {
                    // HiFiGAN weights might not be available - output raw mel
                    eprintln!("[S3Gen] HiFTGenerator failed to load: {:?}", e);
                    eprintln!("[S3Gen] WARNING: Will output raw mel spectrograms (static noise)");
                    None
                }
            }
        };

        Ok(Self {
            embedding,
            encoder,
            mu_proj,
            decoder,
            campplus,
            hifigan,
        })
    }

    /// Forward pass: converts speech tokens to audio waveform
    /// Returns mel spectrogram if HiFiGAN not available, otherwise returns audio
    pub fn forward(&self, speech_tokens: &Tensor, spks: Option<&Tensor>) -> Result<Tensor> {
        eprintln!("[S3Gen::forward] START");
        eprintln!(
            "[S3Gen::forward] speech_tokens shape: {:?}",
            speech_tokens.dims()
        );
        if let Some(s) = spks {
            eprintln!("[S3Gen::forward] spks shape: {:?}", s.dims());
        } else {
            eprintln!("[S3Gen::forward] spks: None");
        }

        let embeds = self.embedding.forward(speech_tokens)?;
        eprintln!("[S3Gen::forward] embeds shape: {:?}", embeds.dims());

        let encoder_out = self.encoder.forward(&embeds)?;
        eprintln!(
            "[S3Gen::forward] encoder_out shape: {:?}",
            encoder_out.dims()
        );

        let mu = self.mu_proj.forward(&encoder_out)?.transpose(1, 2)?;
        eprintln!("[S3Gen::forward] mu (transposed) shape: {:?}", mu.dims());

        let n_steps = if self.decoder.estimator.meanflow {
            eprintln!("[S3Gen::forward] Using meanflow=true, n_steps=2");
            2
        } else {
            eprintln!("[S3Gen::forward] Using meanflow=false, n_steps=32");
            32
        };

        let (b, _, t) = mu.dims3()?;
        eprintln!("[S3Gen::forward] batch={}, time_steps={}", b, t);

        let mask = Tensor::ones((b, 1, t), mu.dtype(), mu.device())?;
        eprintln!("[S3Gen::forward] Calling CFM decoder...");

        let mel = self.decoder.forward(&mu, &mask, spks, Some(&mu), n_steps)?;
        eprintln!("[S3Gen::forward] mel output shape: {:?}", mel.dims());

        // Check mel stats
        let mel_flat = mel.flatten_all()?;
        let mel_data: Vec<f32> = mel_flat.to_vec1()?;
        let mel_min = mel_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let mel_max = mel_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mel_mean = mel_data.iter().sum::<f32>() / mel_data.len() as f32;
        eprintln!(
            "[S3Gen::forward] mel stats: min={:.4}, max={:.4}, mean={:.4}, len={}",
            mel_min,
            mel_max,
            mel_mean,
            mel_data.len()
        );

        // Convert mel to audio using HiFiGAN vocoder if available
        if let Some(ref hifigan) = self.hifigan {
            eprintln!("[S3Gen::forward] Running HiFiGAN vocoder...");
            let audio = hifigan.inference(&mel)?;
            eprintln!("[S3Gen::forward] audio output shape: {:?}", audio.dims());

            // Check audio stats
            let audio_flat = audio.flatten_all()?;
            let audio_data: Vec<f32> = audio_flat.to_vec1()?;
            let audio_min = audio_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let audio_max = audio_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let audio_mean = audio_data.iter().sum::<f32>() / audio_data.len() as f32;
            eprintln!(
                "[S3Gen::forward] audio stats: min={:.4}, max={:.4}, mean={:.4}, len={}",
                audio_min,
                audio_max,
                audio_mean,
                audio_data.len()
            );
            eprintln!("[S3Gen::forward] END (with HiFiGAN)");
            Ok(audio)
        } else {
            eprintln!("[S3Gen::forward] WARNING: No HiFiGAN - returning raw mel!");
            eprintln!("[S3Gen::forward] END (raw mel)");
            Ok(mel)
        }
    }

    /// Forward pass that explicitly returns mel spectrogram without vocoder
    pub fn forward_mel(&self, speech_tokens: &Tensor, spks: Option<&Tensor>) -> Result<Tensor> {
        let embeds = self.embedding.forward(speech_tokens)?;
        let encoder_out = self.encoder.forward(&embeds)?;
        let mu = self.mu_proj.forward(&encoder_out)?.transpose(1, 2)?;
        let n_steps = if self.decoder.estimator.meanflow {
            2
        } else {
            32
        };
        let (b, _, t) = mu.dims3()?;
        let mask = Tensor::ones((b, 1, t), mu.dtype(), mu.device())?;
        self.decoder.forward(&mu, &mask, spks, Some(&mu), n_steps)
    }
}
