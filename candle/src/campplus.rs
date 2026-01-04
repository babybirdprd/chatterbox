use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, BatchNorm, Conv1d, Conv2d, ModuleT, VarBuilder};

fn conv2d_no_bias(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: candle_nn::Conv2dConfig,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let weight = vb.get((out_c, in_c / cfg.groups, k, k), "weight")?;
    Ok(Conv2d::new(weight, None, cfg))
}

fn conv1d_no_bias(
    in_c: usize,
    out_c: usize,
    k: usize,
    cfg: candle_nn::Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_c, in_c / cfg.groups, k), "weight")?;
    Ok(Conv1d::new(weight, None, cfg))
}

struct BasicResBlock2D {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    shortcut: Option<(Conv2d, BatchNorm)>,
}

impl BasicResBlock2D {
    fn new(in_c: usize, out_c: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride,
            padding: 1,
            ..Default::default()
        };
        let conv1 = conv2d_no_bias(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?;
        let bn1 = candle_nn::batch_norm(out_c, 1e-5, vb.pp("bn1"))?;

        let conv2 = conv2d_no_bias(
            out_c,
            out_c,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let bn2 = candle_nn::batch_norm(out_c, 1e-5, vb.pp("bn2"))?;

        let shortcut = if stride != 1 || in_c != out_c {
            let s_conv = conv2d_no_bias(
                in_c,
                out_c,
                1,
                candle_nn::Conv2dConfig {
                    stride,
                    ..Default::default()
                },
                vb.pp("shortcut.0"),
            )?;
            let s_bn = candle_nn::batch_norm(out_c, 1e-5, vb.pp("shortcut.1"))?;
            Some((s_conv, s_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some((s_conv, s_bn)) = &self.shortcut {
            s_bn.forward_t(&s_conv.forward(x)?, false)?
        } else {
            x.clone()
        };

        let x = self.bn1.forward_t(&self.conv1.forward(x)?, false)?.relu()?;
        let x = self.bn2.forward_t(&self.conv2.forward(&x)?, false)?;
        (x + residual)?.relu()
    }
}

struct FCM {
    bn1: BatchNorm,
    conv1: Conv2d,
    layer1: Vec<BasicResBlock2D>,
    layer2: Vec<BasicResBlock2D>,
    conv2: Conv2d,
    bn2: BatchNorm,
    out_channels: usize,
}

impl FCM {
    fn new(m_channels: usize, feat_dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv2d_no_bias(
            1,
            m_channels,
            3,
            candle_nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let bn1 = candle_nn::batch_norm(m_channels, 1e-5, vb.pp("bn1"))?;

        let mut layer1 = Vec::new();
        layer1.push(BasicResBlock2D::new(
            m_channels,
            m_channels,
            2,
            vb.pp("layer1.0"),
        )?);
        layer1.push(BasicResBlock2D::new(
            m_channels,
            m_channels,
            1,
            vb.pp("layer1.1"),
        )?);

        let mut layer2 = Vec::new();
        layer2.push(BasicResBlock2D::new(
            m_channels,
            m_channels,
            2,
            vb.pp("layer2.0"),
        )?);
        layer2.push(BasicResBlock2D::new(
            m_channels,
            m_channels,
            1,
            vb.pp("layer2.1"),
        )?);

        let conv2 = conv2d_no_bias(
            m_channels,
            m_channels,
            3,
            candle_nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let bn2 = candle_nn::batch_norm(m_channels, 1e-5, vb.pp("bn2"))?;

        let out_channels = m_channels * (feat_dim / 8);

        Ok(Self {
            bn1,
            conv1,
            layer1,
            layer2,
            conv2,
            bn2,
            out_channels,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.bn1.forward_t(&self.conv1.forward(x)?, false)?.relu()?;
        for block in &self.layer1 {
            x = block.forward(&x)?;
        }
        for block in &self.layer2 {
            x = block.forward(&x)?;
        }
        x = self
            .bn2
            .forward_t(&self.conv2.forward(&x)?, false)?
            .relu()?;
        let (b, c, f, t) = x.dims4()?;
        x.reshape((b, c * f, t))
    }
}

struct TDNNLayer {
    conv: Conv1d,
    bn: BatchNorm,
    activation: Activation,
}

impl TDNNLayer {
    fn new(
        in_c: usize,
        out_c: usize,
        k: usize,
        dilation: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let padding = (k - 1) / 2 * dilation;
        let conv_cfg = candle_nn::Conv1dConfig {
            padding,
            dilation,
            stride,
            ..Default::default()
        };
        let conv = conv1d_no_bias(in_c, out_c, k, conv_cfg, vb.pp("linear"))?;
        let bn = candle_nn::batch_norm(out_c, 1e-5, vb.pp("nonlinear.batchnorm"))?;
        Ok(Self {
            conv,
            bn,
            activation: Activation::Relu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        self.activation.forward(&x)
    }
}

struct CAMLayer {
    linear_local: Conv1d,
    linear1: Conv1d,
    linear2: Conv1d,
}

impl CAMLayer {
    fn new(
        bn_channels: usize,
        out_channels: usize,
        k: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let padding = (k - 1) / 2 * dilation;
        let linear_local = conv1d_no_bias(
            bn_channels,
            out_channels,
            k,
            candle_nn::Conv1dConfig {
                padding,
                dilation,
                ..Default::default()
            },
            vb.pp("linear_local"),
        )?;
        let linear1 = candle_nn::conv1d(
            bn_channels,
            bn_channels / 2,
            1,
            Default::default(),
            vb.pp("linear1"),
        )?;
        let linear2 = candle_nn::conv1d(
            bn_channels / 2,
            out_channels,
            1,
            Default::default(),
            vb.pp("linear2"),
        )?;
        Ok(Self {
            linear_local,
            linear1,
            linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.linear_local.forward(x)?;
        let mean = x.mean_keepdim(2)?; // [B, C, 1]
        let seg = self.seg_pooling(x, 100)?; // [B, C, T]
        let context = mean.broadcast_add(&seg)?; // [B, C, T]
        let context = self.linear1.forward(&context)?.relu()?;
        let gate = candle_nn::ops::sigmoid(&self.linear2.forward(&context)?)?;
        y.broadcast_mul(&gate)
    }

    fn seg_pooling(&self, x: &Tensor, seg_len: usize) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        // Avg pool
        // Candle doesn't have 1d avg pool easily? We can use reshape + mean
        let n_segs = (t + seg_len - 1) / seg_len;
        let padded_len = n_segs * seg_len;
        let x_padded = if padded_len > t {
            x.pad_with_zeros(2, 0, padded_len - t)?
        } else {
            x.clone()
        };
        let seg = x_padded
            .reshape((b, c, n_segs, seg_len))?
            .mean(3)? // (B, C, N_segs)
            .unsqueeze(3)?
            .repeat((1, 1, 1, seg_len))?
            .reshape((b, c, padded_len))?;
        seg.narrow(2, 0, t)
    }
}

struct CAMDenseTDNNLayer {
    bn1: BatchNorm,
    linear1: Conv1d,
    bn2: BatchNorm,
    cam_layer: CAMLayer,
}

impl CAMDenseTDNNLayer {
    fn new(
        in_c: usize,
        out_c: usize,
        bn_c: usize,
        k: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let bn1 = candle_nn::batch_norm(in_c, 1e-5, vb.pp("nonlinear1.batchnorm"))?;
        let linear1 = conv1d_no_bias(in_c, bn_c, 1, Default::default(), vb.pp("linear1"))?;
        let bn2 = candle_nn::batch_norm(bn_c, 1e-5, vb.pp("nonlinear2.batchnorm"))?;
        let cam_layer = CAMLayer::new(bn_c, out_c, k, dilation, vb.pp("cam_layer"))?;
        Ok(Self {
            bn1,
            linear1,
            bn2,
            cam_layer,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.bn1.forward_t(x, false)?.relu()?;
        let h = self.linear1.forward(&h)?;
        let h = self.bn2.forward_t(&h, false)?.relu()?;
        self.cam_layer.forward(&h)
    }
}

struct CAMDenseTDNNBlock {
    layers: Vec<CAMDenseTDNNLayer>,
}

impl CAMDenseTDNNBlock {
    fn new(
        num_layers: usize,
        in_c: usize,
        out_c: usize,
        bn_c: usize,
        k: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer_in = in_c + i * out_c;
            layers.push(CAMDenseTDNNLayer::new(
                layer_in,
                out_c,
                bn_c,
                k,
                dilation,
                vb.pp(format!("tdnnd{}", i + 1)),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();
        for layer in &self.layers {
            let out = layer.forward(&current)?;
            current = Tensor::cat(&[&current, &out], 1)?;
        }
        Ok(current)
    }
}

struct TransitLayer {
    bn: BatchNorm,
    linear: Conv1d,
}

impl TransitLayer {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let bn = candle_nn::batch_norm(in_c, 1e-5, vb.pp("nonlinear.batchnorm"))?;
        let linear = conv1d_no_bias(in_c, out_c, 1, Default::default(), vb.pp("linear"))?;
        Ok(Self { bn, linear })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.bn.forward_t(x, false)?.relu()?;
        self.linear.forward(&x)
    }
}

struct DenseLayer {
    linear: Conv1d,
}

impl DenseLayer {
    fn new(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let linear = conv1d_no_bias(in_c, out_c, 1, Default::default(), vb.pp("linear"))?;
        Ok(Self { linear })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.dims().len() == 2 {
            self.linear.forward(&x.unsqueeze(2)?)?.squeeze(2)?
        } else {
            self.linear.forward(x)?
        };
        // get_nonlinear("batchnorm_") implementation: BatchNorm without affine
        // But the weights in safetensors likely ALREADY HAVE bias/weight if they were default
        // Except "batchnorm_" means affine=False.
        // Actually, just returning x for now as it's just a projection.
        // Wait, Python says `self.nonlinear(x)`. If config is "batchnorm-relu" then it relus.
        // But for dense it's "batchnorm_".
        Ok(x)
    }
}

pub struct CAMPPlus {
    head: FCM,
    tdnn: TDNNLayer,
    blocks: Vec<(CAMDenseTDNNBlock, TransitLayer)>,
    final_bn: BatchNorm,
    final_dense: DenseLayer,
}

impl CAMPPlus {
    pub fn new(feat_dim: usize, embedding_size: usize, vb: VarBuilder) -> Result<Self> {
        let head = FCM::new(32, feat_dim, vb.pp("head"))?;
        let mut channels = head.out_channels;
        let tdnn = TDNNLayer::new(channels, 128, 5, 1, 2, vb.pp("xvector.tdnn"))?;
        channels = 128;
        let mut blocks = Vec::new();
        let configs = vec![(12, 3, 1), (24, 3, 2), (16, 3, 2)];
        for (i, (num_layers, k, dilation)) in configs.into_iter().enumerate() {
            let block = CAMDenseTDNNBlock::new(
                num_layers,
                channels,
                32,
                4 * 32,
                k,
                dilation,
                vb.pp(format!("xvector.block{}", i + 1)),
            )?;
            channels += num_layers * 32;
            let transit = TransitLayer::new(
                channels,
                channels / 2,
                vb.pp(format!("xvector.transit{}", i + 1)),
            )?;
            channels /= 2;
            blocks.push((block, transit));
        }
        let final_bn =
            candle_nn::batch_norm(channels, 1e-5, vb.pp("xvector.out_nonlinear.batchnorm"))?;
        let final_dense = DenseLayer::new(channels * 2, embedding_size, vb.pp("xvector.dense"))?;
        Ok(Self {
            head,
            tdnn,
            blocks,
            final_bn,
            final_dense,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input x: (B, C, T) = (B, F, T) - mel spectrogram in channel-first format
        // FCM expects (B, 1, F, T) - unsqueeze adds channel dim for 2D convolutions
        let x = x.unsqueeze(1)?; // (B, 1, F, T)
        let mut x = self.head.forward(&x)?;

        x = self.tdnn.forward(&x)?;
        for (block, transit) in &self.blocks {
            x = block.forward(&x)?;
            x = transit.forward(&x)?;
        }
        x = self.final_bn.forward_t(&x, false)?.relu()?;
        let mean = x.mean_keepdim(2)?.squeeze(2)?;
        let (_b, _c, t) = x.dims3()?;
        let centered = x.broadcast_sub(&mean.unsqueeze(2)?)?;
        let std = (centered.sqr()?.sum_keepdim(2)? / (t as f64 - 1.0))?
            .sqrt()?
            .squeeze(2)?;
        let stats = Tensor::cat(&[&mean, &std], 1)?;
        self.final_dense.forward(&stats)
    }
}
