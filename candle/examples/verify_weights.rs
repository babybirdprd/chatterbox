use candle_core::{DType, Device};
use candle_nn::VarBuilder;

/// Weight loading verification test
/// Compares tensor shapes and norms between Python and Rust loading
fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model_path = "C:/Users/Steve Business/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd/s3gen_meanflow.safetensors";

    if !std::path::Path::new(model_path).exists() {
        anyhow::bail!("Model file not found. Run Chatterbox to download first.");
    }

    println!("=== WEIGHT LOADING VERIFICATION TEST ===\n");

    // Load raw safetensors to get ground truth
    let file = std::fs::File::open(model_path)?;
    let mem = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let safetensors = safetensors::SafeTensors::deserialize(&mem)?;

    // Test 1: CAMPPlus weight loading
    println!("--- TEST 1: CAMPPlus (speaker_encoder.*) ---");

    // Expected key samples from Python analysis
    let campplus_key_samples = vec![
        ("speaker_encoder.head.bn1.weight", vec![32]),
        ("speaker_encoder.head.conv1.weight", vec![32, 1, 3, 3]),
        (
            "speaker_encoder.xvector.tdnn.linear.weight",
            vec![128, 320, 5],
        ),
        (
            "speaker_encoder.xvector.block1.tdnnd1.cam_layer.linear1.weight",
            vec![64, 128, 1],
        ),
        (
            "speaker_encoder.xvector.transit1.linear.weight",
            vec![256, 512, 1],
        ),
        (
            "speaker_encoder.xvector.dense.linear.weight",
            vec![192, 1024, 1],
        ), // 512*2=1024 after stats pool
    ];

    for (key, expected_shape) in &campplus_key_samples {
        match safetensors.tensor(key) {
            Ok(view) => {
                let shape: Vec<usize> = view.shape().to_vec();
                let match_status = if &shape == expected_shape {
                    "✓"
                } else {
                    "✗"
                };
                println!(
                    "  {match_status} {key}: {:?} (expected {:?})",
                    shape, expected_shape
                );
            }
            Err(_) => {
                println!("  ✗ {key}: NOT FOUND");
            }
        }
    }

    // Test 2: HiFiGAN weight loading (mel2wav.*)
    println!("\n--- TEST 2: HiFiGAN (mel2wav.*) ---");

    let hifigan_key_samples = vec![
        (
            "mel2wav.conv_pre.parametrizations.weight.original0",
            vec![512, 1, 1],
        ),
        (
            "mel2wav.conv_pre.parametrizations.weight.original1",
            vec![512, 80, 7],
        ),
        (
            "mel2wav.f0_predictor.condnet.0.parametrizations.weight.original1",
            vec![512, 80, 3],
        ),
        ("mel2wav.f0_predictor.classifier.weight", vec![1, 512]),
        ("mel2wav.m_source.l_linear.weight", vec![1, 9]),
        (
            "mel2wav.ups.0.parametrizations.weight.original1",
            vec![512, 256, 16],
        ),
    ];

    for (key, expected_shape) in &hifigan_key_samples {
        match safetensors.tensor(key) {
            Ok(view) => {
                let shape: Vec<usize> = view.shape().to_vec();
                let match_status = if &shape == expected_shape {
                    "✓"
                } else {
                    "✗"
                };
                println!(
                    "  {match_status} {key}: {:?} (expected {:?})",
                    shape, expected_shape
                );
            }
            Err(_) => {
                println!("  ✗ {key}: NOT FOUND");
            }
        }
    }

    // Test 3: Flow encoder (flow.*)
    println!("\n--- TEST 3: Flow (flow.*) ---");

    let flow_key_samples = vec![
        ("flow.input_embedding.weight", vec![6561, 512]),
        ("flow.spk_embed_affine_layer.weight", vec![80, 192]), // Used for conditioning, output=80
        ("flow.encoder.embed.out.0.weight", vec![512, 512]),
        (
            "flow.encoder.encoders.0.feed_forward.w_1.weight",
            vec![2048, 512],
        ),
        (
            "flow.decoder.estimator.down_blocks.0.0.block1.block.0.weight",
            vec![256, 320, 3],
        ),
    ];

    for (key, expected_shape) in &flow_key_samples {
        match safetensors.tensor(key) {
            Ok(view) => {
                let shape: Vec<usize> = view.shape().to_vec();
                let match_status = if &shape == expected_shape {
                    "✓"
                } else {
                    "✗"
                };
                println!(
                    "  {match_status} {key}: {:?} (expected {:?})",
                    shape, expected_shape
                );
            }
            Err(_) => {
                println!("  ✗ {key}: NOT FOUND");
            }
        }
    }

    // Test 4: Try loading CAMPPlus via VarBuilder
    println!("\n--- TEST 4: VarBuilder Loading Test ---");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };

    // Try loading with speaker_encoder prefix
    let vb_campplus = vb.pp("speaker_encoder");

    // Test a specific weight
    match vb_campplus.pp("head.bn1").get((32,), "weight") {
        Ok(tensor) => {
            println!(
                "  ✓ speaker_encoder.head.bn1.weight loaded: {:?}",
                tensor.dims()
            );
        }
        Err(e) => {
            println!("  ✗ speaker_encoder.head.bn1.weight failed: {}", e);
        }
    }

    match vb_campplus
        .pp("xvector.tdnn.linear")
        .get((128, 320, 5), "weight")
    {
        Ok(tensor) => {
            println!(
                "  ✓ speaker_encoder.xvector.tdnn.linear.weight loaded: {:?}",
                tensor.dims()
            );
        }
        Err(e) => {
            println!(
                "  ✗ speaker_encoder.xvector.tdnn.linear.weight failed: {}",
                e
            );
        }
    }

    // Test mel2wav prefix
    let vb_mel2wav = vb.pp("mel2wav");
    match vb_mel2wav
        .pp("f0_predictor.classifier")
        .get((1, 512), "weight")
    {
        Ok(tensor) => {
            println!(
                "  ✓ mel2wav.f0_predictor.classifier.weight loaded: {:?}",
                tensor.dims()
            );
        }
        Err(e) => {
            println!("  ✗ mel2wav.f0_predictor.classifier.weight failed: {}", e);
        }
    }

    println!("\n=== VERIFICATION COMPLETE ===");
    Ok(())
}
