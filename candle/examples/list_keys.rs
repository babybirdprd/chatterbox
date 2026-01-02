use candle_core::Device;
use candle_nn::VarBuilder;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: list_keys <file_path>");
        return Ok(());
    }
    let path = PathBuf::from(&args[1]);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, &Device::Cpu)?
    };

    // VarBuilder doesn't provide a direct way to list all keys it *can* see,
    // but the underlying safetensors file does.
    let file = std::fs::File::open(&args[1])?;
    let mem = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let safetensors = safetensors::SafeTensors::deserialize(&mem)?;

    println!("File: {}", args[1]);
    let mut names: Vec<_> = safetensors.names().into_iter().collect();
    names.sort();
    for name in names {
        let view = safetensors.tensor(&name)?;
        println!("{}: {:?}", name, view.shape());
    }

    Ok(())
}
