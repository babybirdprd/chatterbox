import torch
import os
import sys

# --- PYTHONPATH Setup (Commented out, assuming external setup) ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..')) # Adjust if script is not in root
# src_path = os.path.join(project_root, 'src')
# if src_path not in sys.path:
#    sys.path.insert(0, src_path)

try:
    from chatterbox.models.s3gen.hifigan import HiFTGenerator
    from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
    # S3GEN_SR is used for sampling_rate, defined in chatterbox.models.s3gen.const
    from chatterbox.models.s3gen.const import S3GEN_SR # S3GEN_SR = 24000
except ModuleNotFoundError as e:
    print(f"Failed to import chatterbox modules. Ensure PYTHONPATH is set correctly.")
    print(f"Example: export PYTHONPATH=/path/to/project/src:$PYTHONPATH")
    print(f"Error: {e}")
    sys.exit(1)

def prepare_hifigan_for_export():
    """
    Prepares the HiFTGenerator model and dummy inputs for ONNX export.
    The actual torch.onnx.export call is commented out.
    """
    print('Instantiating HiFTGenerator model...')
    
    # Parameters based on S3Token2Wav.mel2wav in src/chatterbox/models/s3gen/s3gen.py
    # Default HiFTGenerator values:
    # in_channels=80, base_channels=256, num_blocks=3,
    # resblock_kernel_sizes=[3, 7, 11], resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    # source_num_blocks=2, source_resblock_kernel_sizes=[7, 7, 11], source_resblock_dilation_sizes=[[1, 1, 1], [1, 1, 1]],
    # leaky_relu_slope=0.1, use_weight_norm=True, use_spectral_norm=False
    
    # Values from s3gen.py for S3Token2Wav's mel2wav:
    hifigan_params_from_s3gen = {
        "sampling_rate": S3GEN_SR, # 24000
        "upsample_rates": [8, 5, 3],
        "upsample_kernel_sizes": [16, 11, 7],
        # The s3gen.py code passes 'source_resblock_kernel_sizes' and 'source_resblock_dilation_sizes'
        # to HiFTGenerator. These names match HiFTGenerator's __init__ arguments.
        "source_resblock_kernel_sizes": [7, 7, 11], # This is what s3gen.py uses
        "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]], # This is what s3gen.py uses
        # Other HiFTGenerator params will use their defaults if not specified in s3gen.py's instantiation
        # For example, resblock_kernel_sizes will use HiFTGenerator's default [3,7,11]
        # in_channels will be 80 (default, matching mel bins)
        # base_channels will be 256 (default)
    }

    f0_predictor = ConvRNNF0Predictor() # Default instantiation
    
    model = HiFTGenerator(
        sampling_rate=hifigan_params_from_s3gen["sampling_rate"],
        upsample_rates=hifigan_params_from_s3gen["upsample_rates"],
        upsample_kernel_sizes=hifigan_params_from_s3gen["upsample_kernel_sizes"],
        source_resblock_kernel_sizes=hifigan_params_from_s3gen["source_resblock_kernel_sizes"],
        source_resblock_dilation_sizes=hifigan_params_from_s3gen["source_resblock_dilation_sizes"],
        f0_predictor=f0_predictor
        # Other parameters like in_channels, base_channels, resblock_kernel_sizes, etc., use defaults from HiFTGenerator
    )
    model.eval()
    print('HiFTGenerator model instantiated successfully and set to eval mode.')

    # --- Prepare Dummy Inputs ---
    # Based on HiFTGenerator.inference(self, speech_feat, cache_source)
    print('Preparing dummy inputs...')
    batch_size = 1 
    num_mels = 80 # Standard mel bins, matches HiFTGenerator default in_channels
    t_mel = 50  # Example number of mel frames
    t_cache = 0 # For non-streaming export, cache is initially empty

    dummy_speech_feat = torch.randn(batch_size, num_mels, t_mel)
    dummy_cache_source = torch.zeros(batch_size, 1, t_cache) 
    
    print(f'  Dummy speech_feat shape: {dummy_speech_feat.shape}')
    print(f'  Dummy cache_source shape: {dummy_cache_source.shape}')

    # The model to be exported is the HiFTGenerator instance itself
    traced_inputs = (dummy_speech_feat, dummy_cache_source)

    # --- Define Export Parameters ---
    onnx_filepath = 'hifigan_vocoder.onnx' 
    input_names = ['speech_feat', 'cache_source']
    # HiFTGenerator.inference returns: wav, cache_source_out, f0_out
    output_names = ['output_audio', 'output_cache_source', 'output_f0']
    
    # Dynamic axes
    # T_audio_out depends on T_mel and upsample_rates. For T_mel=50, upsample_rates=[8,5,3], total_upsample = 120. T_audio_out = 50 * 120 = 6000
    # T_cache_out also depends on T_mel.
    # F0 output shape is (B, 1, T_mel)
    dynamic_axes = {
        'speech_feat': {0: 'batch_size', 2: 'time_mel'}, 
        'cache_source': {0: 'batch_size', 2: 'time_cache_in'},
        'output_audio': {0: 'batch_size', 1: 'time_audio_out'},
        'output_cache_source': {0: 'batch_size', 2: 'time_cache_out'},
        'output_f0': {0: 'batch_size', 2: 'time_mel'} # F0 is at mel_frame resolution
    }
    opset_version = 14

    print(f"\n--- ONNX Export Parameters ---")
    print(f"  ONNX file path: {onnx_filepath}")
    print(f"  Input names: {input_names}")
    print(f"  Output names: {output_names}")
    print(f"  Dynamic axes: {dynamic_axes}")
    print(f"  Opset version: {opset_version}")
    print(f"  Model to export: HiFTGenerator instance")
    print(f"  Traced input shapes:")
    for name, tensor in zip(input_names, traced_inputs):
        print(f"    {name}: {tensor.shape}")

    print(f"\n--- Notes on Custom Modules ---")
    print(f"  - Snake Activation: Used in ResBlocks. It's a torch.nn.Module composed of basic ops (x + (1/alpha) * sin(x*alpha)^2) and should be traceable by ONNX.")
    print(f"  - SineGen Module: Used in SourceModuleHnNSF. Its forward method uses torch.cumsum, torch.sin, and torch.randn for noise in unvoiced segments.")
    print(f"    The torch.randn call makes this part of the model non-deterministic. For a deterministic ONNX model, this noise generation might need to be controlled (e.g., by seeding, making noise an input, or disabling for export).")
    print(f"  - F0Predictor (ConvRNNF0Predictor): This is a standard PyTorch model and is traced as part of the HiFTGenerator graph.")

    # --- Commented-Out Export Call ---
    # print(f'\nAttempting ONNX export to {onnx_filepath} (call is commented out)...')
    # try:
    #     torch.onnx.export(
    #         model, # Direct model instance
    #         traced_inputs,
    #         onnx_filepath,
    #         input_names=input_names,
    #         output_names=output_names,
    #         dynamic_axes=dynamic_axes,
    #         opset_version=opset_version,
    #         verbose=True # Set to True for detailed ONNX conversion logs if uncommented
    #     )
    #     print(f"ONNX export call (commented out) would have targeted: {os.getcwd()}/{onnx_filepath}")
    # except Exception as e:
    #     print(f"Error during ONNX export (if it were run): {e}")
    #     import traceback
    #     traceback.print_exc()
    
    print(f"\nScript finished. To perform the actual export:")
    print(f"1. Ensure this script ('{os.path.basename(__file__)}') is executable or run with 'python {os.path.basename(__file__)}'.")
    print(f"2. Make sure PYTHONPATH includes the 'src' directory of the chatterbox project.")
    print(f"   Example: export PYTHONPATH=/path/to/your/chatterbox_project/src:$PYTHONPATH")
    print(f"3. Uncomment the 'torch.onnx.export(...)' section in this script.")
    print(f"4. Run the script.")

if __name__ == '__main__':
    prepare_hifigan_for_export()
