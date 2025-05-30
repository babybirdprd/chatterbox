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
    from chatterbox.models.s3gen.decoder import ConditionalDecoder
except ModuleNotFoundError as e:
    print(f"Failed to import chatterbox modules. Ensure PYTHONPATH is set correctly.")
    print(f"Example: export PYTHONPATH=/path/to/project/src:$PYTHONPATH")
    print(f"Error: {e}")
    sys.exit(1)

def prepare_cfm_estimator_for_export():
    """
    Prepares the ConditionalDecoder (CFM Estimator) model and dummy inputs for ONNX export.
    The actual torch.onnx.export call is commented out.
    """
    print('Instantiating ConditionalDecoder (CFM Estimator) model...')
    
    # Parameters from S3Token2Mel in src/chatterbox/models/s3gen/s3gen.py
    estimator_params = {
        "in_channels_x": 320, # Matches 'in_channels' when estimator is created
        "in_channels_mu": 320, # Assuming mu conditioning has same channels as x for direct input
        "in_channels_cond": 512, # Example: dimensionality of token embeddings from encoder
        "channels": [256],
        "n_blocks": 4,
        "num_mid_blocks": 12,
        "attention_head_dim": 64,
        "dropout": 0.0,
        "n_feats": 80, # out_channels, also n_mels
        "spk_emb_dim": 80, # As used in CausalConditionalCFM
        "num_heads": 8,
        "act_fn": 'gelu',
        "causal": True, # As set in S3Token2Mel
    }
    # Note: ConditionalDecoder __init__ takes n_feats (for output projection)
    # and spk_emb_dim (for speaker embedding projection).
    # in_channels_x, in_channels_mu, in_channels_cond are for the different types of input tensors.
    
    estimator = ConditionalDecoder(
        in_channels_x=estimator_params["in_channels_x"],
        in_channels_mu=estimator_params["in_channels_mu"],
        in_channels_cond=estimator_params["in_channels_cond"],
        channels=estimator_params["channels"],
        n_blocks=estimator_params["n_blocks"],
        num_mid_blocks=estimator_params["num_mid_blocks"],
        attention_head_dim=estimator_params["attention_head_dim"],
        dropout=estimator_params["dropout"],
        n_feats=estimator_params["n_feats"], 
        spk_emb_dim=estimator_params["spk_emb_dim"],
        num_heads=estimator_params["num_heads"],
        act_fn=estimator_params["act_fn"],
        causal=estimator_params["causal"]
    )
    estimator.eval()
    print('ConditionalDecoder model instantiated successfully and set to eval mode.')

    # --- Prepare Dummy Inputs ---
    # Based on ConditionalDecoder.forward(self, x, mask, mu, t, spks, cond)
    print('Preparing dummy inputs...')
    batch_size = 1 
    t_mel = 50 # Example number of mel frames

    # x: (B, in_channels_x, T_mel)
    dummy_x = torch.randn(batch_size, estimator_params["in_channels_x"], t_mel)
    # mask: (B, 1, T_mel)
    dummy_mask = torch.ones(batch_size, 1, t_mel)
    # mu: (B, in_channels_mu, T_mel) - Main conditioning signal
    dummy_mu = torch.randn(batch_size, estimator_params["in_channels_mu"], t_mel)
    # t: (B,) or (B,1) - Time step embeddings (usually normalized 0-1)
    dummy_t = torch.rand(batch_size) 
    # spks: (B, spk_emb_dim)
    dummy_spks = torch.randn(batch_size, estimator_params["spk_emb_dim"])
    # cond: (B, in_channels_cond, T_mel) - Other conditioning, e.g., token embeddings
    dummy_cond = torch.randn(batch_size, estimator_params["in_channels_cond"], t_mel)
    
    print(f'  Dummy x shape: {dummy_x.shape}')
    print(f'  Dummy mask shape: {dummy_mask.shape}')
    print(f'  Dummy mu shape: {dummy_mu.shape}')
    print(f'  Dummy t shape: {dummy_t.shape}')
    print(f'  Dummy spks shape: {dummy_spks.shape}')
    print(f'  Dummy cond shape: {dummy_cond.shape}')

    # The model to be exported is the estimator itself
    # Inputs for torch.onnx.export call
    traced_inputs = (dummy_x, dummy_mask, dummy_mu, dummy_t, dummy_spks, dummy_cond)

    # --- Define Export Parameters ---
    onnx_filepath = 'cfm_estimator.onnx' 
    input_names = ['x', 'mask', 'mu', 't', 'spks', 'cond']
    # ConditionalDecoder.forward returns the processed 'x'
    output_names = ['estimated_flow'] 
    
    dynamic_axes = {
        'x': {0: 'batch_size', 2: 'time_mel'}, 
        'mask': {0: 'batch_size', 2: 'time_mel'},
        'mu': {0: 'batch_size', 2: 'time_mel'},
        't': {0: 'batch_size'},
        'spks': {0: 'batch_size'},
        'cond': {0: 'batch_size', 2: 'time_mel'},
        'estimated_flow': {0: 'batch_size', 2: 'time_mel'}
    }
    opset_version = 14

    print(f"\n--- ONNX Export Parameters ---")
    print(f"  ONNX file path: {onnx_filepath}")
    print(f"  Input names: {input_names}")
    print(f"  Output names: {output_names}")
    print(f"  Dynamic axes: {dynamic_axes}")
    print(f"  Opset version: {opset_version}")
    print(f"  Model to export: ConditionalDecoder instance")
    print(f"  Traced input shapes:")
    for name, tensor in zip(input_names, traced_inputs):
        print(f"    {name}: {tensor.shape}")
    
    # --- Commented-Out Export Call ---
    # print(f'\nAttempting ONNX export to {onnx_filepath} (call is commented out)...')
    # try:
    #     torch.onnx.export(
    #         estimator,
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
    prepare_cfm_estimator_for_export()
